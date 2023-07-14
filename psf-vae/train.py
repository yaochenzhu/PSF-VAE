import os
import time
import logging
import argparse

import sys
sys.path.append(os.path.join("libs"))

from utils import PiecewiseSchedule
from evaluate import binary_crossentropy

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K

from data import DiscDataTrainValGenerator
from data import PSFairDataTrainValGenerator
from data import PSFairDataValTestGenerator
from model import PSFairVariationalAutoencoder

from evaluate import multinomial_crossentropy
from evaluate import EvaluateModel
from evaluate import mse, neg_mse
from evaluate import Recall_at_k, NDCG_at_k, HiR_at_k

import warnings
warnings.filterwarnings('ignore')

### Fix the random seeds.
np.random.seed(98765)
tf.set_random_seed(98765)

BASELINE_MODE = False
### set this to true if non-sensitive features are available.
use_feature = False
    
movielen_args = {
    "hidden_sizes":[], 
    "latent_size":200,
    "latent_size_disc":50,
    "hidden_sizes_disc":[50],
    "encoder_activs" : [],
    "decoder_activs" : ["softmax"],
    "encoder_activs_disc" : ["tanh"],
    "decoder_activs_disc" : ["tanh", "softmax"],
    "dropout_rate" : 0.5
}

amazon_args = {
    "hidden_sizes":[], 
    "latent_size":200,
    "latent_size_disc":50,
    "hidden_sizes_disc":[50],
    "encoder_activs" : [],
    "decoder_activs" : ["softmax"],
    "encoder_activs_disc" : ["tanh"],
    "decoder_activs_disc" : ["tanh", "softmax"],
    "dropout_rate" : 0.5
}

data_args_dict = {
    "ml-1m" : movielen_args,
    "amazon-vg" : amazon_args
}

def get_collabo_vae(dataset, input_dim_dict):
    get_collabo_vae = PSFairVariationalAutoencoder(
         input_dim_dict = input_dim_dict,
         **data_args_dict[dataset]
    )
    return get_collabo_vae

def train_vae_model():
    '''
        Basic usage:
            python train.py --dataset ml-1m --split 0
            python train.py --dataset amazon-vg --split 0
    '''
    ### Parse the console arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--rr", type=str, default="0.3")
    parser.add_argument("--fc", type=str, default="0.9")
    parser.add_argument("--bc", type=str, default="0.9")
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--device" , type=str, default="2")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    ### Set up the tensorflow session.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    K.set_session(sess)

    ### Get the train, val data generator for the model
    simu_name = "rr{}-fc{}-bc{}".format(args.rr, args.fc, args.bc)
    data_root = os.path.join("data", args.dataset, simu_name, str(args.split))
    if BASELINE_MODE:
        data_root = os.path.join("..", "..", data_root)
        
    print("Use non-sensitive user features: {}".format(use_feature))
    dvae_train_gen = DiscDataTrainValGenerator(
        data_root = data_root, phase="train",
        batch_size = args.batch_size, use_feature=use_feature,        
    )
    dvae_valid_gen = DiscDataTrainValGenerator(
        data_root = data_root, phase="val",
        batch_size = args.batch_size, use_feature=use_feature,        
    )   
    rvae_train_gen = PSFairDataTrainValGenerator(
        data_root = data_root, phase="train", mode="rvae_no_critic",
        batch_size = args.batch_size, use_feature=use_feature,
    )
    rvae_valid_gen = PSFairDataValTestGenerator(
        data_root = data_root, phase="val",
        batch_size = args.batch_size, use_feature=use_feature,
    )
    
    PSFair_vae = get_collabo_vae(args.dataset, input_dim_dict=rvae_train_gen.input_dim_dict)
    dvae_train = PSFair_vae.build_dvae_train()
    dvae_eval = PSFair_vae.build_dvae_eval()
    dvae_infer = PSFair_vae.build_dvae_infer()
    
    rvae_train_no_critic = PSFair_vae.build_rvae_train_no_critic()
    critic = PSFair_vae.build_critic()
    rvae_train_with_critic = PSFair_vae.build_rvae_train_with_critic()
    rvae_eval = PSFair_vae.build_rvae_eval()
    rvae_infer = PSFair_vae.build_rvae_infer()
    
    psfvae_train = PSFair_vae.build_psfvae_train()
    psfvae_eval = PSFair_vae.build_psfvae_eval()

    ### Train the U_b-specific part of the ELBO
    rec_loss = multinomial_crossentropy
    dvae_train.compile(loss=rec_loss, optimizer=optimizers.Adam(), metrics=[rec_loss])
    
    save_root = os.path.join("models", args.dataset, simu_name, str(args.split))
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    training_dynamics_dvae = os.path.join(save_root, "training_dynamics_dvae.csv")
    with open(training_dynamics_dvae, "w") as f:
        f.write("recall@20,NDCG@100\n")
    best_dvae_path = os.path.join(save_root, "best_dvae.model")

    recall_func = Recall_at_k; NDCG_func = NDCG_at_k; HiR_func = HiR_at_k
    best_dvae_recall, best_dvae_NDCG, best_dvae_sum = -np.inf, -np.inf, -np.inf
    lamb_schedule_gauss = PiecewiseSchedule([[0, 0.0], [80, 0.2]], outside_value=0.2)
    lr_schedule = PiecewiseSchedule([[0, 1e-3], [50, 1e-3], [51, 5e-4]], outside_value=5e-5)
    
    print("Training the U_b-specific part begins!")
    dvae_pre_epochs = 100
    for epoch in range(dvae_pre_epochs):
        K.set_value(dvae_train.optimizer.lr, lr_schedule.value(epoch))
        K.set_value(PSFair_vae.disc_add_gauss_loss.lamb_kl, lamb_schedule_gauss.value(epoch))       
        print("-"*10 + "Epoch:{}".format(epoch), "-"*10)
        dvae_train.fit_generator(dvae_train_gen, workers=4, epochs=1, validation_data=dvae_valid_gen)
        
        dvae_recall = EvaluateModel(dvae_eval, dvae_valid_gen, recall_func, k=20)
        dvae_NDCG = EvaluateModel(dvae_eval, dvae_valid_gen, NDCG_func, k=100)

        if dvae_recall > best_dvae_recall:
            best_dvae_recall = dvae_recall

        if dvae_NDCG > best_dvae_NDCG:
            best_dvae_NDCG = dvae_NDCG

        cur_sum = dvae_recall + dvae_NDCG
        if cur_sum > best_dvae_sum:
            best_dvae_sum = cur_sum
            ### The weights of vae_w/o_critic are tied togother
            dvae_train.save_weights(best_dvae_path, save_format="tf")

        with open(training_dynamics_dvae, "a") as f:
            f.write("{:.4f},{:.4f}\n".\
                format(dvae_recall, dvae_NDCG))

        print("-"*5+"Epoch: {}".format(epoch)+"-"*5)
        print("cur recall@20: {:5f}, best recall@20: {:5f}".format(dvae_recall, best_dvae_recall))
        print("cur NDCG@100: {:5f}, best NDCG@100: {:5f}".format(dvae_NDCG, best_dvae_NDCG))           
    print("Training the U_b-specific part ends!")
 
    ### Train the other part of the ELBO + adversarial loss
    PSFair_vae.load_weights_dvae(best_dvae_path)
    dvae_train.trainable = False
    rvae_train_gen.update_Ub(dvae_infer.predict(rvae_train_gen.get4dvae_infer()))

    rvae_train_no_critic.compile(loss=rec_loss, optimizer=optimizers.Adam(), metrics=[rec_loss])    
    training_dynamics_rvae = os.path.join(save_root, "training_dynamics_rvae.csv")
    with open(training_dynamics_rvae, "w") as f:
        f.write("recall@20,NDCG@100\n")
    best_rvae_path = os.path.join(save_root, "best_rvae.model")
    best_rvae_recall, best_rvae_NDCG, best_rvae_sum = -np.inf, -np.inf, -np.inf
    
    print("Fitting the observed biased ratings begins!")
    ### Pretrain the model with one step of the actor
    rvae_pre_epochs = 1
    rvae_train_gen.set_mode("rvae_no_critic")
    for epoch in range(rvae_pre_epochs):
        K.set_value(rvae_train_no_critic.optimizer.lr, lr_schedule.value(epoch))
        K.set_value(PSFair_vae.add_gauss_loss.lamb_kl, lamb_schedule_gauss.value(epoch))       
        print("-"*10 + "Epoch:{}".format(epoch), "-"*10)

        rvae_valid_gen.set_type("obsv")
        rvae_train_no_critic.fit_generator(rvae_train_gen, workers=4, epochs=1, validation_data=rvae_valid_gen)
        
        rvae_recall = EvaluateModel(rvae_eval, rvae_valid_gen, recall_func, k=20)
        rvae_NDCG = EvaluateModel(rvae_eval, rvae_valid_gen, NDCG_func, k=100)

        if rvae_recall > best_rvae_recall:
            best_rvae_recall = rvae_recall

        if rvae_NDCG > best_rvae_NDCG:
            best_rvae_NDCG = rvae_NDCG

        cur_sum = rvae_recall + rvae_NDCG
        if cur_sum > best_rvae_sum:
            best_rvae_sum = cur_sum
            ### The weights of vae_w/o_critic are tied togother
            rvae_train_no_critic.save_weights(best_rvae_path, save_format="tf")

        with open(training_dynamics_rvae, "a") as f:
            f.write("{:.4f},{:.4f}\n".\
                format(rvae_recall, rvae_NDCG))

        print("-"*5+"Epoch: {}".format(epoch)+"-"*5)
        print("cur recall@20: {:5f}, best recall@20: {:5f}".format(rvae_recall, best_rvae_recall))
        print("cur NDCG@100: {:5f}, best NDCG@100: {:5f}".format(rvae_NDCG, best_rvae_NDCG))
        
    ### Alternatively training the critic and the ELBO with dvae part fixed
    PSFair_vae.set_critic_dense(trainable=True)
    critic.compile(loss=mse, optimizer=optimizers.Adam(lr=1e-3))
    PSFair_vae.set_critic_dense(trainable=False)
    rvae_train_with_critic.compile(loss=[rec_loss, neg_mse], optimizer=optimizers.Adam(), 
                                   loss_weights=[1, 0.075], metrics=[[rec_loss],[]])
                
    epochs = 100
    critic_steps = 2
    for epoch in range(rvae_pre_epochs, epochs):
        print("-"*10 + "Epoch:{}".format(epoch), "-"*10)
        
        ### Infer the user latent variable
        rvae_train_gen.update_Uf(rvae_infer.predict(rvae_train_gen.get4rvae_infer()))
        rvae_train_gen.set_mode("critic")
        critic.fit_generator(rvae_train_gen, workers=4, epochs=critic_steps)
        
        rvae_train_gen.set_mode("rvae_with_critic")
        rvae_valid_gen.set_type("obsv")
        K.set_value(rvae_train_with_critic.optimizer.lr, lr_schedule.value(epoch))
        K.set_value(PSFair_vae.add_gauss_loss.lamb_kl, lamb_schedule_gauss.value(epoch)) 
        rvae_train_with_critic.fit_generator(rvae_train_gen, workers=4, epochs=1)
        
        rvae_recall = EvaluateModel(rvae_eval, rvae_valid_gen, recall_func, k=20)
        rvae_NDCG = EvaluateModel(rvae_eval, rvae_valid_gen, NDCG_func, k=100)

        if rvae_recall > best_rvae_recall:
            best_rvae_recall = rvae_recall

        if rvae_NDCG > best_rvae_NDCG:
            best_rvae_NDCG = rvae_NDCG

        cur_sum = rvae_recall + rvae_NDCG
        if cur_sum > best_rvae_sum:
            best_rvae_sum = cur_sum
            ### The weights of vae_w/o_critic are tied togother
            rvae_train_no_critic.save_weights(best_rvae_path, save_format="tf")

        with open(training_dynamics_rvae, "a") as f:
            f.write("{:.4f},{:.4f}\n".\
                format(rvae_recall, rvae_NDCG))

        print("-"*5+"Epoch: {}".format(epoch)+"-"*5)
        print("cur recall@10: {:5f}, best recall@10: {:5f}".format(rvae_recall, best_rvae_recall))
        print("cur NDCG@100: {:5f}, best NDCG@100: {:5f}".format(rvae_NDCG, best_rvae_NDCG))
    print("Fitting observed biased ratings ends!")

    ### Train the prediction model with zero path-specific bias
    rvae_infer.trainable = False
    psfvae_train.compile(loss=rec_loss, optimizer=optimizers.Adam(), metrics=[rec_loss])
    
    rvae_train_gen.set_mode("psfvae")
    training_dynamics_psfvae = os.path.join(save_root, "training_dynamics_psfvae.csv")
    with open(training_dynamics_psfvae, "w") as f:
        f.write("recall@20,NDCG@100,HiR@10_disc\n")
    best_psfvae_path = os.path.join(save_root, "best_psfvae.model")
    best_psfvae_recall, best_psfvae_NDCG, best_psfvae_sum = -np.inf, -np.inf, -np.inf
    lr_schedule = PiecewiseSchedule([[0, 1e-3], [50, 1e-3], [51, 5e-4]], outside_value=5e-5)
            
    print("Training ps-fair model begins!")
    epochs = 100
    for epoch in range(epochs):
        print("-"*10 + "Epoch:{}".format(epoch), "-"*10)
        
        ### Infer the user latent variable
        K.set_value(psfvae_train.optimizer.lr, lr_schedule.value(epoch))
        psfvae_train.fit_generator(rvae_train_gen, workers=4, epochs=1)
        
        rvae_valid_gen.set_type("obsv")
        psfvae_recall = EvaluateModel(psfvae_eval, rvae_valid_gen, recall_func, k=20)
        psfvae_NDCG = EvaluateModel(psfvae_eval, rvae_valid_gen, NDCG_func, k=100)
        rvae_valid_gen.set_type("disc")
        psfvae_HiR = EvaluateModel(psfvae_eval, rvae_valid_gen, HiR_func, k=10)
        
        if psfvae_recall > best_psfvae_recall:
            best_psfvae_recall = psfvae_recall

        if psfvae_NDCG > best_psfvae_NDCG:
            best_psfvae_NDCG = psfvae_NDCG
 
        cur_sum = psfvae_recall + psfvae_NDCG - float(args.rr)*psfvae_HiR
        if cur_sum > best_psfvae_sum:
            best_psfvae_sum = cur_sum
            psfvae_train.save_weights(best_psfvae_path, save_format="tf")

        with open(training_dynamics_psfvae, "a") as f:
            f.write("{:.4f},{:.4f}, {:.4f}\n".\
                format(psfvae_recall, psfvae_NDCG, psfvae_HiR))

        print("-"*5+"Epoch: {}".format(epoch)+"-"*5)
        print("cur recall@20: {:5f}, best recall@20: {:5f}".format(psfvae_recall, best_psfvae_recall))
        print("cur NDCG@100: {:5f}, best NDCG@100: {:5f}".format(psfvae_NDCG, best_psfvae_NDCG))
        print("cur HiR@10 for unfair items: {}".format(psfvae_HiR))
    print("Training ps-fair model ends!")

if __name__ == '__main__':
    train_vae_model()