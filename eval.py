import os
import time
import logging
import argparse

import sys
sys.path.append(os.path.join("libs"))

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from data import PSFairDataValTestGenerator
from train import get_collabo_vae, use_feature

from evaluate import EvaluateModel
from evaluate import Recall_at_k, NDCG_at_k, HiR_at_k

import warnings
warnings.filterwarnings('ignore')

### Fix the random seeds.
np.random.seed(98765)
tf.set_random_seed(98765)

BASELINE_MODE = False

def predict_and_evaluate():
    ### Parse the console arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--rr", type=str, default="0.3")
    parser.add_argument("--fc", type=str, default="0.9")
    parser.add_argument("--bc", type=str, default="0.9")
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--device" , type=str, default="0")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    ### Set up the tensorflow session.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    K.set_session(sess)

    ### Get the test data generator
    simu_name = "rr{}-fc{}-bc{}".format(args.rr, args.fc, args.bc)
    data_root = os.path.join("data", args.dataset, simu_name, str(args.split))
    if BASELINE_MODE:
        data_root = os.path.join("..", "..", data_root)
        
    model_root = os.path.join("models", args.dataset, simu_name, str(args.split))

    test_gen = PSFairDataValTestGenerator(
        data_root = data_root, phase = "test", use_feature=use_feature,
        batch_size = args.batch_size, shuffle=False, 
    )

    ### Build test model and load trained weights
    collab_vae = get_collabo_vae(args.dataset, test_gen.input_dim_dict)
    collab_vae.load_weights_psfvae(os.path.join(model_root, "best_psfvae.model"))

    vae_eval = collab_vae.build_psfvae_eval()
    recall_func = Recall_at_k; NDCG_func = NDCG_at_k; HiR_func = HiR_at_k

    k4recalls = [10, 20, 50, 75, 100]
    k4ndcgs = [10, 20, 50, 75, 100]
    
    for dtype in ["obsv", "disc"]:
        ### Evaluate and save the results
        print("Currently evaluating {}".format(dtype))
        test_gen.set_type(dtype)
        recalls, NDCGs = [], []
        for k in k4recalls:
            if dtype == "disc":
                recalls.append("{:.4f}".format(EvaluateModel(vae_eval, test_gen, HiR_func, k=k)))
            else:
                recalls.append("{:.4f}".format(EvaluateModel(vae_eval, test_gen, recall_func, k=k)))
        for k in k4ndcgs:
            NDCGs.append("{:.4f}".format(EvaluateModel(vae_eval, test_gen, NDCG_func, k=k)))

        recall_table = pd.DataFrame({"k":k4recalls, "recalls":recalls}, columns=["k", "recalls"])
        recall_table.to_csv(os.path.join(model_root, "{}_recalls.csv".format(dtype)), index=False)

        ndcg_table = pd.DataFrame({"k":k4ndcgs, "NDCGs": NDCGs}, columns=["k", "NDCGs"])
        ndcg_table.to_csv(os.path.join(model_root, "{}_NDCGs.csv".format(dtype)), index=False)
        if dtype != "disc":
            print("recall@{} : {}; NDCG@{} : {}".format(k4recalls[1], recalls[1], k4ndcgs[-1], NDCGs[-1]))
        else:
            print("HiR@{} : {}".format(k4recalls[0], recalls[0]))
    print("Done evaluation! Results saved to {}".format(model_root))


if __name__ == '__main__':
    predict_and_evaluate()