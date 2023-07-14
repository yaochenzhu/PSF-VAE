import os
import time
import logging
import argparse
import itertools

import sys
sys.path.append("libs")

import numpy as np
import pandas as pd
import bottleneck as bn
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.decomposition import PCA

tf.get_logger().setLevel(logging.ERROR)
from train import get_collabo_vae

data_user_dict = {
	"ml-1m" : 6000,
	"amazon-vg" : 7253,
}

def dim_reduce(data, dimen):
    data = PCA(n_components=dimen).fit_transform(data)
    return data

# def dim_reduce(data, dim):
#     ori_dim = data.shape[-1]
#     idxes = np.random.choice(ori_dim, dim, replace=False)
#     idxes = np.sort(idxes)
#     return np.copy(data[:,idxes])

def get_expo_perc(rate_table):
    num_users = rate_table.uid.unique().size
    num_items = rate_table.uid.unique().size
    num_interactions = rate_table.size
    return num_interactions/(num_users*num_items)


def adj_rate_dist(rate_dist_raw, weighted=True):
    if weighted:
        base=1.3
        weights = np.array([base**i for i in range(5)][::-1])
    else:
        weights = np.array([1,1,1,1,1])
    rate_dist_raw = rate_dist_raw*weights
    return rate_dist_raw / rate_dist_raw.sum()


def get_rate_dist(rate_table):
    rate_dist_raw = np.array(
        rate_table.groupby("rating").count()["uid"].sort_index())
    return adj_rate_dist(rate_dist_raw)


def simulate():
    '''
        Basic usage:
            python simulate.py --dataset ml-1m
    '''
    ### Parse the console arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--sensitive_dim", type=int, default=50)
    parser.add_argument("--feature_dim", type=int, default=50)
    parser.add_argument("--bias_dim", type=int, default=50)
    parser.add_argument("--num_users" , type=int, default=-1)
    parser.add_argument("--device" , type=str, default="0")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device    

    ### Set up the tensorflow session.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    K.set_session(sess)

    ### Fix the random seeds.
    np.random.seed(98765)
    tf.set_random_seed(98765)

    if args.num_users == -1:
        num_users = data_user_dict[args.dataset]
    else:
        num_users = args.num_users

    ### Load rating data in real-world dataset
    rate_data_root = os.path.join("data", args.dataset, str(args.split), "exposure")
    meta_table = pd.read_csv(os.path.join(rate_data_root, "meta.csv"))
    num_items = meta_table["num_items"][0]
    rate_table = pd.read_csv(os.path.join(rate_data_root, "train.csv"))
    #expo_perc = get_expo_perc(rate_table)
    if args.dataset == "ml-1m":
        expo_perc = 0.04468
    else:
        expo_perc = 0.00406
    
    ### Load the pretrained rating model
    rate_model_root = os.path.join("models", args.dataset, str(args.split), "exposure")
    rate_vae = get_collabo_vae(args.dataset, num_items)
    rate_vae.load_weights(os.path.join(rate_model_root, "best.model"))
    obsv_rate_gen = rate_vae.build_vae_gen()

    latent_dim = obsv_rate_gen.input.shape.as_list()[-1]
    bias_dim = args.bias_dim
    fair_dim = latent_dim - bias_dim
    true_rate_gen, disc_gen = rate_vae.build_split_gens(dim1=fair_dim)
    
    ### Simulate the average number of exposed/discriminatory items
    expo_mean = expo_perc*num_items
    #### Estimated from real-world datasets
    if args.dataset == "amazon-vg":
        disc_mean = expo_mean//1.5
    else:
        disc_mean = expo_mean//6

    ### For testing the codes
    report_rate, fair_coeffs, bias_coeffs = [0.3], [0.9], [0.9]
    
    print("-"*5+"Statistics of the Real-world Dataset"+"-"*5)
    print("num_users: {}".format(num_users))
    print("num_items: {}".format(num_items))
    print("exposure percent: {}".format(expo_perc))

    print("-"*5+"Simulation Statistics"+"-"*5)
    print("fair dim: {}".format(fair_dim))
    print("bias dim: {}".format(bias_dim))
    print("average number of exposed items: {}".format(expo_mean))
    print("average number of discriminatory items: {}".format(disc_mean))
    
    for report_perc, fair_coeff, bias_coeff in \
        itertools.product(report_rate, fair_coeffs, bias_coeffs):
        
        simu_name = "rr{:.1f}-fc{:.1f}-bc{:.1f}".format(report_perc, fair_coeff, bias_coeff)
        print("-"*5+"Simulation {} starts!".format(simu_name)+"-"*5)
        
        # The confounder from which sensitive attributes are derived 
        confounder = np.random.randn(num_users, fair_dim)

        # Exogenous variables for U_fair and U_bias
        eps_fair = np.random.randn(num_users, fair_dim)
        eps_bias = np.random.randn(num_users, bias_dim)

        # User fair and bias embeddings
        U_fair = fair_coeff*confounder+np.sqrt(1-fair_coeff**2)*eps_fair
        bias_idx = np.sort(np.random.choice(fair_dim, bias_dim, replace=False))
        U_bias = bias_coeff*confounder[:, bias_idx]+np.sqrt(1-bias_coeff**2)*eps_bias

        # Get the raw obs/true ratings and discriminations
        U_concat = np.concatenate([U_fair, U_bias], axis=1)
        obsv_rate_raw = obsv_rate_gen.predict(U_concat)
        true_rate_raw = true_rate_gen.predict(U_fair)
        disc_raw = disc_gen.predict(U_bias)
        
        # Simulate the observed ratings
        obsv_rate_sorted = np.sort(obsv_rate_raw.reshape(-1))[::-1]
        rate_cut = obsv_rate_sorted[int(num_users*expo_mean)]
        obsv_rate_sim = (obsv_rate_raw>rate_cut).astype(np.int32)
        
        # Simulate the true ratings
        true_rate_sorted = np.sort(true_rate_raw.reshape(-1))[::-1]
        rate_cut = true_rate_sorted[int(num_users*expo_mean)]
        true_rate_sim = (true_rate_raw>rate_cut).astype(np.int32)
        
        # Simulate the discriminations
        disc_raw_sorted = np.sort(disc_raw.reshape(-1))[::-1]
        disc_cut = disc_raw_sorted[int(num_users*disc_mean)]
        disc_sim = (disc_raw>disc_cut).astype(np.int32)
        disc_sim_full = disc_sim
        noreport_user_idxes = np.random.choice(num_users, 
            int((1-report_perc)*num_users), replace=False)
        disc_sim_miss = np.copy(disc_sim_full)
        disc_sim_miss[noreport_user_idxes, :] = 0

        save_root = os.path.join("..", "psf-vae", "data", \
            args.dataset, simu_name, "raw")
        
        if not os.path.exists(save_root):
            os.makedirs(save_root)

        # Create a view of sensitive attributes and user features
        sensi_attrs_sim = dim_reduce(confounder, args.sensitive_dim)
        #sensi_attrs_sim = dim_reduce(U_concat, args.sensitive_dim)
        feats_sim = dim_reduce(eps_fair, args.feature_dim)

        np.save(os.path.join(save_root, "obsv_rate.npy"), obsv_rate_sim)
        np.save(os.path.join(save_root, "true_rate.npy"), true_rate_sim)
        np.save(os.path.join(save_root, "disc_full.npy"), disc_sim_full)
        np.save(os.path.join(save_root, "disc_miss.npy"), disc_sim_miss)
        np.save(os.path.join(save_root, "sensi_attrs.npy"), sensi_attrs_sim)
        np.save(os.path.join(save_root, "feats.npy"), feats_sim)

        print("Done simulation {}!".format(simu_name))


if __name__ == '__main__':
    simulate()