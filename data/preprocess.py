import os
import sys
import random
import argparse

import numpy as np
import pandas as pd
from scipy import sparse

import warnings
warnings.filterwarnings('ignore')

random.seed(98765)
np.random.seed(98765)

def get_counts(raw_data, attr):
    counts_group = raw_data[[attr]].groupby(attr, as_index=False)
    counts = counts_group.size()
    return counts


def split_observed_unknown(data, source, unk_frac=0.2):
    data_group = data.groupby(source)
    obs_list, unk_list = [], []
    for i, (_, group) in enumerate(data_group):
        n_records = len(group)
        if n_records >= 5:
            idx = np.zeros(n_records, dtype='bool')
            idx[np.random.choice(n_records, size=max(int(unk_frac*n_records), 1), 
                                 replace=False).astype('int64')] = True
            obs_list.append(group[np.logical_not(idx)])
            unk_list.append(group[idx])
        else:
            obs_list.append(group)
        if i % 200 == 0:
            print("{} source sampled".format(i))
            sys.stdout.flush()
    data_obs = pd.concat(obs_list)
    data_unk = pd.concat(unk_list)
    return data_obs, data_unk


def split_ratings(ratings, holdout_rate, shape):
    ratings = sparse.coo_matrix(ratings)
    rating_table = pd.DataFrame(
        {"uid":ratings.row, "vid":ratings.col, "ratings":ratings.data},
        columns=["uid", "vid", "ratings"]
        )
    obsv_table, hout_table = split_observed_unknown(rating_table, "uid", holdout_rate)
    obsv_rate = sparse.csr_matrix((list(obsv_table.ratings),
        (list(obsv_table.uid), list(obsv_table.vid))), shape=shape)
    hout_rate = sparse.csr_matrix((list(hout_table.ratings),
        (list(hout_table.uid), list(hout_table.vid))), shape=shape)
    return obsv_rate, hout_rate


def save_mat(path, mat):
    if path.endswith(".npy"):
        np.save(path, mat)
    elif path.endswith(".npz"):
        sparse.save_npz(path, mat)
    else:
        raise NotImplementedError

if __name__ == '__main__':
    '''
        Usage:
            python preprocess --dataset ml-1m
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="specify the dataset")
    parser.add_argument("--test_rate", type=float, default=0.10,
        help="use which percent of user as test user")
    parser.add_argument("--holdout_rate", type=float, default=0.2,
        help="use which percent of exposure for prediction check")
    parser.add_argument("--rr", type=str, default="0.3",
        help="percentage of users who report discriminations")
    parser.add_argument("--fc", type=str, default="0.9",
        help="influence strength of S on u_f")
    parser.add_argument("--bc", type=str, default="0.9",
        help="influence strenght of S on u_i")
    parser.add_argument("--split", type=int, default=10,
        help="split the dataset into which number of train/val/test")
    args = parser.parse_args()

    simu_name = "rr{}-fc{}-bc{}".format(args.rr, args.fc, args.bc)
    raw_root = os.path.join(args.dataset, simu_name, "raw")
    assert os.path.exists(raw_root), "No raw dataset found in {}!".format(raw_root)
    
    obsv_rate = np.load(os.path.join(raw_root, "obsv_rate.npy"))
    true_rate = np.load(os.path.join(raw_root, "true_rate.npy"))
    disc_full = np.load(os.path.join(raw_root, "disc_full.npy"))
    disc_miss = np.load(os.path.join(raw_root, "disc_miss.npy"))
    feats = np.load(os.path.join(raw_root, "feats.npy"))
    sensi_attrs = np.load(os.path.join(raw_root, "sensi_attrs.npy"))

    num_users, num_items = obsv_rate.shape

    for split in range(args.split):
        idxes_perm = np.random.permutation(num_users)
        num_test  = int(args.test_rate*num_users)
        train_end = int(num_users - 2*num_test)
        val_end   = train_end + num_test
        assert train_end > 0, "the number of training samples should > 0"

        ### Get the indices for train/val/test users
        train_idxes = idxes_perm[:train_end]
        val_idxes = idxes_perm[train_end:val_end]
        test_idxes = idxes_perm[val_end:]

        ### Get the train/val/test observed rating data
        train_obsv_rate = sparse.csr_matrix(obsv_rate[train_idxes])
        val_obsv_rate = sparse.csr_matrix(obsv_rate[val_idxes])
        test_obsv_rate = sparse.csr_matrix(obsv_rate[test_idxes])
        
        ### Get the train/val/test discriminatory data
        train_disc = sparse.csr_matrix(disc_miss[train_idxes])
        val_disc = sparse.csr_matrix(disc_miss[val_idxes])
        test_disc = sparse.csr_matrix(disc_miss[test_idxes])

        ### Get fold-in and hold-out ratings for val/test users
        val_obsv_rate, val_hout_rate = split_ratings(\
            val_obsv_rate, args.holdout_rate, shape=[num_test, num_items])
        test_obsv_rate, test_hout_rate = split_ratings(\
            test_obsv_rate, args.holdout_rate, shape=[num_test, num_items])

        ### Get the true ratings/discriminiations of test users
        ### This is available only due to simulation
        ### In real-world dataset we cannot acquire this.
        val_true_rate = true_rate[val_idxes] 
        val_true_disc = disc_full[val_idxes]
        test_true_rate = true_rate[test_idxes]
        test_true_disc = disc_full[test_idxes]
                
        ### Save the split results of train/val/test rating data
        save_root = raw_root.split(os.path.sep)
        save_root[-1] = str(split)
        save_root = os.path.sep.join(save_root)
        if not os.path.exists(save_root):
            os.makedirs(save_root)
            
        ratings = [train_obsv_rate, val_obsv_rate, val_hout_rate,
                   val_true_rate, test_obsv_rate, test_hout_rate,
                   test_true_rate]
        names = ["train_obsv_rate.npz", "val_obsv_rate.npz", "val_hout_rate.npz",
                 "val_true_rate.npy", "test_obsv_rate.npz", "test_hout_rate.npz", 
                 "test_true_rate.npy"]
        for (name, rating) in zip(names, ratings):
            save_mat(os.path.join(save_root, name), rating)
        
        ### Save the split results of discriminatory data
        discs = [train_disc, val_disc, test_disc, val_true_disc, test_true_disc]
        names = ["train_disc.npz", "val_disc.npz", "test_disc.npz", 
                 "val_true_disc.npy", "test_true_disc.npy"]
        for (name, disc) in zip(names, discs):
            save_mat(os.path.join(save_root, name), disc)
        
        ### Split the user features and sensitive attributes
        train_feats, train_sensi_attrs = feats[train_idxes], sensi_attrs[train_idxes]
        val_feats, val_sensi_attrs = feats[val_idxes], sensi_attrs[val_idxes]
        test_feats, test_sensi_attrs = feats[test_idxes], sensi_attrs[test_idxes]
        
        ### Save the user features
        features = [train_feats, val_feats, test_feats]
        names = ["train_feats.npy", "val_feats.npy", "test_feats.npy"]
        for (name, feature) in zip(names, features):
            np.save(os.path.join(save_root, name), feature)
            
        ### Save the sensitive attributes
        sensitive_attributes = [train_sensi_attrs, val_sensi_attrs, test_sensi_attrs]
        names = ["train_sensi_attrs.npy", "val_sensi_attrs.npy", "test_sensi_attrs.npy"]
        for (name, sensitive_attribute) in zip(names, sensitive_attributes):
            np.save(os.path.join(save_root, name), sensitive_attribute)

        print("Done for {}".format(save_root))
         
    print("Done preprocessing the {} data!".format(args.dataset))