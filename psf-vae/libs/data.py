import os
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from tensorflow import keras

sys.path.append("libs")
from utils import load_mat

class DiscDataTrainValGenerator(keras.utils.Sequence):
    '''
        Generate the training and validation data for unfair items
    '''
    def __init__(self, 
                 data_root,
                 phase,
                 batch_size,
                 use_feature=False,
                 shuffle=True):
        self.phase = phase
        self.batch_size = batch_size
        self.use_feature = use_feature
        self.load_data(data_root)

        self.shuffle = shuffle
        if self.shuffle:
            self.on_epoch_end()
        
    def load_data(self, data_root):
        ### Load the rating and feature data
        obsv_rate_path = os.path.join(data_root, "{}_obsv_rate.npz".format(self.phase))
        obsv_disc_path = os.path.join(data_root, "{}_disc.npz".format(self.phase))
        self.obsv_rate = load_mat(obsv_rate_path, dense=True)
        self.obsv_disc = load_mat(obsv_disc_path, dense=True)
        self.remove_empty()
        
        self.num_users, self.num_items = self.obsv_rate.shape
        self.indexes = np.arange(self.num_users)
        self._input_dim_dict = {"ratings" : self.num_items}
                    
        sensi_attrs_path = os.path.join(data_root, "{}_sensi_attrs.npy".format(self.phase))
        self.sensi_attrs = load_mat(sensi_attrs_path)
        self._input_dim_dict["sensi_attrs"] = self.sensi_attrs.shape[-1]
        
        if self.use_feature:
            feat_path = os.path.join(data_root, "{}_feats.npy".format(self.phase))
            self.feats = load_mat(feat_path)
            self._input_dim_dict["features"] = self.feats.shape[-1]
        return "Successfully load the data!"
    
    def remove_empty(self):
        assert hasattr(self, "obsv_rate") & hasattr(self, "obsv_disc")
        nonempty_idxes = self.obsv_disc.sum(axis=1)>0
        self.obsv_rate = self.obsv_rate[nonempty_idxes]
        self.obsv_disc = self.obsv_disc[nonempty_idxes]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        batch_num = self.num_users//self.batch_size
        if self.num_users%self.batch_size != 0:
            batch_num+=1
        return batch_num      

    def __getitem__(self, i):
        batch_idxes = self.indexes[i*self.batch_size:(i+1)*self.batch_size]
        batch_X = [self.obsv_rate[batch_idxes], self.sensi_attrs[batch_idxes]]
        if self.use_feature:
            batch_X += [self.feats[batch_idxes]]
        batch_Y = self.obsv_disc[batch_idxes]
        return (batch_X, batch_Y)

    @property
    def input_dim_dict(self):
        return self._input_dim_dict


class PSFairDataTrainValGenerator(keras.utils.Sequence):
    '''
        Generate the training and validation data
    '''
    def __init__(self, 
                 data_root,
                 phase,
                 batch_size,
                 mode = "vae_no_critic",
                 use_feature=False,
                 shuffle=True):
        assert phase in ["train", "val"], "Invalid phase!"
        self.phase = phase
        self.mode_list = ["rvae_no_critic", "critic", "rvae_with_critic", "psfvae"]
        assert mode in self.mode_list
        self.mode = mode
        self.batch_size = batch_size
        self.use_feature = use_feature
        self.load_data(data_root)

        self.shuffle = shuffle
        if self.shuffle:
            self.on_epoch_end()
        
    def load_data(self, data_root):
        ### Load the rating and feature data
        if self.phase=="train":
            obsv_rate_path = os.path.join(data_root, "train_obsv_rate.npz")
            self.obsv_rate = load_mat(obsv_rate_path, dense=True)
            self.hout_rate = self.obsv_rate
        else:
            obsv_rate_path = os.path.join(data_root, "val_obsv_rate.npz")
            hout_rate_path = os.path.join(data_root, "val_hout_rate.npz")
            self.obsv_rate = load_mat(obsv_rate_path, dense=True)
            self.hout_rate = load_mat(hout_rate_path, dense=True)
        
        self.num_users, self.num_items = self.obsv_rate.shape
        self.indexes = np.arange(self.num_users)
        self._input_dim_dict = {"ratings" : self.num_items}
                    
        sensi_attrs_path = os.path.join(data_root, "{}_sensi_attrs.npy".format(self.phase))
        self.sensi_attrs = load_mat(sensi_attrs_path)
        self._input_dim_dict["sensi_attrs"] = self.sensi_attrs.shape[-1]
        
        if self.use_feature:
            feat_path = os.path.join(data_root, "{}_feats.npy".format(self.phase))
            self.feats = load_mat(feat_path)
            self._input_dim_dict["features"] = self.feats.shape[-1]
        return "Successfully load the data!"
    
    def set_mode(self, mode):
        assert mode in self.mode_list
        self.mode = mode
        
    def update_Ub(self, Ub):
        self.Ub = Ub
        
    def update_Uf(self, Uf):
        self.Uf = Uf

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def set_return_sensi(self, status):
        self.return_sensi_attr = status

    def __len__(self):
        batch_num = self.num_users//self.batch_size
        if self.num_users%self.batch_size != 0:
            batch_num+=1
        return batch_num
    
    def get4rvae_infer(self):
        X = [self.obsv_rate, self.sensi_attrs]
        if self.use_feature:
            X += [self.feats]
        return X

    def get4dvae_infer(self):
        X = [self.obsv_rate, self.sensi_attrs]
        if self.use_feature:
            X += [self.feats]
        return X       
    
    def get4rvae_no_critic(self, i):
        batch_idxes = self.indexes[i*self.batch_size:(i+1)*self.batch_size]
        batch_X = [self.obsv_rate[batch_idxes], self.sensi_attrs[batch_idxes]]
        if self.use_feature:
            batch_X += [self.feats[batch_idxes]]
        batch_Y = self.hout_rate[batch_idxes]
        return (batch_X, batch_Y)        
        
    def get4critic(self, i):
        assert hasattr(self, "Ub") & hasattr(self, "Uf")
        batch_idxes = self.indexes[i*self.batch_size:(i+1)*self.batch_size]
        batch_X = [self.Uf[batch_idxes], self.sensi_attrs[batch_idxes]]
        batch_Y = self.Ub[batch_idxes]
        return (batch_X, batch_Y)
        
    def get4rvae_with_critic(self, i):
        batch_idxes = self.indexes[i*self.batch_size:(i+1)*self.batch_size]
        batch_X = [self.obsv_rate[batch_idxes], self.sensi_attrs[batch_idxes]]
        if self.use_feature:
            batch_X += [self.feats[batch_idxes]]
        batch_Y = [self.hout_rate[batch_idxes],self.Ub[batch_idxes]]
        return (batch_X, batch_Y)
    
    def get4psfvae(self, i):
        #assert hasattr(self, "Uf")
        batch_idxes = self.indexes[i*self.batch_size:(i+1)*self.batch_size]
        batch_X = [self.obsv_rate[batch_idxes], self.sensi_attrs[batch_idxes]]
        if self.use_feature:
            batch_X += [self.feats[batch_idxes]]
        batch_Y = [self.hout_rate[batch_idxes]]
        return (batch_X, batch_Y) 

    def __getitem__(self, i):
        if self.mode == "rvae_no_critic":
            batch_X, batch_Y = self.get4rvae_no_critic(i)
        elif self.mode == "critic":
            batch_X, batch_Y = self.get4critic(i)
        elif self.mode == "rvae_with_critic":
            batch_X, batch_Y = self.get4rvae_with_critic(i)
        elif self.mode == "psfvae":
            batch_X, batch_Y = self.get4psfvae(i)
        return (batch_X, batch_Y)

    @property
    def input_dim_dict(self):
        return self._input_dim_dict
    

class PSFairDataValTestGenerator(keras.utils.Sequence):
    '''
        Generate the val/testing data
    '''
    def __init__(self, 
                 data_root,
                 phase,
                 batch_size,
                 use_feature=False,
                 shuffle=True):
        assert phase in ["val", "test"], "Invalid phase!"
        self.phase = phase
        self.batch_size = batch_size
        self.use_feature = use_feature
        self.load_data(data_root)

        self.shuffle = shuffle
        if self.shuffle:
            self.on_epoch_end()

        self.set_type("obsv")

    def load_data(self, data_root):
        ### Load the rating and feature data
        obsv_rate_path = os.path.join(data_root, "{}_obsv_rate.npz".format(self.phase))
        hout_rate_path = os.path.join(data_root, "{}_hout_rate.npz".format(self.phase))
        true_rate_path = os.path.join(data_root, "{}_true_rate.npy".format(self.phase))
        disc_path = os.path.join(data_root, "{}_true_disc.npy".format(self.phase)) 
        
        self.obsv_rate = load_mat(obsv_rate_path, dense=True)
        self.hout_rate = load_mat(hout_rate_path, dense=True)
        self.true_rate = load_mat(true_rate_path)
        self.disc = load_mat(disc_path)
        
        self.num_users, self.num_items = self.obsv_rate.shape
        self.indexes = np.arange(self.num_users)
        self._input_dim_dict = {"ratings" : self.num_items}

        sensi_attrs_path = os.path.join(data_root, "{}_sensi_attrs.npy".format(self.phase))
        self.sensi_attrs = load_mat(sensi_attrs_path)
        self._input_dim_dict["sensi_attrs"] = self.sensi_attrs.shape[-1]
        
        if self.use_feature:
            feat_path = os.path.join(data_root, "{}_feats.npy".format(self.phase))
            self.feats = load_mat(feat_path)
            self._input_dim_dict["features"] = self.feats.shape[-1]
        return "Successfully load the data!"
    
    def get4rvae_infer(self):
        X = [self.obsv_rate, self.sensi_attrs]
        if self.use_feature:
            X += [self.feats]
        return X
    
    def set_type(self, dtype):
        assert dtype in ["obsv", "true", "disc"]
        self.type=dtype
        
    def update_Uf(self, Uf):
        self.Uf = Uf

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        batch_num = self.num_users//self.batch_size
        if self.num_users%self.batch_size != 0:
            batch_num+=1
        return batch_num

    def __getitem__(self, i):
        batch_idxes = self.indexes[i*self.batch_size:(i+1)*self.batch_size]
        batch_X = [self.obsv_rate[batch_idxes], self.sensi_attrs[batch_idxes]]
        if self.use_feature:
            batch_X += [self.feats[batch_idxes]]
        if self.type == "obsv":
            batch_Y = self.hout_rate[batch_idxes]
        elif self.type == "true":
            batch_Y = self.true_rate[batch_idxes]
        elif self.type == "disc":
            batch_Y = self.disc[batch_idxes]
        return (batch_X, batch_Y)

    @property
    def input_dim_dict(self):
        return self._input_dim_dict

if __name__ == '__main__':
    pass