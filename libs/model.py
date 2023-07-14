from audioop import mul
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
from tensorflow.keras import backend as K
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.engine import network

from evaluate import mse
from layers import DenseForSparse
from layers import TransposedSharedDense
from layers import AddMSELoss
from layers import AddGaussianLoss, ReparameterizeGaussian
from layers import AddBernoulliLoss, ReparameterizeBernoulli


class MLP(network.Network):
    '''
        Multilayer Perceptron (MLP). 
    '''
    def __init__(self, 
                 hidden_sizes,
                 activations,
                 l2_normalize=True,
                 l2_dim=0,
                 dropout_rate=0.5,
                 **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.m_name = kwargs.get("name", None)

        self.dense_list = []
        self.dropout_rate = dropout_rate
        self.l2_normalize = l2_normalize

        for i, (size, activation) in enumerate(zip(hidden_sizes, activations)):
            self.dense_list.append(
                layers.Dense(size, activation=activation,
                    kernel_initializer='glorot_uniform',
                    bias_initializer='zeros',
                    name="mlp_dense_{}".format(i)
            ))      
        self.l2_dim = l2_dim

    def build(self, input_shapes):
        x_in = layers.Input(input_shapes[1:])
        if self.dropout_rate:
            if self.l2_dim > 0:
                r_in = layers.Lambda(lambda x: x[:, :self.l2_dim])(x_in)
                f_in = layers.Lambda(lambda x: x[:, self.l2_dim:])(x_in)
                r_in = layers.Lambda(tf.nn.l2_normalize, arguments={"axis":1})(r_in)
                r_in = layers.Dropout(self.dropout_rate)(r_in)
                h_out = layers.Concatenate(axis=-1)([r_in, f_in])
            else:
                h_out = x_in
                if self.l2_normalize:
                    h_out = layers.Lambda(tf.nn.l2_normalize, arguments={"axis":1})(h_out)
                h_out = layers.Dropout(self.dropout_rate)(h_out)
        else:
            h_out = x_in
            
        if len(self.dense_list) > 0:
            h_out = self.dense_list[0](h_out)
            for dense in self.dense_list[1:]:
                h_out = dense(h_out)
        self._init_graph_network(x_in, h_out, self.m_name)
        super(MLP, self).build(input_shapes)


class SamplerGaussian(network.Network):
    '''
        Sample from the variational Gaussian, and add its KL 
        with the prior to loss
    '''
    def __init__(self, **kwargs):
        super(SamplerGaussian, self).__init__(**kwargs)
        self.m_name = kwargs.get("name", None)
        self.rep_gauss = ReparameterizeGaussian()

    def build(self, input_shapes):
        mean = layers.Input(input_shapes[0][1:])
        std  = layers.Input(input_shapes[1][1:])
        sample  = self.rep_gauss([mean, std])
        self._init_graph_network([mean, std], [sample], name=self.m_name)
        super(SamplerGaussian, self).build(input_shapes)


class CollaborativeLatentCore(network.Network):
    '''
        The latent core for collaborative VAE
    '''
    def __init__(self,  
                 latent_size, 
                 **kwargs):
        super(CollaborativeLatentCore, self).__init__(**kwargs)
        self.m_name = kwargs.get("name", None)
        self.dense_mean = layers.Dense(latent_size, name="mean")
        self.dense_std  = layers.Dense(latent_size, name="logstd")
        self.clip = layers.Lambda(lambda x:K.clip(x, -20, 2), name="clip")
        self.exp = layers.Lambda(lambda x:K.exp(x), name="exp")
        self.z_sampler = SamplerGaussian(name="z_sampler")

    def build(self, input_shapes):
        x_in = layers.Input(input_shapes[1:])
        mean = self.dense_mean(x_in)
        std  = self.exp(self.clip(self.dense_std(x_in)))
        z_b = self.z_sampler([mean, std])
        self._init_graph_network([x_in], [mean, std, z_b], name=self.m_name)
        super(CollaborativeLatentCore, self).build(input_shapes)


class PSFairVariationalAutoencoder():
    '''
       	The collaborative variational autoencoder with unawareness
    '''
    def __init__(self, 
                 input_dim_dict,
                 hidden_sizes,
                 latent_size,
                 hidden_sizes_disc,
                 latent_size_disc,
                 encoder_activs,
                 decoder_activs,
                 encoder_activs_disc,
                 decoder_activs_disc,
                 dropout_rate=0.5):
        ### Get the necessary information of the model
        self.input_dim_dict = input_dim_dict
        ### Naive fair model need to disentangle sensi attr from U
        self.sensi_attr_dim = self.input_dim_dict["sensi_attrs"]
        ### Naive fair model can use non-sensitive user features
        self.use_feature = False
        if "features" in self.input_dim_dict:
            self.use_feature = True
            self.feature_dim = self.input_dim_dict["features"]
        self.rating_dim = self.input_dim_dict["ratings"]
        self.input_dim = self.rating_dim + self.sensi_attr_dim
        if self.use_feature:
            self.input_dim += self.feature_dims
        self.latent_size = latent_size
        self.latent_size_disc = latent_size_disc
        self.l2_dim = self.rating_dim
        
        ### Build the encoder-decoder auto-graphs
        self.encoder = MLP(hidden_sizes, activations=encoder_activs, l2_dim=self.l2_dim,
                           dropout_rate=dropout_rate, name="Encoder")
        self.encoder.build(input_shapes=[None, self.input_dim])
        self.latent_core = CollaborativeLatentCore(latent_size, name="Latent")
        self.decoder = MLP(hidden_sizes[::-1]+[self.rating_dim], activations=decoder_activs, 
                           l2_normalize=False, dropout_rate=None, name="Decoder")
        
        ### Discriminator model that predicts Ub with Uf and sensi_attrs
        self.critic_dense = layers.Dense(self.latent_size_disc, use_bias=False, name="Critic")

        ### Build the encoder-decoder auto-graph for unfair items
        self.disc_encoder = MLP(hidden_sizes_disc, activations=encoder_activs_disc, l2_dim=self.l2_dim,
                                dropout_rate=dropout_rate, name="EncoderDisc")
        self.disc_encoder.build(input_shapes=[None, self.input_dim])
        
        self.disc_latent_core = CollaborativeLatentCore(latent_size_disc, name="LatentDisc")
        self.disc_decoder = MLP(hidden_sizes_disc[::-1]+[self.rating_dim], activations=decoder_activs_disc, 
                                l2_normalize=False, dropout_rate=None, name="DecoderDisc")
        
        ### Build the path-specific fair prediction model
        self.psf_encoder = MLP(hidden_sizes, activations=encoder_activs, l2_dim=self.l2_dim,
                               dropout_rate=dropout_rate, name="EncoderPSFair")
        self.psf_encoder.build(input_shapes=[None, self.input_dim])
        self.psf_latent_core = CollaborativeLatentCore(latent_size, name="LatentPSFair")
        self.psf_decoder = MLP(hidden_sizes[::-1]+[self.rating_dim], activations=decoder_activs, 
                               l2_normalize=False, dropout_rate=None, name="DecoderPSFair")
        
    def set_decoder(self, trainable=True):
        self.decoder.trainable = trainable
        
    def set_critic_dense(self, trainable=True):
        self.critic_dense.trainable = trainable
    
    def build_rvae_train_no_critic(self):
        '''
            Get the training form of the rvae model without adversarial training
        '''
        if not hasattr(self, "rvae_train_no_critic"):
            r_in = layers.Input(shape=[self.rating_dim,], name="ratings")
            s_in = layers.Input(shape=[self.sensi_attr_dim,], name="sensitive_attributes")
            all_ins = [r_in, s_in]
            if self.use_feature:
                f_in = layers.Input(shape=[self.feature_dim,], name="features")
                all_ins.append(f_in)
            concat_in = layers.Concatenate(axis=-1)(all_ins)
            
            h_mid = self.encoder(concat_in)
            mu_f, std_f, z_f = self.latent_core(h_mid)
            h_mid = self.disc_encoder(concat_in)
            mu_b, std_b, z_b = self.disc_latent_core(h_mid)
            z = layers.Concatenate(axis=-1)([z_f, z_b])
            r_rec = self.decoder(z)
            self.rvae_train_no_critic = models.Model(inputs=all_ins, outputs=[r_rec])
            
            if not hasattr(self, "add_gauss_loss"):
                self.add_gauss_loss = AddGaussianLoss()
            kl_loss_f = self.add_gauss_loss([mu_f, std_f])
            kl_loss_b = self.add_gauss_loss([mu_b, std_b])
            kl_loss = kl_loss_f + kl_loss_b
            self.rvae_train_no_critic.add_metric(kl_loss, name='kl_loss', aggregation='mean')
            self.rvae_train_no_critic.add_loss(kl_loss)
            
            core_idxes = []
            for i, layer in enumerate(self.rvae_train_no_critic.layers):
                if type(layer) == CollaborativeLatentCore:
                    core_idxes.append(i)
            weights_reg_loss = 0
            for i in core_idxes:
                weights_reg_loss+= tf.nn.l2_loss(self.rvae_train_no_critic.layers[i].dense_mean.weights[0]) + \
                                   tf.nn.l2_loss(self.rvae_train_no_critic.layers[i].dense_mean.weights[1])
                                   
            self.rvae_train_no_critic.add_loss(lambda: 2e-4*weights_reg_loss)
            self.rvae_train_no_critic.add_metric(2e-4*weights_reg_loss, name='reg_loss', aggregation='mean')
        return self.rvae_train_no_critic
    
    def build_critic(self):
        '''
            Get the training form of the critic
        '''
        if not hasattr(self, "critic"):
            zf_in = layers.Input(shape=[self.latent_size,], name="user_embedding")
            s_in = layers.Input(shape=[self.sensi_attr_dim,], name="sensitive_attributes")
            concat_in = layers.Concatenate(axis=-1)([zf_in, s_in])
            zb_out = self.critic_dense(concat_in)
            self.critic = models.Model(inputs=[zf_in, s_in], outputs=zb_out)
        return self.critic
    
    def build_rvae_train_with_critic(self):
        '''
            Get the training form of the rvae model with adversarial training
        '''
        if not hasattr(self, "rvae_train_with_critic"):
            r_in = layers.Input(shape=[self.rating_dim,], name="ratings")
            s_in = layers.Input(shape=[self.sensi_attr_dim,], name="sensitive_attributes")
            all_ins = [r_in, s_in]
            if self.use_feature:
                f_in = layers.Input(shape=[self.feature_dim,], name="features")
                all_ins.append(f_in)
            concat_in = layers.Concatenate(axis=-1)(all_ins)
            
            h_mid = self.encoder(concat_in)
            mu_f, std_f, z_f = self.latent_core(h_mid)
            h_mid = self.disc_encoder(concat_in)
            mu_b, std_b, z_b = self.disc_latent_core(h_mid)
            zb_pred = self.critic([mu_f, s_in])
            z = layers.Concatenate(axis=-1)([z_f, z_b])
            r_rec = self.decoder(z)
            self.rvae_train_with_critic = models.Model(inputs=all_ins, outputs=[r_rec, zb_pred])
            
            kl_loss_f = self.add_gauss_loss([mu_f, std_f])
            kl_loss_b = self.add_gauss_loss([mu_b, std_b])
            kl_loss = kl_loss_f + kl_loss_b
            self.rvae_train_with_critic.add_metric(kl_loss, name='kl_loss', aggregation='mean')
            self.rvae_train_with_critic.add_loss(kl_loss)
            
            core_idxes = []
            for i, layer in enumerate(self.rvae_train_with_critic.layers):
                if type(layer) == CollaborativeLatentCore:
                    core_idxes.append(i)
            weights_reg_loss = 0
            for i in core_idxes:
                weights_reg_loss+= tf.nn.l2_loss(self.rvae_train_with_critic.layers[i].dense_mean.weights[0]) + \
                                   tf.nn.l2_loss(self.rvae_train_with_critic.layers[i].dense_mean.weights[1])
            self.rvae_train_with_critic.add_loss(lambda: 2e-4*weights_reg_loss)
            self.rvae_train_with_critic.add_metric(2e-4*weights_reg_loss, name='reg_loss', aggregation='mean')
        return self.rvae_train_with_critic
    
    def build_rvae_eval(self):
        '''
            For evaluation, use the mean deterministically
        '''
        if not hasattr(self, "rvae_eval"):
            r_in = layers.Input(shape=[self.rating_dim,], name="ratings")
            s_in = layers.Input(shape=[self.sensi_attr_dim,], name="sensitive_attributes")
            all_ins = [r_in, s_in]
            if self.use_feature:
                f_in = layers.Input(shape=[self.feature_dim,], name="features")
                all_ins.append(f_in)
            concat_in = layers.Concatenate(axis=-1)(all_ins)
            h_mid = self.encoder(concat_in)
            mu_f, _, _ = self.latent_core(h_mid)
            h_mid = self.disc_encoder(concat_in)
            mu_b, _, _ = self.disc_latent_core(h_mid)
            mu = layers.Concatenate(axis=-1)([mu_f, mu_b])
            r_rec = self.decoder(mu)
            self.rvae_eval = models.Model(inputs=all_ins, outputs=r_rec)
        return self.rvae_eval
    
    def build_rvae_infer(self):
        '''
            To make inference of the user fair latent variables
        '''
        if not hasattr(self, "rvae_infer"):
            r_in = layers.Input(shape=[self.rating_dim,], name="ratings")
            s_in = layers.Input(shape=[self.sensi_attr_dim,], name="sensitive_attributes")
            all_ins = [r_in, s_in]
            if self.use_feature:
                f_in = layers.Input(shape=[self.feature_dim,], name="features")
                all_ins.append(f_in)
            concat_in = layers.Concatenate(axis=-1)(all_ins)
            h_mid = self.encoder(concat_in)
            mu_f, _, _ = self.latent_core(h_mid)
            self.rvae_infer = models.Model(inputs=all_ins, outputs=mu_f)
        return self.rvae_infer
    
    def build_dvae_train(self):
        '''
            Build the training model for discrimination
        '''
        if not hasattr(self, "dvae_train"):
            r_in = layers.Input(shape=[self.rating_dim,], name="ratings")
            s_in = layers.Input(shape=[self.sensi_attr_dim,], name="sensitive_attributes")
            all_ins = [r_in, s_in]
            if self.use_feature:
                f_in = layers.Input(shape=[self.feature_dim,], name="features")
                all_ins.append(f_in)
            concat_in = layers.Concatenate(axis=-1)(all_ins)
            h_mid = self.disc_encoder(concat_in)
            mu, std, z_b = self.disc_latent_core(h_mid)
            r_rec = self.disc_decoder(z_b)
            self.dvae_train = models.Model(inputs=all_ins, outputs=r_rec)
            
            self.disc_add_gauss_loss = AddGaussianLoss()
            kl_loss = self.disc_add_gauss_loss([mu, std])
            self.dvae_train.add_metric(kl_loss, name='kl_loss', aggregation='mean')
            self.dvae_train.add_loss(kl_loss)
            
            core_idx = None
            for i, layer in enumerate(self.dvae_train.layers):
                if type(layer) == CollaborativeLatentCore:
                    core_idx = i
            weights_reg_loss = tf.nn.l2_loss(self.dvae_train.layers[core_idx].dense_mean.weights[0]) + \
                               tf.nn.l2_loss(self.dvae_train.layers[core_idx].dense_mean.weights[1])
            self.dvae_train.add_loss(lambda: 2e-4*weights_reg_loss)
            self.dvae_train.add_metric(2e-4*weights_reg_loss, name='reg_loss', aggregation='mean')            
        return self.dvae_train
        
    def build_dvae_eval(self):
        '''
            For evaluation, use the mean deterministically 
        '''
        if not hasattr(self, "dvae_eval"):
            r_in = layers.Input(shape=[self.rating_dim,], name="ratings")
            s_in = layers.Input(shape=[self.sensi_attr_dim,], name="sensitive_attributes")
            all_ins = [r_in, s_in]
            if self.use_feature:
                f_in = layers.Input(shape=[self.feature_dim,], name="features")
                all_ins.append(f_in)
            concat_in = layers.Concatenate(axis=-1)(all_ins)
            h_mid = self.disc_encoder(concat_in)
            mu, _, _ = self.disc_latent_core(h_mid)
            r_rec = self.disc_decoder(mu)
            self.dvae_eval = models.Model(inputs=all_ins, outputs=r_rec)
        return self.dvae_eval
            
    def build_dvae_infer(self):
        '''
            To make inference of user bias latent variable
        '''
        if not hasattr(self, "dvae_infer"):
            r_in = layers.Input(shape=[self.rating_dim,], name="ratings")
            s_in = layers.Input(shape=[self.sensi_attr_dim,], name="sensitive_attributes")
            all_ins = [r_in, s_in]
            if self.use_feature:
                f_in = layers.Input(shape=[self.feature_dim,], name="features")
                all_ins.append(f_in)
            concat_in = layers.Concatenate(axis=-1)(all_ins)
            h_mid = self.disc_encoder(concat_in)
            mu, _, _ = self.disc_latent_core(h_mid)
            self.dvae_infer = models.Model(inputs=all_ins, outputs=mu)
        return self.dvae_infer
    
    def build_psfvae_train(self):
        '''
            The vae model with zero path-specific bias
        '''
        if not hasattr(self, "psfvae_train"):
            r_in = layers.Input(shape=[self.rating_dim,], name="ratings")
            s_in = layers.Input(shape=[self.sensi_attr_dim,], name="sensitive_attributes")
            all_ins = [r_in, s_in]
            if self.use_feature:
                f_in = layers.Input(shape=[self.feature_dim,], name="features")
                all_ins.append(f_in)
            concat_in = layers.Concatenate(axis=-1)(all_ins)
            h_mid = self.encoder(concat_in)
            _, _, z = self.latent_core(h_mid)
            r_rec = self.psf_decoder(z)            
            self.psfvae_train = models.Model(inputs=all_ins, outputs=r_rec)
        return self.psfvae_train
    
    def build_psfvae_eval(self):
        '''
            The vae model with zero path-specific bias
        '''
        if not hasattr(self, "psfvae_eval"):
            r_in = layers.Input(shape=[self.rating_dim,], name="ratings")
            s_in = layers.Input(shape=[self.sensi_attr_dim,], name="sensitive_attributes")
            all_ins = [r_in, s_in]
            if self.use_feature:
                f_in = layers.Input(shape=[self.feature_dim,], name="features")
                all_ins.append(f_in)
            concat_in = layers.Concatenate(axis=-1)(all_ins)
            h_mid = self.encoder(concat_in)
            mu, _, _ = self.latent_core(h_mid)
            r_rec = self.psf_decoder(mu)            
            self.psfvae_eval = models.Model(inputs=all_ins, outputs=r_rec)
        return self.psfvae_eval
       
    def load_weights_dvae(self, weight_path):
        '''
            Load weights from a pretrained dvae
        '''
        self.build_dvae_train()
        self.dvae_train.load_weights(weight_path)
        
    def load_weights_psfvae(self, weight_path):
        '''
            Load weights from a pretrained psf-vae
        '''
        self.build_psfvae_train()
        self.psfvae_train.load_weights(weight_path)
    
    def load_weights(self, weight_path):
        '''
            Load weights from a pretrained rvae
        '''
        self.build_rvae_train_no_critic()
        self.rvae_train_no_critic.load_weights(weight_path)


if __name__ == "__main__":
    pass