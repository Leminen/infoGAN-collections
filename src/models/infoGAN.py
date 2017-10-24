#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 16:55:23 2017

@author: leminen
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 16:43:52 2017

@author: leminen
"""
import os
import tensorflow as tf
import src.utils as utils
import src.models.tf_util as tf_util


class infoGAN(object):
    def __init__(self):
        self.model = 'BasicModel'
        self.dir_logs        = 'models/' + self.model + '/checkpoints'
        self.dir_checkpoints = 'models/' + self.model + '/checkpoints'
        self.dir_results     = 'models/' + self.model + '/checkpoints'
        
        utils.checkfolder(self.dir_checkpoints)
        utils.checkfolder(self.dir_logs)
        utils.checkfolder(self.dir_results)
       
    def _create_inference(self):
        """ Define the inference model for the network
        Args:
    
        Returns:
        """
        ### input placeholders:
        # Images
        input_shape = [batch_size, image_height, image_width, image_channels]
        self.inputs = tf.placeholder(dtype = tf.float32, shape = input_shape, name = 'real_images')
        # Labels
        self.c = tf.placeholder(dtype = tf.float32, shape = [batch_size, code_dim], name = 'code_vector')
        # Noise
        self.z = tf.placeholder(dtype = tf.float32, shape = [batch_size, noise_dim], name = 'noise_vector')
        
        ### outputs
        # output of D for real images
        _ , self.D_real_logits, _ = self.__discriminator(self.inputs, is_training=True, reuse=False)
        # output of D for fake images
        G = self.__generator(self.z, self.c, is_training=True, reuse=False)
        _ , self.D_fake_logits, input4classifier_fake = self.__discriminator(G, is_training=True, reuse=True)
        # output classifier branch
        self.code_fake, self.code_logit_fake = self.__classifier(input4classifier_fake, is_training=True, reuse=False)
        
        
    def __discriminator(self, x, is_training = True, reuse = False):
        
        batch_size = x.get_shape()[0].value
        
        with tf.variable_scope('discriminator', reuse = reuse):
            net = tf_util.conv2d(x, 64, kernel_size = [4,4], stride = [2,2], scope ='d_conv1', activation_fn=tf_util.leaky_relu)
            net = tf_util.conv2d(net, 128, kernel_size = [4,4], stride = [2,2], scope ='d_conv2', bn = True, is_training = is_training, activation_fn=tf_util.leaky_relu)
            net = tf.reshape(net, [batch_size, -1])
            net = tf_util.fully_connected(net, 1024, scope='d_fc3', bn = True, is_training = is_training, activation_fn=tf_util.leaky_relu)
            out_logit = tf_util.fully_connected(net, 1, scope='d_fc4', activation_fn = None)
            out = tf.nn.sigmoid(out_logit)
            
            return out, out_logit, net
    
    def __classifier(self, x, is_training = True, reuse = False):
        
        with tf.variable_scope("classifier", reuse = reuse):
            
            net = tf_util.fully_connected(x, 64, scope='c_fc1', bn = True, is_training = is_training, activation_fn=tf_util.leaky_relu)
            out_logit = tf_util.fully_connected(net, code_dim, scope='c_fc2', activation_fn = None)
            out = tf.nn.sigmoid(out_logit)

            return out, out_logit
    
    def __generator(self, z, c, is_training=True, reuse=False):
        
        batch_size = z.get_shape()[0].value

        with tf.variable_scope("generator", reuse=reuse):

            # merge noise and code
            z = tf.concat([z, c], 1)
            
            net = tf_util.fully_connected(x, 1024, scope='g_fc1', bn = True, is_training = is_training)
            net = tf_util.fully_connected(net, 128 * 7 * 7, scope='g_fc2', bn = True, is_training = is_training)
            net = tf.reshape(net, [batch_size, 7, 7, 128])
            net = tf_util.conv2d_transpose(net, 64, kernel_size = [4,4], stride = [2,2], scope='g_dconv3', bn = True, is_training = is_training)
            net = tf.util.conv2d_transpose(net, 1, kernel_size = [4,4], stride = [2,2], scope='g_dconv4', activation_fn = None)
            out = sigmoid(net)

            return out
    
    
    def _create_losses(self):
        """ Define loss function[s] for the network
        Args:
    
        Returns:
        """
        # Discriminator loss
        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real_logits, labels=tf.ones_like(self.D_real_logits)))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_logits, labels=tf.zeros_like(self.D_fake_logits)))

        self.d_loss = d_loss_real + d_loss_fake

        # Generator loss
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_logits, labels=tf.ones_like(self.D_fake_logits)))
        
        # Information loss
        # discrete code : categorical
        disc_code_est = self.code_logit_fake[:, :self.len_discrete_code]
        disc_code_tg = self.c[:, :self.len_discrete_code]
        q_disc_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=disc_code_est, labels=disc_code_tg))

        # continuous code : gaussian
        cont_code_est = self.code_fake[:, self.len_discrete_code:]
        cont_code_tg = self.c[:, self.len_discrete_code:]
        q_cont_loss = tf.reduce_mean(tf.reduce_sum(tf.square(cont_code_tg - cont_code_est), axis=1))

        self.q_loss = q_disc_loss + q_cont_loss
        
    def _create_optimizer(self):
        """ Create optimizer for the network
        Args:
    
        Returns:
        """
        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]
        q_vars = [var for var in t_vars if ('d_' in var.name) or ('c_' in var.name) or ('g_' in var.name)]

        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                .minimize(self.d_loss, var_list=d_vars)
            self.g_optim = tf.train.AdamOptimizer(self.learning_rate * 5, beta1=self.beta1) \
                .minimize(self.g_loss, var_list=g_vars)
            self.q_optim = tf.train.AdamOptimizer(self.learning_rate * 5, beta1=self.beta1) \
                .minimize(self.q_loss, var_list=q_vars)
        
    def _create_summaries(self):
        """ Create summaries for the network
        Args:
    
        Returns:
        """
        
        ### Add summaries
        
        self.summary_op = tf.summary.merge_all()
        
        
    def train(self):
        """ Run training of the network
        Args:
    
        Returns:
        """
        
        self._create_inference()
        self._create_losses()
        self._create_optimizer()
        self._create_summaries()
        
        
        with tf.Session() as sess:
            
            # Initialize all model Variables.
            sess.run(tf.global_variables_initializer())
            
            # Create Saver object for loading and storing checkpoints
            saver = tf.train.Saver()
            
            # Create Writer object for storing graph and summaries for TensorBoard
            writer = tf.summary.FileWriter(self.dir_logs, sess.graph)
            
            
            # Reload Tensor values from latest checkpoint
            ckpt = tf.train.get_checkpoint_state(self.dir_checkpoints)
            epoch_start = 0
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                epoch_start = int(ckpt_name.split('-')[-1])
            
            
            # Do training loops
            for epoch_n in range(epoch_start, epoch_N):
                for batch_n in range(batch_N):

                    ### _, summary = sess.run([self.optimizer_op, self.summary_op])
                    writer.add_summary(summary, global_step=((epoch_n * batch_N) + batch_n))
                
                
                if epoch_n % 1 == 0:
                    saver.save(sess,os.path.join(self.dir_checkpoints, self.model + '.model'), global_step=epoch_n)

                
            
            
    
    def predict(self):
        """ Run prediction of the network
        Args:
    
        Returns:
        """
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
    

