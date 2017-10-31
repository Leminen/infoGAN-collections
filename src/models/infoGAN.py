#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 14:01:52 2017

@author: leminen
"""
import os
import tensorflow as tf
import matplotlib.pyplot as plt

import src.data.process_dataset as process_dataset
import src.models.ops_util as ops
import src.utils as utils


class infoGAN(object):
    def __init__(self):
        self.model = 'infoGAN'
        self.dir_logs        = 'models/' + self.model + '/logs'
        self.dir_checkpoints = 'models/' + self.model + '/checkpoints'
        self.dir_results     = 'models/' + self.model + '/results'
        
        utils.checkfolder(self.dir_checkpoints)
        utils.checkfolder(self.dir_logs)
        utils.checkfolder(self.dir_results)
       
    def _create_inference(self):
        """ Define the inference model for the network
        Args:
    
        Returns:
        """
        ### self.output = f(self.input) ## define f
        

        # output of D for real images
        _ , self.D_real_logits, _ = self.__discriminator(self.inputImage, is_training=self.isTraining, reuse=False)
        # output of D for fake images
        self.img_fake = self.__generator(self.inputNoise, self.inputCode, is_training=self.isTraining, reuse=False)
        _ , self.D_fake_logits, input4classifier_fake = self.__discriminator(self.img_fake, is_training=self.isTraining, reuse=True)
        # output classifier branch
        self.code_est, self.code_logit_est = self.__classifier(input4classifier_fake, is_training=self.isTraining, reuse=False)
        
        
    def __discriminator(self, x, is_training = True, reuse = False):
        
        with tf.variable_scope('discriminator', reuse = reuse):
            net = ops.conv2d(x, 64, kernel_size = [4,4], stride = [2,2], scope ='d_conv1', activation_fn=ops.leaky_relu)
            net = ops.conv2d(net, 128, kernel_size = [4,4], stride = [2,2], scope ='d_conv2', bn = True, is_training = is_training, activation_fn=ops.leaky_relu)
            net = tf.reshape(net, [-1, 128*7*7])
            net = ops.fully_connected(net, 1024, scope='d_fc3', bn = True, is_training = is_training, activation_fn=ops.leaky_relu)
            out_logit = ops.fully_connected(net, 1, scope='d_fc4', activation_fn = None)
            out = tf.nn.sigmoid(out_logit)
            
            return out, out_logit, net
    
    def __classifier(self, x, is_training = True, reuse = False):
        
        with tf.variable_scope("classifier", reuse = reuse):
            
            net = ops.fully_connected(x, 64, scope='c_fc1', bn = True, is_training = is_training, activation_fn=ops.leaky_relu)
            out_logit = ops.fully_connected(net, 12, scope='c_fc2', activation_fn = None)
            out = tf.nn.sigmoid(out_logit)

            return out, out_logit
    
    def __generator(self, z, c, is_training=True, reuse=False):

        with tf.variable_scope("generator", reuse=reuse):

            # merge noise and code
            z = tf.concat([z, c], 1)
            
            net = ops.fully_connected(z, 1024, scope='g_fc1', bn = True, is_training = is_training)
            net = ops.fully_connected(net, 128 * 7 * 7, scope='g_fc2', bn = True, is_training = is_training)
            net = tf.reshape(net, [-1, 7, 7, 128])
            net = ops.conv2d_transpose(net, 64, kernel_size = [4,4], stride = [2,2], scope='g_dconv3', bn = True, is_training = is_training)
            net = ops.conv2d_transpose(net, 1, kernel_size = [4,4], stride = [2,2], scope='g_dconv4', activation_fn = None)
            out = tf.nn.sigmoid(net)

            return out
    
    
    
    def _create_losses(self):
        """ Define loss function[s] for the network
        Args:
    
        Returns:
        """
        ### self.loss = f(self.output, self.input) ## define f
        
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
        code_est_disc = self.code_logit_est[:,:10]
        code_target_disc = self.inputCode[:,:10]
        q_loss_disc = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=code_est_disc, labels=code_target_disc))

        # continuous code : gaussian
        code_est_cont = self.code_est[:,10:]
        code_target_cont = self.inputCode[:,10:]
        q_loss_cont = tf.reduce_mean(tf.reduce_sum(tf.square(code_target_cont - code_est_cont), axis=1))

        self.q_loss = q_loss_disc + q_loss_cont
        
    def _create_optimizer(self):
        """ Create optimizer for the network
        Args:
    
        Returns:
        """
        ### self.optimizer_op = f(self.loss) ## define f
        
        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]
        q_vars = [var for var in t_vars if ('d_' in var.name) or ('c_' in var.name) or ('g_' in var.name)]

        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_optimizer_op = tf.train.AdamOptimizer(learning_rate = 0.0002, beta1 = 0.5) \
                .minimize(self.d_loss, var_list=d_vars)
            self.g_optimizer_op = tf.train.AdamOptimizer(learning_rate = 0.0002, beta1 = 0.5) \
                .minimize(self.g_loss, var_list=g_vars)
            self.q_optimizer_op = tf.train.AdamOptimizer(learning_rate = 0.0002, beta1 = 0.5) \
                .minimize(self.q_loss, var_list=q_vars)
        
    def _create_summaries(self):
        """ Create summaries for the network
        Args:
    
        Returns:
        """
        
        ### Add summaries
        with tf.name_scope("summaries"):
            tf.summary.scalar('placeholderScalar', 1) # placeholder summary
            self.summary_op = tf.summary.merge_all()
        
        
    def train(self, dataset_str, epoch_N, batch_N):
        """ Run training of the network
        Args:
    
        Returns:
        """
        
        # Use dataset for loading in datasamples from .tfrecord (https://www.tensorflow.org/programmers_guide/datasets#consuming_tfrecord_data)
        # The iterator will get a new batch from the dataset each time a sess.run() is executed on the graph.
        filenames = tf.placeholder(tf.string, shape=[None])
        
        dataset = tf.contrib.data.TFRecordDataset(filenames)
        dataset = dataset.map(process_dataset._decodeData)      # decoding the tfrecord
        dataset = dataset.map(self._genLatentCodes)
        dataset = dataset.shuffle(buffer_size = 10000, seed = None)
        dataset = dataset.batch(batch_size = batch_N)
        iterator = dataset.make_initializable_iterator()
        self.input_getBatch = iterator.get_next()
        
        
        # Create input placeholders
        input_shape = [None, 28, 28, 1] # input image shape [batch_size, image_height, image_width, image_channels]
        self.inputImage = tf.placeholder(dtype = tf.float32, shape = input_shape, name = 'real_images')
        # Labels
        self.inputCode = tf.placeholder(dtype = tf.float32, shape = [None, 12], name = 'code_vector') # input code shape [batch_size, code_dim]
        # Noise
        self.inputNoise = tf.placeholder(dtype = tf.float32, shape = [None, 62], name = 'noise_vector') # input noise shape [batch_size, noise_dim]
        # Training flag
        self.isTraining = tf.placeholder(dtype = tf.bool, name = 'training_flag')
        
        # define model, loss, optimizer and summaries.
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
            
            counter = 0
            
            # Do training loops
            for epoch_n in range(epoch_start, epoch_N):
                
                training_filenames = ['data/processed/' + dataset_str + '/train.tfrecords'] # EXAMPLE
                sess.run(iterator.initializer, feed_dict={filenames: training_filenames})
                
                print('Running training epoch no: ', epoch_n)
                
                while True:
                    try:
                        imgs_batch, codes_batch, noise_batch = sess.run(self.input_getBatch)
                        
                        # update D network
                        _, summary = sess.run([self.d_optimizer_op, self.summary_op], 
                                              feed_dict={self.inputImage: imgs_batch, 
                                                         self.inputCode: codes_batch, 
                                                         self.inputNoise: noise_batch,
                                                         self.isTraining: True}
                                              )
                        
                        # update G and Q network
                        _, _, summary = sess.run([self.g_optimizer_op, self.q_optimizer_op, self.summary_op],
                                                 feed_dict={self.inputImage: imgs_batch, 
                                                            self.inputCode: codes_batch, 
                                                            self.inputNoise: noise_batch,
                                                            self.isTraining: True}
                                                 )
                    
                        writer.add_summary(summary, global_step=counter)
                        counter =+ 1
                        
                    except tf.errors.OutOfRangeError:
                        break
                
                if epoch_n % 1 == 0:
                    saver.save(sess,os.path.join(self.dir_checkpoints, self.model + '.model'), global_step=epoch_n)
                
                
                ### TEST of Input
#                for _ in range(10):
#                    inputs = sess.run(self.input)
#                            
#                    print('Label = ', inputs[0], 'Input Data Shape = ', inputs[1].shape, 'Plotting first image!')
#                    plt.imshow(inputs[1][0].squeeze())
#                    plt.show()
            
    
    def predict(self):
        """ Run prediction of the network
        Args:
    
        Returns:
        """
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
    


    def _genLatentCodes(self, image_proto, lbl_proto):
        image = image_proto
        
        code = tf.one_hot(lbl_proto,10)
        code = tf.concat([code, tf.random_uniform([2], minval = -1, maxval = 1)],0)
        
        noise = tf.random_uniform([62], minval = -1, maxval = 1)
        
        return image, code, noise