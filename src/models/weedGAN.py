#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 14:01:52 2017

@author: leminen
"""
import os
import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt

import src.data.process_dataset as process_dataset
import src.models.ops_util as ops
import src.utils as utils


class weedGAN(object):
    def __init__(self):
        self.model = 'weedGAN'
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
        """ Defines the Discriminator network model
        Args:
    
        Returns:
        """
        
        with tf.variable_scope('discriminator', reuse = reuse):
            # alexNet 
            net = ops.conv2d(inputImage, 96, kernel_size = [11,11], stride = [4,4], scope='d_conv1')
            net = ops.max_pool2d(net, kernel_size = [3,3], padding='VALID', scope = 'd_pool1')
            net = ops.conv2d(net, 256, kernel_size = [5,5], stride = [1,1], scope='d_conv2')
            net = ops.max_pool2d(net, kernel_size = [3,3], padding='VALID', scope = 'd_pool2')
            net = ops.conv2d(net, 384, kernel_size = [3,3], stride = [1,1], scope='d_conv3')
            net = ops.conv2d(net, 384, kernel_size = [3,3], stride = [1,1], scope='d_conv4')
            net = ops.conv2d(net, 256, kernel_size = [3,3], stride = [1,1], scope='d_conv5')
            net = ops.max_pool2d(net, kernel_size = [3,3], padding='VALID', scope = 'd_pool5')
            net = tf.reshape(net,[-1,6*6*256])
            net = ops.fully_connected(net, 4096, scope='d_fc6')
            net = ops.dropout(net, is_training = is_training, scope='d_drop6')
            net = ops.fully_connected(net, 4096, scope='d_fc7')
            net = ops.dropout(net, is_training = is_training, scope='d_drop7')
            out_logit = ops.fully_connected(net, 1, scope='d_fc8', activation_fn = None)
            out = tf.nn.sigmoid(out_logit)
            
            return out, out_logit, net
    
    def __classifier(self, x, is_training = True, reuse = False):
        """ Defines the Clasifier network model
        Args:
    
        Returns:
        """
        
        with tf.variable_scope("classifier", reuse = reuse):
            
            out_logit = ops.fully_connected(x, 12, scope='c_fc1', activation_fn = None)
            out = tf.nn.sigmoid(out_logit)

            return out, out_logit
    
    def __generator(self, z, c, is_training=True, reuse=False):
        """ Defines the Generator network model
        Args:
    
        Returns:
        """

        with tf.variable_scope("generator", reuse=reuse):

            # merge noise and code
            z = tf.concat([z, c], 1)
            net = ops.fully_connected(z, 4096, scope='g_fc1')
            net = ops.dropout(net, is_training = is_training, scope='g_drop1')
            net = ops.fully_connected(net, 4096, scope='g_fc2')
            net = ops.dropout(net, is_training = is_training, scope='g_drop2')
            net = ops.fully_connected(net, 7*7*256, scope='g_fc3')
            net = ops.dropout(net, is_training = is_training, scope='g_drop3')
            net = tf.reshape(net, [-1, 7, 7, 256])
            net = ops.conv2d_transpose(net, 384, kernel_size = [ 3, 3], stride = [2,2], scope='g_dconv4', bn = True, bn_decay=0.9, is_training = is_training)
            net = ops.conv2d_transpose(net, 384, kernel_size = [ 3, 3], stride = [1,1], scope='g_dconv5', bn = True, bn_decay=0.9, is_training = is_training)
            net = ops.conv2d_transpose(net, 256, kernel_size = [ 3, 3], stride = [2,2], scope='g_dconv6', bn = True, bn_decay=0.9, is_training = is_training)
            net = ops.conv2d_transpose(net,  96, kernel_size = [ 5, 5], stride = [2,2], scope='g_dconv7', bn = True, bn_decay=0.9, is_training = is_training)
            net = ops.conv2d_transpose(net,   3, kernel_size = [11,11], stride = [4,4], scope='g_dconv8', activation_fn = None)

            # net = ops.conv2d_transpose(net, 384, kernel_size = [4,4], stride = [2,2], scope='g_dconv4', bn = True, bn_decay=0.9, is_training = is_training)
            # net = ops.conv2d_transpose(net, 384, kernel_size = [4,4], stride = [2,2], scope='g_dconv5', bn = True, bn_decay=0.9, is_training = is_training)
            # net = ops.conv2d_transpose(net, 256, kernel_size = [4,4], stride = [2,2], scope='g_dconv6', bn = True, bn_decay=0.9, is_training = is_training)
            # net = ops.conv2d_transpose(net,  96, kernel_size = [4,4], stride = [2,2], scope='g_dconv7', bn = True, bn_decay=0.9, is_training = is_training)
            # net = ops.conv2d_transpose(net,   3, kernel_size = [4,4], stride = [2,2], scope='g_dconv8', activation_fn = None)

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
        with tf.name_scope('summaryHelpers'):
            sum_ImgTest = tf.unstack(self.testImgs) # split last axis (num_images) into list of (h, w)
            sum_ImgTest = tf.concat(sum_ImgTest, axis=1) # tile all images horizontally into single row
            sum_ImgTest = tf.split(sum_ImgTest, 8, axis=1) # split into desired number of rows
            self.imageMosaic = tf.concat(sum_ImgTest, axis=0) # tile rows vertically
            
            
        
        with tf.name_scope("SummaryImages"):
            self.summary_img_op = tf.summary.image('testImg', self.testImgMosaics, max_outputs = 20)
            
        
        ### Add summaries
        with tf.name_scope("summaryLosses"):
            sum_Gloss = tf.summary.scalar('lossGenerator', self.g_loss)
            sum_Dloss = tf.summary.scalar('lossDiscriminator', self.d_loss)
            sum_Qloss = tf.summary.scalar('lossClassifier',self.q_loss)
            
            self.summary_loss_op = tf.summary.merge([sum_Gloss, 
                                                     sum_Dloss, 
                                                     sum_Qloss])
            
#            tf.summary.scalar('placeholderScalar', 1) # placeholder summary
#            self.summary_op = tf.summary.merge_all()
        
        
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
        input_shape = [None, 224, 224, 3] # input image shape [batch_size, image_height, image_width, image_channels]
        self.inputImage = tf.placeholder(dtype = tf.float32, shape = input_shape, name = 'real_images')
        self.inputCode = tf.placeholder(dtype = tf.float32, shape = [None, 12], name = 'code_vector') # input code shape [batch_size, code_dim]
        self.inputNoise = tf.placeholder(dtype = tf.float32, shape = [None, 62], name = 'noise_vector') # input noise shape [batch_size, noise_dim]
        self.isTraining = tf.placeholder(dtype = tf.bool, name = 'training_flag')
        
        
        # Create test placeholders
        self.testCategory = tf.placeholder(dtype = tf.int32, shape = [], name = 'testCategory')
        self.testImgs = tf.placeholder(dtype = tf.float32, shape = [64, 224, 224, 3], name = 'testImages')
        self.testImgMosaics = tf.placeholder(dtype = tf.float32, shape = [10, 8*224, 8*224, 3], name = 'testImageMosaics')
        
        
        # Define generator for test variables
        testCodes_generator, test_noise_generator = self._genTestCodes()
        
        
        # Define model, loss, optimizer and summaries.
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
            
            ### --------------------------------------------------------------
            ### Do training loops
            for epoch_n in range(epoch_start, epoch_N):
                
                training_filenames = ['data/processed/' + dataset_str + '/data.tfrecords'] # EXAMPLE
                sess.run(iterator.initializer, feed_dict={filenames: training_filenames})
                
                ### ----------------------------------------------------------
                ### Test the current model
                imageMosaics = np.empty([10, 1792, 1792, 3], dtype=np.float32)
                for n_category in range(10):
                    # Generate a batch of test codes and noise with category n_category
                    codes_test, noise_test = sess.run([testCodes_generator, test_noise_generator], 
                                                      feed_dict = {self.testCategory: n_category}
                                                      )
                    
                    # Make foward pass through generator using the test batch
                    test_img = sess.run(self.img_fake, 
                                        feed_dict={self.inputCode:  codes_test, 
                                                   self.inputNoise: noise_test, 
                                                   self.isTraining: False}
                                        )
                    
                    # Create mosaic from the generated images and store it
                    test_imgMosaic = sess.run(self.imageMosaic,
                                              feed_dict={self.testImgs:  test_img}
                                              )
                    imageMosaics[n_category,:,:,:] = test_imgMosaic

                # Generate summary for TensorBoard
                summaryImg = sess.run(self.summary_img_op,
                                      feed_dict={self.testImgMosaics: imageMosaics})
                writer.add_summary(summaryImg, global_step=epoch_n)
                
                
                ### ----------------------------------------------------------
                ### Update model
                print('Running training epoch no: ', epoch_n)
                while True:
                    try:
                        # Get training bathc from the dataset
                        imgs_batch, codes_batch, noise_batch = sess.run(self.input_getBatch)
                        
                        # Update Discriminator network
                        _ = sess.run([self.d_optimizer_op], 
                                     feed_dict={self.inputImage:    imgs_batch, 
                                                self.inputCode:     codes_batch, 
                                                self.inputNoise:    noise_batch,
                                                self.isTraining:    True}
                                     )
                        
                        # Update Generator and Classifier network
                        _, _, summaryLoss = sess.run([self.g_optimizer_op, self.q_optimizer_op, self.summary_loss_op],
                                                 feed_dict={self.inputImage:    imgs_batch, 
                                                            self.inputCode:     codes_batch, 
                                                            self.inputNoise:    noise_batch,
                                                            self.isTraining:    True}
                                                 )
                        
                        # Write model losses to TensorBoard
                        writer.add_summary(summaryLoss, global_step=counter)
                        counter += 1
                        
                    except tf.errors.OutOfRangeError:
                        break
                
                # Save model variables to checkpoint
                if epoch_n % 1 == 0:
                    saver.save(sess,os.path.join(self.dir_checkpoints, self.model + '.model'), global_step=epoch_n)
                

            
    
    def predict(self):
        """ Run prediction of the network
        Args:
    
        Returns:
        """
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
    


    def _genLatentCodes(self, image_proto, lbl_proto):
        """ Augment dataset entries. Generates latent codes based on class 
            labels, usign one-hot encoding, and adds two continuous latent 
            codes for the network to estimate. Also generates a GAN noise
            vector per data sample.
        Args:
    
        Returns:
        """
        
        image = image_proto
        
        code = tf.one_hot(lbl_proto,10)
        code = tf.concat([code, tf.random_uniform([2], minval = -1, maxval = 1)],0)
        
        noise = tf.random_uniform([62], minval = -1, maxval = 1)
        
        return image, code, noise
    
    def _genTestCodes(self):
        """ Defines test code and noise generator. Generates laten codes based
            on a testCategory input.
        Args:
    
        Returns:
        """
    
        n_rowImage = 8
        n_totImage = n_rowImage * n_rowImage
        
        cat_code = tf.fill([n_totImage],self.testCategory)
        cat_code = tf.one_hot(cat_code,10)
        
        cont_code = tf.lin_space(-1.,1.,n_rowImage)
        cont_code1, cont_code2 = tf.meshgrid(cont_code,cont_code)
        cont_code1 = tf.reshape(cont_code1,[-1])
        cont_code2 = tf.reshape(cont_code2,[-1])
        
        cont_code = tf.stack([cont_code1,cont_code2], axis = 1)
        
        code = tf.concat([cat_code, cont_code], axis = 1)
        noise = tf.zeros([n_totImage,62])
        
        return code, noise