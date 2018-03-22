#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 14:01:52 2017

@author: leminen
"""
import sys
import os
import tensorflow as tf
import numpy as np
import itertools
import functools
import matplotlib.pyplot as plt
import datetime

sys.path.append('/home/leminen/Documents/RoboWeedMaps/GAN/weed-gan-v1')
import src.data.process_dataset as process_dataset
import src.utils as utils

tfgan = tf.contrib.gan
layers = tf.contrib.layers
framework = tf.contrib.framework
ds = tf.contrib.distributions

leaky_relu = lambda net: tf.nn.leaky_relu(net, alpha=0.01)

class infoGAN_rgb(object):
    def __init__(self):
        # Setup model folders
        self.model = 'infoGAN_rgb'
        self.dir_logs        = 'models/' + self.model + '/logs'
        self.dir_checkpoints = 'models/' + self.model + '/checkpoints'
        self.dir_results     = 'models/' + self.model + '/results'
        
        utils.checkfolder(self.dir_checkpoints)
        utils.checkfolder(self.dir_logs)
        utils.checkfolder(self.dir_results)

        #
        self.unstructured_noise_dim = 62
        self.categorical_noise_dim = 10
        self.continuous_noise_dim = 3
 
    def __generator(self, inputs, categorical_dim, weight_decay = 2.5e-5, is_training = True):
        """InfoGAN discriminator network on MNIST digits.

        Based on a paper https://arxiv.org/abs/1606.03657 and their code
        https://github.com/openai/InfoGAN.
        
        Args:
            inputs: A 3-tuple of Tensors (unstructured_noise, categorical structured
                noise, continuous structured noise). `inputs[0]` and `inputs[2]` must be
                2D, and `inputs[1]` must be 1D. All must have the same first dimension.
            categorical_dim: Dimensions of the incompressible categorical noise.
            weight_decay: The value of the l2 weight decay.
            is_training: If `True`, batch norm uses batch statistics. If `False`, batch
                norm uses the exponential moving average collected from population 
                statistics.
        
        Returns:
            A generated image in the range [-1, 1].
        """

        unstructured_noise, cat_noise, cont_noise = inputs
        cat_noise_onehot = tf.one_hot(cat_noise, categorical_dim)
        all_noise = tf.concat([unstructured_noise, cat_noise_onehot, cont_noise], axis=1)
    
        with framework.arg_scope(
            [layers.fully_connected, layers.conv2d_transpose],
            activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,
            weights_regularizer=layers.l2_regularizer(weight_decay)),\
        framework.arg_scope([layers.batch_norm], is_training=is_training):
            net = layers.fully_connected(all_noise, 1024)
            net = layers.fully_connected(net, 7 * 7 * 128)
            net = tf.reshape(net, [-1, 7, 7, 128])
            net = layers.conv2d_transpose(net, 64, [4, 4], stride = 2)
            net = layers.conv2d_transpose(net, 32, [4, 4], stride = 2)
            # Make sure that generator output is in the same range as `inputs`
            # ie [-1, 1].
            net = layers.conv2d(net, 3, 4, normalizer_fn=None, activation_fn=tf.tanh)
    
            return net
    
    def __discriminator(self, img, unused_conditioning, weight_decay=2.5e-5, categorical_dim=10, continuous_dim=2, is_training=True):
        """InfoGAN discriminator network on MNIST digits.
    
        Based on a paper https://arxiv.org/abs/1606.03657 and their code
        https://github.com/openai/InfoGAN.
    
        Args:
            img: Real or generated MNIST digits. Should be in the range [-1, 1].
                unused_conditioning: The TFGAN API can help with conditional GANs, which
                would require extra `condition` information to both the generator and the
                discriminator. Since this example is not conditional, we do not use this
                argument.
            weight_decay: The L2 weight decay.
            categorical_dim: Dimensions of the incompressible categorical noise.
            continuous_dim: Dimensions of the incompressible continuous noise.
            is_training: If `True`, batch norm uses batch statistics. If `False`, batch
                norm uses the exponential moving average collected from population statistics.
    
        Returns:
            Logits for the probability that the image is real, and a list of posterior 
                distributions for each of the noise vectors.
        """
        with framework.arg_scope(
            [layers.conv2d, layers.fully_connected],
            activation_fn=leaky_relu, normalizer_fn=None,
            weights_regularizer=layers.l2_regularizer(weight_decay),
            biases_regularizer=layers.l2_regularizer(weight_decay)):
            net = layers.conv2d(img,  64, [4, 4], stride = 2)
            net = layers.conv2d(net, 128, [4, 4], stride = 2)
            net = layers.flatten(net)
            net = layers.fully_connected(net, 1024, normalizer_fn=layers.layer_norm)
    
            logits_real = layers.fully_connected(net, 1, activation_fn=None)

            # Recognition network for latent variables has an additional layer
            with framework.arg_scope([layers.batch_norm], is_training=is_training):
                encoder = layers.fully_connected(
                    net, 128, normalizer_fn=layers.batch_norm)

            # Compute logits for each category of categorical latent.
            logits_cat = layers.fully_connected(
                encoder, categorical_dim, activation_fn=None)
            q_cat = ds.Categorical(logits_cat)

            # Compute mean for Gaussian posterior of continuous latents.
            mu_cont = layers.fully_connected(
                encoder, continuous_dim, activation_fn=None)
            sigma_cont = tf.ones_like(mu_cont)
            q_cont = ds.Normal(loc=mu_cont, scale=sigma_cont)

            return logits_real, [q_cat, q_cont]
    

    def _create_inference(self):
        """ Define the inference model for the network
        Args:
    
        Returns:
        """
        # Create input placeholders
        self.real_images = tf.placeholder(
            dtype = tf.float32, 
            shape = [None,28,28,3], 
            name = 'input_images')
        self.unstructured_noise = tf.placeholder(
            dtype = tf.float32, 
            shape = [None, self.unstructured_noise_dim], 
            name = 'input_unstructured_noise')
        self.categorical_noise = tf.placeholder(
            dtype = tf.int32,   
            shape = [None], 
            name = 'input_categorial_noise')
        self.continuous_noise = tf.placeholder(
            dtype = tf.float32, 
            shape = [None,self.continuous_noise_dim], 
            name = 'input_continuous_noise')

        # Combine noise inputs
        unstructured_inputs = [self.unstructured_noise]
        structured_inputs = [self.categorical_noise, self.continuous_noise]

        # Create generator and discriminator functions and setup infoGAN model
        generator_fn = functools.partial(self.__generator, 
                                         categorical_dim = self.categorical_noise_dim)
        discriminator_fn = functools.partial(self.__discriminator, 
                                             categorical_dim = self.categorical_noise_dim, 
                                             continuous_dim = self.continuous_noise_dim)

        self.infogan_model = tfgan.infogan_model(
            generator_fn = generator_fn,
            discriminator_fn = discriminator_fn,
            real_data = self.real_images,
            unstructured_generator_inputs = unstructured_inputs,
            structured_generator_inputs = structured_inputs)
    
    def _create_losses(self):
        """ Define loss function[s] for the network
        Args:
    
        Returns:
        """
        
        # Create infoGAN losses
        self.infogan_loss = tfgan.gan_loss(
            self.infogan_model,
            gradient_penalty_weight = 1.0,
            mutual_information_penalty_weight = 1.0)
        
    def _create_optimizer(self):
        """ Create optimizer for the network
        Args:
    
        Returns:
        """

        # Create optimizers and Create update operations
        generator_optimizer = tf.train.AdamOptimizer(learning_rate = 0.001, beta1 = 0.5)
        discriminator_optimizer = tf.train.AdamOptimizer(learning_rate = 0.00009, beta1 = 0.5)

        self.gan_train_ops = tfgan.gan_train_ops(
            self.infogan_model,
            self.infogan_loss,
            generator_optimizer = generator_optimizer,
            discriminator_optimizer = discriminator_optimizer)
        
    def _create_summaries(self):
        """ Create summaries for the network
        Args:
    
        Returns:
        """

        # Create image summaries to inspect the variation due to categorical latent codes
        with tf.name_scope("SummaryImages_CategoricalVariation"):
            grid_size = 10
            continuous_vars = [0,1]
            images_test = []
            for categorical_var in range(0,10):
                with tf.variable_scope('Generator', reuse=True):
                    noise = self._genTestCodes(categorical_var, continuous_vars, grid_size)
                    images_cat = self.infogan_model.generator_fn(noise, is_training=False)
                    images_cat = tfgan.eval.image_reshaper(tf.concat(images_cat, 0), num_cols=grid_size)
                    images_test.append(images_cat[0,:,:,:])

            self.summary_imgCat_op = tf.summary.image('test_images', images_test, max_outputs = 20)

        # Create image summaries to inspect the variation due to continuous latent codes 
        with tf.name_scope("SummaryImages_ContinuousVariation"):
            grid_size = 10
            categorical_var = 0
            continuous_variables = list(itertools.combinations(range(0,self.continuous_noise_dim),2))
            images_test = []
            for continuous_vars in continuous_variables:
                with tf.variable_scope('Generator', reuse=True):
                    noise = self._genTestCodes(categorical_var, continuous_vars, grid_size)
                    images_cat = self.infogan_model.generator_fn(noise, is_training=False)
                    images_cat = tfgan.eval.image_reshaper(tf.concat(images_cat, 0), num_cols=grid_size)
                    images_test.append(images_cat[0,:,:,:])

            self.summary_imgCont_op = tf.summary.image('test_images', images_test, max_outputs = 20)

        ### Add loss summaries
        with tf.name_scope("SummaryLosses"):
            summary_gloss = tf.summary.scalar('loss_generator', self.infogan_loss.generator_loss)
            summary_dloss = tf.summary.scalar('loss_discriminator', self.infogan_loss.discriminator_loss)
            
            self.summary_loss_op = tf.summary.merge([summary_gloss, 
                                                     summary_dloss])
                                                                 
        
    def train(self, dataset_str, epoch_N, batch_size):
        """ Run training of the network
        Args:
    
        Returns:
        """
        
        # Use dataset for loading in datasamples from .tfrecord (https://www.tensorflow.org/programmers_guide/datasets#consuming_tfrecord_data)
        # The iterator will get a new batch from the dataset each time a sess.run() is executed on the graph.
        filenames = ['data/processed/' + dataset_str + '/train.tfrecords']
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(process_dataset._decodeData)      # decoding the tfrecord
        dataset = dataset.map(self._genLatentCodes)
        dataset = dataset.shuffle(buffer_size = 10000, seed = None)
        dataset = dataset.batch(batch_size = batch_size)
        iterator = dataset.make_initializable_iterator()
        input_getBatch = iterator.get_next()        
        
        # Define model, loss, optimizer and summaries.
        self._create_inference()
        self._create_losses()
        self._create_optimizer()
        self._create_summaries()

        ### From first TFGAN implementation
        # global_step = tf.train.get_or_create_global_step()
        # train_step_fn = tfgan.get_sequential_train_steps()

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
                epoch_start = int(ckpt_name.split('-')[-1]) + 1
            
            interationCnt = 0
            for epoch_n in range(epoch_start, epoch_N):

                # Test model output before any training
                if epoch_n == 0:
                    summaryImg = sess.run(self.summary_imgCat_op)
                    writer.add_summary(summaryImg, global_step=-1)

                    summaryImg = sess.run(self.summary_imgCont_op)
                    writer.add_summary(summaryImg, global_step=-1)

                # Initiate or Re-initiate iterator
                sess.run(iterator.initializer)
                
                ### ----------------------------------------------------------
                ### Update model
                print(datetime.datetime.now(),'- Running training epoch no:', epoch_n)
                while True:
                    try:
                        image_batch, unst_noise_batch, cat_noise_batch, cont_noise_batch = sess.run(input_getBatch)

                        _ = sess.run(
                            [self.gan_train_ops.discriminator_train_op],
                             feed_dict={self.real_images:        image_batch, 
                                        self.unstructured_noise: unst_noise_batch, 
                                        self.categorical_noise:  cat_noise_batch,
                                        self.continuous_noise:   cont_noise_batch})

                        _, summaryLoss = sess.run(
                            [self.gan_train_ops.generator_train_op, self.summary_loss_op],
                             feed_dict={self.real_images:        image_batch, 
                                        self.unstructured_noise: unst_noise_batch, 
                                        self.categorical_noise:  cat_noise_batch,
                                        self.continuous_noise:   cont_noise_batch})

                        writer.add_summary(summaryLoss, global_step=interationCnt)
                        interationCnt += 1                        
                        
                    except tf.errors.OutOfRangeError:
                        # Test current model
                        summaryImg = sess.run(self.summary_imgCat_op)
                        writer.add_summary(summaryImg, global_step=epoch_n)

                        summaryImg = sess.run(self.summary_imgCont_op)
                        writer.add_summary(summaryImg, global_step=epoch_n)
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
        """ Augment dataset entries. Adds two continuous latent 
            codes for the network to estimate. Also generates a GAN noise
            vector per data sample.
        Args:

        Returns:
        """
    
        # scalar = tf.random_uniform([1], minval = 0, maxval = 1)
        # scalar = tf.reshape(scalar, [])
        # mask = tf.convert_to_tensor([scalar, 1-scalar, scalar], dtype=tf.float32)
        # scalar = tf.concat(([1],tf.random_uniform([2], minval = 0, maxval = 1)),0)
        scalar = tf.random_uniform([3],minval = 0, maxval = 1)
        mask = tf.tile(scalar,[28*28])
        mask = tf.reshape(mask,[28,28,3])

        image = tf.image.grayscale_to_rgb(image_proto)
        image = image + 1
        image = tf.multiply(image, mask)
        image = image - 1
        # image = tf.div((image - 0.5),0.5) 

        unstructured_noise = tf.random_normal([self.unstructured_noise_dim])

        categorical_noise = lbl_proto
        continuous_noise = (scalar * 2) - 1 #tf.random_uniform([self.continuous_noise_dim], minval = -1, maxval = 1)
    
        return image, unstructured_noise, categorical_noise, continuous_noise
    
    def _genTestCodes(self, cat_code, var_cont_dim, grid_dim):
        """ Defines test code and noise generator. Generates laten codes based
            on a testCategory input.
        Args:
    
        Returns:
        """

        n_images = grid_dim ** 2

        unstructured_noise = np.random.normal(size=[n_images, self.unstructured_noise_dim])
        categorical_noise = np.tile(cat_code, n_images)
        continuous_noise_vals = np.linspace(-1,1, grid_dim)
        continuous_noise_temp = []

        for continuous_noise_1 in continuous_noise_vals:
            for continuous_noise_2 in continuous_noise_vals:
                continuous_noise_temp.append([continuous_noise_1, continuous_noise_2])
        
        continuous_noise_temp = np.array(continuous_noise_temp)
        continuous_noise = np.zeros(shape = [n_images,self.continuous_noise_dim])
        continuous_noise[:, var_cont_dim] = continuous_noise_temp

        return unstructured_noise, categorical_noise, continuous_noise

# model = infoGAN_rgb()
# model.train('MNIST', 40, 32)
