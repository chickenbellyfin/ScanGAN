import os

import numpy as np

from keras import backend as K
from keras.layers import Activation, Dense, Flatten, Input, Lambda, merge, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import AveragePooling2D, Convolution2D, UpSampling2D
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD
from keras.regularizers import l2

import tensorflow as tf


class GAN(object):

  def __init__(self, shape, input_mask=False):
    self.W = shape[0]
    self.H = shape[1]
    if len(shape) == 3:
      self.D = shape[2]
    else:
      self.D = 1;
    self.input_mask = input_mask

    self._build()

  def train(self, x, y, epochs=1, batches=None, callback=None):

    x = np.array(map(lambda a: a.reshape(self.W, self.H, self.D), x))
    y = np.array(map(lambda a: a.reshape(self.W, self.H, self.D), y))

    count = len(x)
    if not batches:
      batches = count
    batch_size = int(count/batches)

    for epoch in range(epochs):

      if epoch == 0 or self.d_loss > .3:
        for batch in range(batches):
          batch_x = x[batch * batch_size : (batch+1) * batch_size]
          batch_y = y[batch * batch_size : (batch+1) * batch_size]
          
          # train the discriminator
          batch_gen = self.generate(batch_x)
          
          self.discriminator.train_on_batch(batch_y, np.zeros(batch_size))
          self.discriminator.train_on_batch(batch_gen, np.ones(batch_size))

      else:
        for batch in range(batches):
          batch_x = x[batch * batch_size : (batch+1) * batch_size]
          batch_y = y[batch * batch_size : (batch+1) * batch_size]

          self.model.train_on_batch(batch_x, [batch_y, np.zeros(batch_size)]) 


      eval_x = np.concatenate((self.generate(x), y))
      eval_y = np.concatenate((np.ones(len(x)), np.zeros(len(y))))
      self.d_loss = self.discriminator.evaluate(eval_x, eval_y,verbose=0)
      
      self.g_losses = self.model.evaluate(x, [y, np.ones(len(x))], verbose=0)
      self.g_loss_mse = self.g_losses[1]
      self.g_loss_al = self.g_losses[2]

      if callback:
        callback(epoch, [self.g_loss_mse, self.g_loss_al], self.d_loss)


  def generate(self, x):
    if len(x.shape) == 3:
      x = np.array([x])
    res = self.model.predict(x.reshape(x.shape[0], self.W, self.H, self.D))
    return res[0]


  def discriminate(self, x):
    return self.discriminator.predict(x)      


  def _build(self, gdepth=48, gactivation=lambda: LeakyReLU(), ddepth=32, dactivation=lambda: LeakyReLU()):
    reg = lambda: None #l2()

    k = 5 # kernel size
    # GENERATOR
    # input
    g_in = Input(shape=(self.W, self.H, self.D), name='g_in')    
    # encode
    g = Convolution2D(gdepth, k, k, border_mode='same', activation=gactivation(), W_regularizer=reg())(g_in)
    g = AveragePooling2D(pool_size=(2,2))(g)
    g = Convolution2D(gdepth, k, k, border_mode='same', activation=gactivation(), W_regularizer=reg())(g)
    g = AveragePooling2D(pool_size=(2,2))(g)
    g = Convolution2D(gdepth, k, k, border_mode='same', activation=gactivation(), W_regularizer=reg())(g)
    g = AveragePooling2D(pool_size=(2,2))(g)

    # g = Flatten()(g)
    # # g = Dense(output_dim=1024, activation=gactivation(), W_regularizer=reg())(g)
    # g = Dense(output_dim=gdepth*(self.W/8)*(self.H/8), activation=gactivation(), W_regularizer=reg())(g)
    # g = Reshape((self.W/8, self.H/8, gdepth))(g)

    #decode
    g = Convolution2D(gdepth, k, k, border_mode='same', activation=gactivation(), W_regularizer=reg())(g)
    g = UpSampling2D(size=(2,2))(g)
    g = Convolution2D(gdepth, k, k, border_mode='same', activation=gactivation(), W_regularizer=reg())(g)
    g = UpSampling2D(size=(2,2))(g)
    g = Convolution2D(gdepth, k, k, border_mode='same', activation=gactivation(), W_regularizer=reg())(g)
    g = UpSampling2D(size=(2,2))(g)    
    g = Convolution2D(gdepth, k, k, border_mode='same', activation=gactivation(), W_regularizer=reg())(g)
    # output
    g_out = Convolution2D(self.D, k, k, border_mode='same', activation=gactivation(), W_regularizer=reg(), name='g_out')(g)


    if self.input_mask:
      contour_combine = lambda x: x[0] + tf.cast(tf.equal(x[0], tf.zeros_like(x[0])), tf.float32) * x[1]
      g_out = merge([g_in, g_out], mode=contour_combine, output_shape=(self.W, self.H, self.D))
    
    print g_out.get_shape()
    # DESCRIMINATOR
    # conv
    d_in = Input(shape=(self.W, self.H, self.D), name='d_in')
    d = Convolution2D(ddepth, k, k, border_mode='same', activation=dactivation())(d_in)
    d = AveragePooling2D(pool_size=(2,2))(d)
    d = Convolution2D(ddepth, k, k, border_mode='same', activation=dactivation())(d)
    d = AveragePooling2D(pool_size=(2,2))(d)
    # fully connected
    d = Flatten()(d)
    d = Dense(output_dim=256, activation=dactivation())(d)
    d_out = Dense(output_dim=1, activation='sigmoid', name='d_out')(d)

    g_optimizer = Adam(1e-2)
    d_optimizer = SGD()    

    self.discriminator = Model(d_in, d_out);    
    self.generator = Model(g_in, g_out);

    gan_input = Input(shape=(self.W, self.H, self.D))
    gan_h = self.generator(gan_input)
    gan_out = self.discriminator(gan_h)

    self.model = Model(
      input=gan_input, 
      output=[gan_h, gan_out])

    self._enable_training(self.discriminator, True)
    self.discriminator.compile(optimizer=d_optimizer, loss='binary_crossentropy')
    self.generator.compile(optimizer='sgd', loss='mse')
    
    self._enable_training(self.discriminator, False)

    self.model.compile(optimizer=g_optimizer,
      loss=['mse', 'binary_crossentropy'],
      loss_weights=[1e4, 1])

    # self.discriminator.summary()
    # self.generator.summary()
    # self.model.summary()


  def _enable_training(self, model, enable):
    model.trainable = enable
    for l in model.layers:
      l.trainable = enable
