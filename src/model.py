import tensorflow as tf 
import tensorflow.keras as keras 
import numpy as np 


class GatedActivationUnit(keras.layers.Layer):

    def __init__(self, activation="tanh", **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activation.get(activation)
    
    def call(self, inputs):
        n_filters = inputs.shape[-1] //2
        linear_output = self.activation(inputs[...,:n_filters])
        gate = keras.activation.sigmoid(inputs[...,n_filters:])
        return self.activation(linear_output)*gate 

def wavenet_residual_block(inputs, n_filters, dilation_rate, padding):
    z = keras.layers.Conv1D(2*n_filters, kernel_size =2, padding = padding,
                            dilation_rate = dilation_rate)(inputs)
    z = GatedActivationUnit()(z)
    z = keras.layers.Conv1D(n_filters, kernel_size=1)(z)
    return keras.layers.Add()([z, inputs]), z 