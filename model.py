import tensorflow as tf
import numpy as np
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Conv2D, Input, Conv2DTranspose, Concatenate, Lambda, Add, Multiply

def Coarse_flow(input_list, upscale_factor):
    input_shape = Concatenate()(input_list)

    conv2d_0 = Conv2D(filters = 24,
                    kernel_size = (5, 5),
                    strides = (2, 2),
                    padding = "same",
                    activation = "relu"
                    )(input_shape)
    conv2d_1 = Conv2D(filters = 24,
                    kernel_size = (3, 3),
                    strides = (1, 1),
                    padding = "same",
                    activation = "relu"
                    )(conv2d_0)
    conv2d_2 = Conv2D(filters = 24,
                    kernel_size = (5, 5),
                    strides = (2, 2),
                    padding = "same",
                    activation = "relu"
                    )(conv2d_1)
    conv2d_3 = Conv2D(filters = 24,
                    kernel_size = (3, 3),
                    strides = (1, 1),
                    padding = "same",
                    activation = "relu"
                    )(conv2d_2)
    conv2d_4 = Conv2D(filters = 32,
                    kernel_size = (3, 3),
                    strides = (1, 1),
                    padding = "same",
                    activation = "tanh"
                    )(conv2d_3)

    pixel_shuffle = Lambda(lambda z: tf.nn.depth_to_space(z, upscale_factor))(conv2d_4)

    delta_x = Multiply()([input_list[1], pixel_shuffle])
    delta_x = Add()([tf.expand_dims(delta_x[:,:,:,0], -1), tf.expand_dims(delta_x[:,:,:,1], -1)])

    I_coarse = Add()([input_list[1], delta_x])

    return pixel_shuffle, I_coarse

def Fine_flow(input_list, upscale_factor):
    input_shape = Concatenate()(input_list)

    conv2d_0 = Conv2D(filters = 24,
                    kernel_size = (5, 5),
                    strides = (2, 2),
                    padding = "same",
                    activation = "relu"
                    )(input_shape)
    conv2d_1 = Conv2D(filters = 24,
                    kernel_size = (3, 3),
                    strides = (1, 1),
                    padding = "same",
                    activation = "relu"
                    )(conv2d_0)
    conv2d_2 = Conv2D(filters = 24,
                    kernel_size = (3, 3),
                    strides = (1, 1),
                    padding = "same",
                    activation = "relu"
                    )(conv2d_1)
    conv2d_3 = Conv2D(filters = 24,
                    kernel_size = (3, 3),
                    strides = (1, 1),
                    padding = "same",
                    activation = "relu"
                    )(conv2d_2)
    conv2d_4 = Conv2D(filters = 8,
                    kernel_size = (3, 3),
                    strides = (1, 1),
                    padding = "same",
                    activation = "tanh"
                    )(conv2d_3)

    pixel_shuffle = Lambda(lambda z: tf.nn.depth_to_space(z, upscale_factor))(conv2d_4)

    return pixel_shuffle   

def MES(input_list): #Motion estimation
    delta_c, I_c = Coarse_flow(input_list, upscale_factor = 4)
    
    input_list.append(delta_c)
    input_list.append(I_c)

    delta_f = Fine_flow(input_list, upscale_factor = 2)

    delta = Add()([delta_c, delta_f])
    delta_x = Multiply()([input_list[1], delta])
    delta_x = Add()([tf.expand_dims(delta_x[:,:,:,0], -1), tf.expand_dims(delta_x[:,:,:,1], -1)])

    I_MES = Add()([input_list[1], delta_x])

    return I_MES

def ESPCN(input_list, input_channels, mag):
    input_shape = Concatenate()(input_list)

    conv2d_0 = Conv2D(filters = len(input_list) * input_channels,
                        kernel_size = (5, 5),
                        padding = "same",
                        activation = "relu",
                        )(input_shape)
    conv2d_1 = Conv2D(filters = 32,
                        kernel_size = (3, 3),
                        padding = "same",
                        activation = "relu",
                        )(conv2d_0)
    conv2d_2 = Conv2D(filters = mag ** 2,
                        kernel_size = (3, 3),
                        padding = "same",
                        )(conv2d_1)

    pixel_shuffle = Lambda(lambda z: tf.nn.depth_to_space(z, mag))(conv2d_2)
        
    return pixel_shuffle

def VESPCN(): #main
    input_t_minus_1 = Input(shape = (None, None, 1), name = "input_t_minus_1")
    input_t = Input(shape = (None, None, 1), name = "input_t")
    input_t_plus_1 = Input(shape = (None, None, 1), name = "input_t_plus_1")

    I_t_minus_1 = MES([input_t, input_t_minus_1])
    I_t_plus_1 = MES([input_t, input_t_plus_1])

    ESPCN_input = [I_t_minus_1, input_t, I_t_plus_1]
    result_t = ESPCN(ESPCN_input, len(ESPCN_input), mag = 4)

    model = Model(inputs = [input_t_minus_1, input_t, input_t_plus_1], outputs = [result_t])

    model.summary()
    return model




    