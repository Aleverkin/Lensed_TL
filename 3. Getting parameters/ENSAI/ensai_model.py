from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import slim as contrib_slim

slim = contrib_slim


trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)
def model_transformer(x_image , scope="transformer", reuse=None):
    
        with tf.variable_scope(scope):
                with slim.arg_scope([slim.conv2d, slim.fully_connected], activation_fn=tf.nn.relu):
                        net = slim.conv2d(x_image, 64, [11, 11], 4, padding='VALID', scope='conv1')
                        net = slim.max_pool2d(net, [2, 2], scope='pool1')
                        net = slim.conv2d(net, 256, [5, 5], padding='VALID', scope='conv2')
                        net = slim.max_pool2d(net, [2, 2], scope='pool2')
                        net = slim.conv2d(net, 512, [3, 3], scope='conv3')
                        net = slim.conv2d(net, 1024, [3, 3], scope='conv4')
                        net = slim.conv2d(net, 1024, [3, 3], scope='conv5')
                        net = slim.max_pool2d(net, [2, 2], scope='pool5')
                        with slim.arg_scope([slim.conv2d], weights_initializer=trunc_normal(0.005), biases_initializer=tf.constant_initializer(0.1)):
                                net = slim.conv2d(net, 3072, [2, 2], padding='VALID', scope='fc6')
                                net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
                                net = slim.conv2d(net, 5, [1, 1], activation_fn=None, normalizer_fn=None,  biases_initializer=tf.zeros_initializer(), scope='fc8')
                                net = slim.flatten(net, scope='Flatten')
                                net = slim.fully_connected(net,  5  , activation_fn = None ,  scope='FC1')
                net_x = slim.fully_connected(net, 1 , activation_fn=None , scope='TF_predict_x')
                net_y = slim.fully_connected(net, 1 , activation_fn=None , scope='TF_predict_y')
                zero_col = net_x * 0
                one_col = net_x * 0 + 1
                transformation_tensor = tf.concat( axis = 1 , values = [one_col, zero_col , 1.0 * net_x / ((192*0.04)/2), zero_col , one_col , 1.0 * net_y / ((192*0.04)/2) ] )
                x_image = transformer(x_image, transformation_tensor , (numpix_side, numpix_side) )
                x_image = tf.reshape(x_image , [-1,numpix_side,numpix_side,1] )
        return x_image, net_x, net_y



batch_norm_params = {
      # Decay for the moving averages.
      'decay': 0.9997,
      # epsilon to prevent 0s in variance.
      'epsilon': 0.001,
      # collection containing update_ops.
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
}


def block_a(inputs, scope=None, reuse=None):
  # By default use stride=1 and SAME padding
  with slim.arg_scope([slim.conv2d], stride=1, padding='SAME'):
    with tf.variable_scope(scope, 'BlockA', [inputs], reuse=reuse):
      with tf.variable_scope('branch_0'):
        branch_0 = slim.conv2d(inputs, 32, [1, 1], scope='Conv2d_0a_1x1')
      with tf.variable_scope('branch_1'):
        branch_1 = slim.conv2d(inputs, 16, [1, 1], scope='Conv2d_0a_1x1')
        branch_1 = slim.conv2d(branch_1, 16, [3, 3], scope='Conv2d_0b_3x3')
      with tf.variable_scope('branch_2'):
        branch_2 = slim.conv2d(inputs, 32, [1, 1], scope='Conv2d_0a_1x1')
        branch_2 = slim.conv2d(branch_2, 32, [3, 3], scope='Conv2d_0b_3x3')
        branch_2 = slim.conv2d(branch_2, 32, [5, 5], scope='Conv2d_0c_5x5')
        branch_2 = slim.conv2d(branch_2, 32, [10, 10], scope='Conv2d_0c_10x10')
      with tf.variable_scope('branch_3'):
        branch_3 = slim.conv2d(inputs, 32, [1, 1], scope='Conv2d_0b_1x1')
      return tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
      mixed = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
      up = slim.conv2d(mixed, inputs.get_shape()[3], 1, normalizer_fn = slim.batch_norm ,  normalizer_params = batch_norm_params , activation_fn=tf.nn.relu, scope='RES_1')
      scale = 0.7
      inputs += scale * up


def model_1(net, scope="EN_Model1", reuse=None):
    with tf.variable_scope(scope):
        with tf.variable_scope('BlockA_1',reuse=reuse):
            net = block_a(net)
        with tf.variable_scope('BlockA_2',reuse=reuse):
            net = block_a(net)
        net = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='MaxPool_1')
        with tf.variable_scope('BlockA_3',reuse=reuse):
            net = block_a(net)
        with tf.variable_scope('BlockA_4',reuse=reuse):
            net = block_a(net)
        net = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='MaxPool_2')
        
        net = slim.flatten(net, scope='Flatten')
        net = slim.fully_connected(net, 124, activation_fn=tf.nn.relu, scope='FC1')
        net = slim.fully_connected(net, 5 , activation_fn=None, scope='read_out_layer')
        net = tf.reshape(net,[-1,5])
        return net


def model_2(net, scope="EN_Model2", reuse=None):
    with tf.variable_scope(scope):
        with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu):
            net = slim.conv2d(net, 16, [3, 3], stride=1 , scope='conv_1')
            net = slim.conv2d(net, 16, [3, 3], stride=1 , scope='conv_2')
            net = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='MaxPool_1')
            net = slim.conv2d(net, 32, [3, 3], stride=1 , scope='conv_3')
            net = slim.conv2d(net, 32, [1, 1], stride=1 , scope='conv_4')
            net = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='MaxPool_2')
            net = slim.conv2d(net, 32, [5, 5], stride=2 , scope='conv_5')
            net = slim.conv2d(net, 32, [5, 5], stride=2 , scope='conv_6')
            net = slim.conv2d(net, 32, [1, 1], stride=1 , scope='conv_7')
            net = slim.conv2d(net, 32, [5, 5], stride=2 , scope='conv_8')
            
            net = slim.flatten(net, scope='Flatten')
            net = slim.fully_connected(net, 124, activation_fn=tf.nn.relu , scope='FC1')
            net = slim.fully_connected(net, 5 , activation_fn=None, scope='read_out_layer')
            net = tf.reshape(net,[-1,5])
        return net

# No pooling, conv stride 2 for reduction
def model_3(net, scope="EN_Model3", reuse=None):
    with tf.variable_scope(scope):
        with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu,  stride=1):
            net = slim.conv2d(net, 16, [3, 3], activation_fn=None , scope='conv_10')
            net = slim.conv2d(net, 16, [1, 1], activation_fn=None , scope='conv_11')
            
            net = slim.conv2d(net, 16, [5, 5], activation_fn=None , scope='conv_20')
            net = slim.conv2d(net, 16, [1, 1] , scope='conv_21')
            
            net = slim.conv2d(net, 16, [10, 10],  stride=2 , scope='conv_30')
            net = slim.conv2d(net, 16, [1, 1] , scope='conv_31')
            
            net = slim.conv2d(net, 16, [10, 10], stride=2 , scope='conv_40')
            net = slim.conv2d(net, 16, [1, 1] , scope='conv_41')
            
            net = slim.conv2d(net, 32, [10, 10], stride=2 , scope='conv_50')
            net = slim.conv2d(net, 32, [1, 1] , scope='conv_51')
            
            net = slim.conv2d(net, 64, [3, 3], stride=2 , scope='conv_60')
            net = slim.conv2d(net, 64, [1, 1] , scope='conv_61')
            
            net = slim.flatten(net, scope='Flatten')
            net = slim.fully_connected(net, 124 , activation_fn=tf.nn.relu ,  scope='FC1')
            net = slim.fully_connected(net, 5 , activation_fn=None, scope='read_out_layer')
            net = tf.reshape(net,[-1,5])
        return net



# Few layers, no pooling, only one stride 2 conv.
def model_4(net, scope="EN_Model4", reuse=None):
        with tf.variable_scope(scope):
                with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu,  stride=1):
                        net = slim.conv2d(net, 16, [3, 3], activation_fn=None , scope='conv_10')
                        net = slim.conv2d(net, 16, [1, 1], activation_fn=None , scope='conv_11')

                        net = slim.conv2d(net, 16, [5, 5], activation_fn=None , scope='conv_20')
                        net = slim.conv2d(net, 16, [1, 1] , scope='conv_21')

                        net = slim.conv2d(net, 16, [20, 20],  stride=2 , scope='conv_30')
                        net = slim.conv2d(net, 16, [1, 1] , scope='conv_31')


                        net = slim.flatten(net, scope='Flatten')
                        net = slim.fully_connected(net, 32 , activation_fn=tf.nn.relu ,  scope='FC1')
                        net = slim.fully_connected(net, 16 , activation_fn=tf.nn.relu ,  scope='FC2')
                        net = slim.fully_connected(net, 5 , activation_fn=None, scope='read_out_layer')
                        net = tf.reshape(net,[-1,5])
                return net


# Overfeat model
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)
def model_5(net, scope="EN_Model5", reuse=None):
    with tf.variable_scope(scope):
        with slim.arg_scope([slim.conv2d], padding = 'SAME', activation_fn=tf.nn.relu,  stride=1):
                net = slim.conv2d(net, 64, [11, 11], stride=4, padding='VALID', scope='conv1')
                net = slim.max_pool2d(net, [2, 2], scope='pool1')
                net = slim.conv2d(net, 256, [5, 5], padding='VALID', scope='conv2')
                net = slim.max_pool2d(net, [2, 2], scope='pool2')
                net = slim.conv2d(net, 512, [3, 3], scope='conv3')
                net = slim.conv2d(net, 1024, [3, 3], scope='conv4')
                net = slim.conv2d(net, 1024, [3, 3], scope='conv5')
                net = slim.max_pool2d(net, [2, 2], scope='pool5')
        with slim.arg_scope([slim.conv2d], weights_initializer=trunc_normal(0.005), biases_initializer=tf.constant_initializer(0.1), activation_fn=tf.nn.relu):
                net = slim.conv2d(net, 3072, [2, 2], padding='VALID', scope='fc6')
                net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
                net = slim.conv2d(net, 5, [1, 1], activation_fn=None, normalizer_fn=None,  biases_initializer=tf.zeros_initializer(), scope='fc8')
                net = slim.flatten(net, scope='Flatten')
                net = slim.fully_connected(net,  5  , activation_fn = None ,  scope='FC1')
                net = tf.reshape(net,[-1,5])
        return net


# Simple, minimalistic model with pool at every layer
def model_6(net, scope="EN_Model6", reuse=None):
        with tf.variable_scope(scope):
            with slim.arg_scope([slim.conv2d], padding = 'SAME', activation_fn=tf.nn.relu,  stride=1):

                net = slim.conv2d(net, 8, [3, 3] , scope='conv1')
                net = slim.max_pool2d(net, [3, 3], scope='pool1')

                net = slim.conv2d(net, 8, [5, 5] , scope='conv2')
                net = slim.max_pool2d(net, [3, 3], scope='pool2')

                net = slim.conv2d(net, 16, [5, 5], scope='conv3')
                net = slim.max_pool2d(net, [2, 2], scope='pool3')

                net = slim.conv2d(net, 16, [3, 3], scope='conv4')
                net = slim.max_pool2d(net, [3, 3], scope='pool4')

                net = slim.conv2d(net, 16, [3, 3], scope='conv5')
                net = slim.max_pool2d(net, [2, 2], scope='pool5')

                net = slim.flatten(net, scope='Flatten')
                net = slim.fully_connected(net, 128 , activation_fn = tf.nn.relu ,  scope='FC1')
                net = slim.fully_connected(net,  5  , activation_fn = None ,  scope='FC3')
                net = tf.reshape(net,[-1,5])
                return net






# only 2 fully connected layers
def model_7(net, scope="EN_Model7", reuse=None):
        with tf.variable_scope(scope):
            net = slim.flatten(net, scope='Flatten')
            net = slim.fully_connected(net, 128 , activation_fn = tf.nn.relu ,  scope='FC1')
            net = slim.fully_connected(net,  64  , activation_fn = tf.nn.relu ,  scope='FC2')
            net = slim.fully_connected(net,  5  , activation_fn = None ,  scope='FC3')
            net = tf.reshape(net,[-1,5])
            return net









# No pooling, conv stride 2 for reduction
def model_8(net, scope="EN_Model8", reuse=None):
        with tf.variable_scope(scope):
                with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu,  stride=1):
                        net = slim.conv2d(net, 32, [3, 3], activation_fn=None , scope='conv_10')
                        net = slim.conv2d(net, 32, [1, 1], activation_fn=None , scope='conv_11')

                        net = slim.conv2d(net, 32, [5, 5], activation_fn=None , scope='conv_20')
                        net = slim.conv2d(net, 32, [1, 1] , scope='conv_21')

                        net = slim.conv2d(net, 32, [10, 10],  stride=2 , scope='conv_30')
                        net = slim.conv2d(net, 32, [1, 1] , scope='conv_31')

                        net = slim.conv2d(net, 32, [10, 10],  stride=1 , scope='conv_40')
                        net = slim.conv2d(net, 32, [1, 1] , scope='conv_41')

                        net = slim.conv2d(net, 64, [10, 10], stride=2 , scope='conv_50')
                        net = slim.conv2d(net, 64, [1, 1] , scope='conv_51')

                        net = slim.conv2d(net, 64, [10, 10],  stride=1 , scope='conv_60')
                        net = slim.conv2d(net, 64, [1, 1] , scope='conv_61')

                        net = slim.conv2d(net, 128, [10, 10], stride=2 , scope='conv_70')
                        net = slim.conv2d(net, 128, [1, 1] , scope='conv_71')

                        net = slim.conv2d(net, 256, [3, 3], stride=1 , scope='conv_80')
                        net = slim.conv2d(net, 256, [1, 1] , scope='conv_81')

                        net = slim.flatten(net, scope='Flatten')
                        net = slim.fully_connected(net, 512 , activation_fn=tf.nn.relu ,  scope='FC1')
                        net = slim.fully_connected(net, 5 , activation_fn=None, scope='read_out_layer')
                        net = tf.reshape(net,[-1,5])
                return net




trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)

def alexnet_v2_arg_scope(weight_decay=0.0005):
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      biases_initializer=tf.constant_initializer(0.1),
                      weights_regularizer=slim.l2_regularizer(weight_decay)):
    with slim.arg_scope([slim.conv2d], padding='SAME'):
      with slim.arg_scope([slim.max_pool2d], padding='VALID') as arg_sc:
        return arg_sc


def model_9(net,num_classes=5,is_training=True,scope='alexnet_v2'):

  with tf.variable_scope(scope, 'alexnet_v2', [net]) as sc:
    net = slim.conv2d(net, 64, [11, 11], 4, padding='VALID',scope='conv1')
    net = slim.max_pool2d(net, [3, 3], 2, scope='pool1')
    net = slim.conv2d(net, 192, [5, 5], scope='conv2')
    net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')
    net = slim.conv2d(net, 384, [3, 3], scope='conv3')
    net = slim.conv2d(net, 384, [3, 3], scope='conv4')
    net = slim.conv2d(net, 256, [3, 3], scope='conv5')
    net = slim.max_pool2d(net, [3, 3], 2, scope='pool5')

    # Use conv2d instead of fully_connected layers.
    with slim.arg_scope([slim.conv2d], weights_initializer=trunc_normal(0.005), biases_initializer=tf.constant_initializer(0.1)):
        net = slim.conv2d(net, 4096, [4, 4], padding='VALID', scope='fc6')
        net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
        net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, biases_initializer=tf.zeros_initializer(),scope='fc8')
        net = tf.reshape(net,[-1,5])
    return net
model_9.default_image_size = 192







# Kitt Peak model
def model_10(net, scope="EN_Model10", reuse=None):
        with tf.variable_scope(scope):
                with slim.arg_scope([slim.conv2d], padding = 'SAME', activation_fn=tf.nn.relu,  stride=1):


                        MASK = tf.abs(tf.sign(net))
                        XX =  net +  ( (1-MASK) * 1000.0)
                        bias_measure_filt = tf.constant((1.0/16.0), shape=[4, 4, 1, 1])
                        bias_measure = tf.nn.conv2d( XX , bias_measure_filt , strides=[1, 1, 1, 1], padding='VALID')
                        im_bias = tf.reshape( tf.reduce_min(bias_measure,axis=[1,2,3]) , [-1,1,1,1] )
                        net = net - (im_bias * MASK )



                        net = slim.conv2d(net, 64, [2, 2] , scope='conv1')
                        net = slim.max_pool2d(net, [2, 2], scope='pool1')

                        net = slim.conv2d(net, 64, [2, 2] , scope='conv2')
                        net = slim.max_pool2d(net, [2, 2], scope='pool2')

                        net = slim.conv2d(net, 64, [2, 2], scope='conv3')
                        net = slim.max_pool2d(net, [2, 2], scope='pool3')

                        net = slim.conv2d(net, 128, [3, 3], scope='conv4')
                        net = slim.max_pool2d(net, [2, 2], scope='pool4')

                        net = slim.conv2d(net, 128, [3, 3], scope='conv5')
                        net = slim.max_pool2d(net, [2, 2], scope='pool5')

                        net = slim.conv2d(net, 256, [3, 3], scope='conv6')

                        net = slim.flatten(net, scope='Flatten')
                        net = slim.fully_connected(net, 1024 , activation_fn = tf.tanh ,  scope='FC1')
                        net = slim.fully_connected(net, 128 , activation_fn = tf.tanh ,  scope='FC2')
                        net = slim.fully_connected(net,  5  , activation_fn = None ,  scope='FC3')
                        net = tf.reshape(net,[-1,5])
                        return net


def cost_tensor(y_conv , scale_pars = [ [1., 0., 0., 0., 0.],[0. , 1. , 0., 0., 0.],[0., 0. , 1., 0., 0.],[0., 0. , 0., 1., 0.],[0., 0. , 0., 0., 1.]] ):
	FLIPXY = tf.constant([ [1., 0., 0., 0., 0.],[0. , -1. , 0., 0., 0.],[0., 0. , -1., 0., 0.],[0., 0. , 0., 1., 0.],[0., 0. , 0., 0., 1.]] )
	y_conv_flipped = tf.matmul(y_conv, FLIPXY)
	scale_par_cost =  tf.constant(scale_pars )
	scaled_delta_1 = tf.matmul(tf.pow(y_conv - y_,2) , scale_par_cost)
	scaled_delta_2 = tf.matmul(tf.pow(y_conv_flipped - y_,2) , scale_par_cost)
	MeanSquareCost = tf.reduce_mean( tf.minimum(tf.reduce_mean( scaled_delta_1 ,axis=1) , tf.reduce_mean(  scaled_delta_2 ,axis=1)) , axis=0)
	return MeanSquareCost, y_conv_flipped
	


# def vgg_arg_scope(weight_decay=0.0005):
#   """Defines the VGG arg scope.
#   Args:
#     weight_decay: The l2 regularization coefficient.
#   Returns:
#     An arg_scope.
#   """
#   with slim.arg_scope([slim.conv2d, slim.fully_connected],
#                       activation_fn=tf.nn.relu,
#                       weights_regularizer=slim.l2_regularizer(weight_decay),
#                       biases_initializer=tf.compat.v1.zeros_initializer()):
#     with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
#       return arg_sc


# def vgg_a(inputs,
#           num_classes=1000,
#           is_training=True,
#           dropout_keep_prob=0.5,
#           spatial_squeeze=True,
#           reuse=None,
#           scope='vgg_a',
#           fc_conv_padding='VALID',
#           global_pool=False):
#   """Oxford Net VGG 11-Layers version A Example.
#   Note: All the fully_connected layers have been transformed to conv2d layers.
#         To use in classification mode, resize input to 224x224.
#   Args:
#     inputs: a tensor of size [batch_size, height, width, channels].
#     num_classes: number of predicted classes. If 0 or None, the logits layer is
#       omitted and the input features to the logits layer are returned instead.
#     is_training: whether or not the model is being trained.
#     dropout_keep_prob: the probability that activations are kept in the dropout
#       layers during training.
#     spatial_squeeze: whether or not should squeeze the spatial dimensions of the
#       outputs. Useful to remove unnecessary dimensions for classification.
#     reuse: whether or not the network and its variables should be reused. To be
#       able to reuse 'scope' must be given.
#     scope: Optional scope for the variables.
#     fc_conv_padding: the type of padding to use for the fully connected layer
#       that is implemented as a convolutional layer. Use 'SAME' padding if you
#       are applying the network in a fully convolutional manner and want to
#       get a prediction map downsampled by a factor of 32 as an output.
#       Otherwise, the output prediction map will be (input / 32) - 6 in case of
#       'VALID' padding.
#     global_pool: Optional boolean flag. If True, the input to the classification
#       layer is avgpooled to size 1x1, for any input size. (This is not part
#       of the original VGG architecture.)
#   Returns:
#     net: the output of the logits layer (if num_classes is a non-zero integer),
#       or the input to the logits layer (if num_classes is 0 or None).
#     end_points: a dict of tensors with intermediate activations.
#   """
#   with tf.compat.v1.variable_scope(scope, 'vgg_a', [inputs], reuse=reuse) as sc:
#     end_points_collection = sc.original_name_scope + '_end_points'
#     # Collect outputs for conv2d, fully_connected and max_pool2d.
#     with slim.arg_scope([slim.conv2d, slim.max_pool2d],
#                         outputs_collections=end_points_collection):
#       net = slim.repeat(inputs, 1, slim.conv2d, 64, [3, 3], scope='conv1')
#       net = slim.max_pool2d(net, [2, 2], scope='pool1')
#       net = slim.repeat(net, 1, slim.conv2d, 128, [3, 3], scope='conv2')
#       net = slim.max_pool2d(net, [2, 2], scope='pool2')
#       net = slim.repeat(net, 2, slim.conv2d, 256, [3, 3], scope='conv3')
#       net = slim.max_pool2d(net, [2, 2], scope='pool3')
#       net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope='conv4')
#       net = slim.max_pool2d(net, [2, 2], scope='pool4')
#       net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope='conv5')
#       net = slim.max_pool2d(net, [2, 2], scope='pool5')

#       # Use conv2d instead of fully_connected layers.
#       net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
#       net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
#                          scope='dropout6')
#       net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
#       # Convert end_points_collection into a end_point dict.
#       end_points = slim.utils.convert_collection_to_dict(end_points_collection)
#       if global_pool:
#         net = tf.reduce_mean(
#             input_tensor=net, axis=[1, 2], keepdims=True, name='global_pool')
#         end_points['global_pool'] = net
#       if num_classes:
#         net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
#                            scope='dropout7')
#         net = slim.conv2d(net, num_classes, [1, 1],
#                           activation_fn=None,
#                           normalizer_fn=None,
#                           scope='fc8')
#         if spatial_squeeze:
#           net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
#         end_points[sc.name + '/fc8'] = net
#       return net, end_points
# vgg_a.default_image_size = 224


def vgg_16(inputs,
           num_classes=5,
           is_training=True,
           dropout_keep_prob=1.0,
           spatial_squeeze=True,
           reuse=None,
           scope='vgg_16',
           fc_conv_padding='VALID',
           global_pool=False):
  """Oxford Net VGG 16-Layers version D Example.
  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.
  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes. If 0 or None, the logits layer is
      omitted and the input features to the logits layer are returned instead.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional scope for the variables.
    fc_conv_padding: the type of padding to use for the fully connected layer
      that is implemented as a convolutional layer. Use 'SAME' padding if you
      are applying the network in a fully convolutional manner and want to
      get a prediction map downsampled by a factor of 32 as an output.
      Otherwise, the output prediction map will be (input / 32) - 6 in case of
      'VALID' padding.
    global_pool: Optional boolean flag. If True, the input to the classification
      layer is avgpooled to size 1x1, for any input size. (This is not part
      of the original VGG architecture.)
  Returns:
    net: the output of the logits layer (if num_classes is a non-zero integer),
      or the input to the logits layer (if num_classes is 0 or None).
    end_points: a dict of tensors with intermediate activations.
  """
  with tf.compat.v1.variable_scope(
      scope, 'vgg_16', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
      net = slim.max_pool2d(net, [2, 2], scope='pool4')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
      net = slim.max_pool2d(net, [2, 2], scope='pool5')

      # Use conv2d instead of fully_connected layers.
      net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout6')
      net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      if global_pool:
        net = tf.reduce_mean(
            input_tensor=net, axis=[1, 2], keepdims=True, name='global_pool')
        end_points['global_pool'] = net
      if num_classes:
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout7')
        net = slim.conv2d(net, num_classes, [1, 1],
                          activation_fn=None,
                          normalizer_fn=None,
                          scope='fc8')
        if spatial_squeeze:
          net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        end_points[sc.name + '/fc8'] = net
      return net, end_points
vgg_16.default_image_size = 192


# Alias
vgg_d = vgg_16


def LeNet(images, num_classes=5, is_training=True,
          dropout_keep_prob=1.0,
          prediction_fn=slim.softmax,
          scope='LeNet'):
  """Creates a variant of the LeNet model.

  Note that since the output is a set of 'logits', the values fall in the
  interval of (-infinity, infinity). Consequently, to convert the outputs to a
  probability distribution over the characters, one will need to convert them
  using the softmax function:

        logits = lenet.lenet(images, is_training=False)
        probabilities = tf.nn.softmax(logits)
        predictions = tf.argmax(logits, 1)

  Args:
    images: A batch of `Tensors` of size [batch_size, height, width, channels].
    num_classes: the number of classes in the dataset.
    is_training: specifies whether or not we're currently training the model.
      This variable will determine the behaviour of the dropout layer.
    dropout_keep_prob: the percentage of activation values that are retained.
    prediction_fn: a function to get predictions out of logits.
    scope: Optional variable_scope.

  Returns:
    logits: the pre-softmax activations, a tensor of size
      [batch_size, `num_classes`]
    end_points: a dictionary from components of the network to the corresponding
      activation.
  """
    end_points = {}

    with tf.variable_scope(scope, 'LeNet', [images, num_classes]):
        net = slim.conv2d(images, 32, [5, 5], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], 2, scope='pool1')
        net = slim.conv2d(net, 64, [5, 5], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], 2, scope='pool2')
        net = slim.flatten(net)
        end_points['Flatten'] = net

        net = slim.fully_connected(net, 1024, scope='fc3')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout3')
        logits = slim.fully_connected(net, num_classes, activation_fn=None,
                                      scope='fc4')
        logits = tf.reshape(logits,[-1,5])

    end_points['Logits'] = logits
    end_points['Predictions'] = prediction_fn(logits, scope='Predictions')

    return logits
lenet.default_image_size = 192


def lenet_arg_scope(weight_decay=0.0):
  """Defines the default lenet argument scope.

  Args:
    weight_decay: The weight decay to use for regularizing the model.

  Returns:
    An `arg_scope` to use for the inception v3 model.
  """
  with slim.arg_scope(
      [slim.conv2d, slim.fully_connected],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
      activation_fn=tf.nn.relu) as sc:
    return sc




