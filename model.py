import tensorflow as tf
import numpy as np

from tensorflow.contrib import slim

tf.app.flags.DEFINE_integer('text_scale', 512, '')
tf.app.flags.DEFINE_integer('resnet_version', 1, 'the version of ResNet')
tf.app.flags.DEFINE_integer('resnet_size', 8, 'n: the size of ResNet-(6n+2) v1 or ResNet-(9n+2) v2')
tf.app.flags.DEFINE_float('weight_decay', 1e-5, 'weight decay rate')
tf.app.flags.DEFINE_integer('ratio',16,'ratio of SENet')
#from nets import resnet_v1

FLAGS = tf.app.flags.FLAGS


def unpool(inputs):
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*2,  tf.shape(inputs)[2]*2])

def my_inception(inputs,c,is_training):
    batch_norm_params = {
        'decay': 0.997,
        'epsilon': 1e-5,
        'scale': True,
        'is_training': is_training
        }
    tmp1 = slim.conv2d(inputs,c,[3,3],activation_fn=tf.nn.relu,normalizer_params=batch_norm_params,weights_regularizer=slim.l2_regularizer(1e-5))
    tmp2 = slim.conv2d(inputs,c,[1,5],activation_fn=tf.nn.relu,normalizer_params=batch_norm_params,weights_regularizer=slim.l2_regularizer(1e-5))
    tmp3 = slim.conv2d(inputs,c,[1,7],activation_fn=tf.nn.relu,normalizer_params=batch_norm_params,weights_regularizer=slim.l2_regularizer(1e-5))
    tmp4 = slim.conv2d(inputs,c,[1,9],activation_fn=tf.nn.relu,normalizer_params=batch_norm_params,weights_regularizer=slim.l2_regularizer(1e-5))
    return tf.concat([tmp1,tmp2,tmp3,tmp4],axis=-1)

def mean_image_subtraction(images, means=[116.82, 121.20, 126.52]):
    '''
   image normalization
    :param images:
    :param means:
   :return:
    '''
    num_channels = images.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=images)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)

def model(images, is_training=True):
    '''
    define the model, we use slim's implemention of resnet
    '''
    images = mean_image_subtraction(images)

    output1 = start_layer(images, is_training)

    if FLAGS.resnet_version == 1:
        block_fn = "standard_block_v1"
    else:
        block_fn = "bottleneck_block_v2"
    outputs = [None, None, None, None]
    outputs[0] = output1
    for i in range(3):
        filters = 64 * (2 ** i)
        strides = 2
        outputs[i+1] = stack_layer(outputs[i], filters, block_fn, strides, is_training)
    for i in range(4):
        print('Shape of outputs_{} {}'.format(i, outputs[i].shape))

    outputs_merge = [None, None, None]
    if block_fn == "standard_block_v1":
        outputs_merge[0] = tf.layers.conv2d(tf.concat([outputs[2],unpool(outputs[3])], axis=-1), 256, 1, 1, "same")
        outputs_merge[0] = batch_norm_relu(outputs_merge[0], is_training)
        outputs_merge[0] = tf.layers.conv2d(outputs_merge[0], 256, 3, 1, "same")
        outputs_merge[0] = batch_norm_relu(outputs_merge[0], is_training)

        outputs_merge[1] = tf.layers.conv2d(tf.concat([outputs[1], unpool(outputs_merge[0])], axis=-1), 128, 1, 1, "same")
        outputs_merge[1] = batch_norm_relu(outputs_merge[1], is_training)
        outputs_merge[1] = tf.layers.conv2d(outputs_merge[1], 128, 3, 1, "same")
        outputs_merge[1] = batch_norm_relu(outputs_merge[1], is_training)

        outputs_merge[2] = tf.layers.conv2d(tf.concat([outputs[0], unpool(outputs_merge[1])], axis=-1), 64, 1, 1, "same")
        outputs_merge[2] = batch_norm_relu(outputs_merge[2], is_training)
        outputs_merge[2] = tf.layers.conv2d(outputs_merge[2], 64, 3, 1, "same")
        outputs_merge[2] = batch_norm_relu(outputs_merge[2], is_training)
    for i in range(3):
        print('Shape of outputs_merge_{} {}'.format(i, outputs_merge[i].shape))

    d = [None, None]
    d[0] = tf.layers.conv2d(outputs_merge[2], 32, 3, 1, "same")
    d[0] = batch_norm_relu(d[0], is_training)
    d[1] = tf.layers.conv2d(outputs_merge[2], 32, 3, 1, "same")
    d[1] = batch_norm_relu(d[1], is_training)
    for i in range(2):
        print('Shape of d_{} {}'.format(i,d[i].shape))
    F_score = tf.layers.conv2d(d[0], 1, 1)
    F_score = tf.nn.sigmoid(F_score)
    # 4 channel of axis aligned bbox and 1 channel rotation angle
    geo_map = tf.layers.conv2d(d[1], 4, 1)
    geo_map = tf.nn.sigmoid(geo_map)*FLAGS.text_scale
    angle_map = tf.layers.conv2d(d[1], 1, 1)   # angle is between [-45, 45]
    angle_map = (tf.nn.sigmoid(angle_map)-0.5)*np.pi/2
    F_geometry = tf.concat([geo_map, angle_map], axis=-1)

    return F_score, F_geometry

def batch_norm_relu(inputs, training):
    """Perform batch normalization then relu."""

    ### YOUR CODE HERE
    outputs = tf.layers.batch_normalization(inputs, axis=-1, training=training)
    outputs = tf.nn.relu(outputs)
    ### END CODE HERE

    return outputs

def se_layer(input_x, out_dim):
    squeeze = tf.reduce_mean(input_x, [1,2])

    excitation = tf.layers.dense(squeeze, out_dim / FLAGS.ratio)
    excitation = tf.nn.relu(excitation)
    excitation = tf.layers.dense(excitation, out_dim)
    excitation = tf.nn.sigmoid(excitation)

    excitation = tf.reshape(excitation, [-1,1,1,out_dim])

    scale = input_x * excitation

    return scale


def start_layer(inputs, training):
    """Implement the start layer.

    Args:
        inputs: A Tensor of shape [<batch_size>, 32, 32, 3].
        training: A boolean. Used by operations that work differently
            in training and testing phases.

    Returns:
        outputs: A Tensor of shape [<batch_size>, 32, 32, self.first_num_filters].
    """

    ### YOUR CODE HERE
    # initial conv1
    outputs = tf.layers.conv2d(inputs, 32, 3, 1, "same")
    outputs = batch_norm_relu(outputs, training)
    outputs = tf.layers.conv2d(outputs, 32, 3, 1, "same")
    outputs = batch_norm_relu(outputs, training)
    outputs = tf.layers.conv2d(outputs, 32, 3, 1, "same")
    outputs = batch_norm_relu(outputs, training)
    outputs = tf.layers.conv2d(outputs, 32, 3, 2, "same")
    ### END CODE HERE

    # We do not include batch normalization or activation functions in V2
    # for the initial conv1 because the first block unit will perform these
    # for both the shortcut and non-shortcut paths as part of the first
    # block's projection.
    if FLAGS.resnet_version == 1:
        outputs = batch_norm_relu(outputs, training)

    return outputs

def stack_layer(inputs, filters, block_fn, strides, training):
    """Creates one stack of standard blocks or bottleneck blocks.

    Args:
        inputs: A Tensor of shape [<batch_size>, height_in, width_in, channels].
        filters: A positive integer. The number of filters for the first
            convolution in a block.
        block_fn: 'standard_block' or 'bottleneck_block'.
        strides: A positive integer. The stride to use for the first block. If
            greater than 1, this layer will ultimately downsample the input.
        training: A boolean. Used by operations that work differently
            in training and testing phases.

    Returns:
        outputs: The output tensor of the block layer.
    """

    filters_out = filters * 4 if FLAGS.resnet_version == 2 else filters

    def projection_shortcut(inputs):
        ### YOUR CODE HERE
        inputs_depth = inputs.shape.as_list()[3]
        return tf.layers.conv2d(inputs, 2 * inputs_depth, 1, 2, "same")

    ### END CODE HERE

    ### YOUR CODE HERE
    # Only the first block per stack_layer uses projection_shortcut
    outputs = inputs
    for i in range(FLAGS.resnet_size):
        if i == 0:
            if block_fn == "bottleneck_block_v2":
                outputs = bottleneck_block_v2(outputs, filters_out, training, projection_shortcut, strides)
            else:
                if strides == 2:
                    outputs = standard_block_v1(outputs, filters_out, training, projection_shortcut, strides)
                else:
                    outputs = standard_block_v1(outputs, filters_out, training, None, 1)
        else:
            if block_fn == "bottleneck_block_v2":
                outputs = bottleneck_block_v2(outputs, filters_out, training, None, 1)
            else:
                outputs = standard_block_v1(outputs, filters_out, training, None, 1)
    ### END CODE HERE

    return outputs


def standard_block_v1(inputs, filters, training, projection_shortcut, strides):
    """Creates a standard residual block for ResNet v1.

    Args:
        inputs: A Tensor of shape [<batch_size>, height_in, width_in, channels].
        filters: A positive integer. The number of filters for the first
            convolution.
        training: A boolean. Used by operations that work differently
            in training and testing phases.
        projection_shortcut: The function to use for projection shortcuts
              (typically a 1x1 convolution when downsampling the input).
        strides: A positive integer. The stride to use for the block. If
            greater than 1, this block will ultimately downsample the input.

    Returns:
        outputs: The output tensor of the block layer.
    """

    shortcut = inputs

    if projection_shortcut is not None:
        ### YOUR CODE HERE
        shortcut = projection_shortcut(inputs)
        shortcut = tf.layers.batch_normalization(shortcut, axis=-1, training=training)
    ### END CODE HERE

    ### YOUR CODE HERE
    residual_output = tf.layers.conv2d(inputs, filters, 3, strides, 'same')
    residual_output = batch_norm_relu(residual_output, training)
    residual_output = tf.layers.conv2d(residual_output, filters, 3, 1, 'same')
    residual_output = tf.layers.batch_normalization(residual_output, axis=-1, training=training)
    residual_output = se_layer(residual_output,filters)
    outputs = tf.nn.relu(residual_output + shortcut)
    ### END CODE HERE

    return outputs


def bottleneck_block_v2(inputs, filters, training, projection_shortcut, strides):
    """Creates a bottleneck block for ResNet v2.

    Args:
        inputs: A Tensor of shape [<batch_size>, height_in, width_in, channels].
        filters: A positive integer. The number of filters for the first
            convolution. NOTE: filters_out will be 4xfilters.
        training: A boolean. Used by operations that work differently
            in training and testing phases.
        projection_shortcut: The function to use for projection shortcuts
              (typically a 1x1 convolution when downsampling the input).
        strides: A positive integer. The stride to use for the block. If
            greater than 1, this block will ultimately downsample the input.

    Returns:
        outputs: The output tensor of the block layer.
    """

    ### YOUR CODE HERE
    shortcut = inputs
    shortcut = self._batch_norm_relu(inputs,
                                     training)  # when pre-activation is used, these projection shortcuts are also with pre-activation
    if projection_shortcut is not None:
        shortcut = tf.layers.conv2d(shortcut, filters, 1, strides, "same")
    residual_output = self._batch_norm_relu(inputs, training)
    residual_output = tf.layers.conv2d(inputs, filters / 4, 1, 1, 'same')
    residual_output = self._batch_norm_relu(residual_output, training)
    residual_output = tf.layers.conv2d(residual_output, filters / 4, 3, strides, 'same')
    residual_output = self._batch_norm_relu(residual_output, training)
    residual_output = tf.layers.conv2d(residual_output, filters, 1, 1, 'same')

    outputs = residual_output + shortcut
    ### END CODE HERE

    return outputs

def dice_coefficient(y_true_cls, y_pred_cls,
                     training_mask):
    '''
    dice loss
    :param y_true_cls:
    :param y_pred_cls:
    :param training_mask:
    :return:
    '''
    eps = 1e-5
    intersection = tf.reduce_sum(y_true_cls * y_pred_cls * training_mask)
    union = tf.reduce_sum(y_true_cls * training_mask) + tf.reduce_sum(y_pred_cls * training_mask) + eps
    loss = 1. - (2 * intersection / union)
    tf.summary.scalar('classification_dice_loss', loss)
    return loss



def loss(y_true_cls, y_pred_cls,
         y_true_geo, y_pred_geo,
         training_mask):
    '''
    define the loss used for training, contraning two part,
    the first part we use dice loss instead of weighted logloss,
    the second part is the iou loss defined in the paper
    :param y_true_cls: ground truth of text
    :param y_pred_cls: prediction os text
    :param y_true_geo: ground truth of geometry
    :param y_pred_geo: prediction of geometry
    :param training_mask: mask used in training, to ignore some text annotated by ###
    :return:
    '''
    classification_loss = dice_coefficient(y_true_cls, y_pred_cls, training_mask)
    # scale classification loss to match the iou loss part
    classification_loss *= 0.01

    # d1 -> top, d2->right, d3->bottom, d4->left
    d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = tf.split(value=y_true_geo, num_or_size_splits=5, axis=3)
    d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = tf.split(value=y_pred_geo, num_or_size_splits=5, axis=3)
    area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
    area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
    w_union = tf.minimum(d2_gt, d2_pred) + tf.minimum(d4_gt, d4_pred)
    h_union = tf.minimum(d1_gt, d1_pred) + tf.minimum(d3_gt, d3_pred)
    area_intersect = w_union * h_union
    area_union = area_gt + area_pred - area_intersect
    L_AABB = -tf.log((area_intersect + 1.0)/(area_union + 1.0))
    L_theta = 1 - tf.cos(theta_pred - theta_gt)
    tf.summary.scalar('geometry_AABB', tf.reduce_mean(L_AABB * y_true_cls * training_mask))
    tf.summary.scalar('geometry_theta', tf.reduce_mean(L_theta * y_true_cls * training_mask))
    L_g = L_AABB + 20 * L_theta
    l2_loss = FLAGS.weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'kernel' in v.name])
    tf.summary.scalar('l2_loss', l2_loss)
    return tf.reduce_mean(L_g * y_true_cls * training_mask) + classification_loss+l2_loss
    #return tf.reduce_sum(L_g * y_true_cls * training_mask)/tf.cast(tf.count_nonzero(y_true_cls*training_mask), tf.float32) + classification_loss + l2_loss
