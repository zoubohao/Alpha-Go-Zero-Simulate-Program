import tensorflow as tf
import numpy as np


def weight_variable(shape):
    """Xavier initialization"""
    stddev = np.sqrt(0.2 / (sum(shape)))
    initial = tf.truncated_normal(shape, stddev=stddev)
    weights = tf.Variable(initial)
    tf.add_to_collection(tf.GraphKeys.WEIGHTS, weights)
    return weights


def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, data_format='NCHW',
                         strides=[1, 1, 1, 1], padding='SAME')

class LeelaZeroNet :

    def __init__(self):
        self.training = tf.placeholder(dtype=tf.bool)
        self.RESIDUAL_FILTERS = 256
        self.RESIDUAL_BLOCKS = 19
        self.inputPlaceHolder = tf.placeholder(dtype=tf.float32,shape=[None,18,19,19])
        self.porbPlaceHolder = tf.placeholder(dtype=tf.float32,shape=[None,19*19+1])
        self.valuePlaceHolder = tf.placeholder(dtype=tf.float32,shape=[None,1])
        self.policyNet , self.valueNet = self.construct_net(planes=self.inputPlaceHolder)
        self.lossAndOptConstruction()



    def batch_norm(self, net):
        # The weights are internal to the batchnorm layer, so apply
        # a unique scope that we can store, and use to look them back up
        # later on.

       net = tf.layers.batch_normalization(
            net,
            epsilon=1e-5, axis=1, fused=True,
            center=True, scale=False,
            training=self.training)
       return net

    def conv_block(self, inputs, filter_size, input_channels, output_channels):
        W_conv = weight_variable([filter_size, filter_size,
                                  input_channels, output_channels])
        #self.addWeightToCollection(W_conv)

        net = inputs
        net = conv2d(net, W_conv)
        net = self.batch_norm(net)
        net = tf.nn.relu(net)
        return net

    def residual_block(self, inputs, channels):
        net = inputs
        orig = tf.identity(net)

        # First convnet weights
        W_conv_1 = weight_variable([3, 3, channels, channels])
        #self.addWeightToCollection(W_conv_1)

        net = conv2d(net, W_conv_1)
        net = self.batch_norm(net)
        net = tf.nn.relu(net)

        # Second convnet weights
        W_conv_2 = weight_variable([3, 3, channels, channels])
        #self.addWeightToCollection(W_conv_2)

        net = conv2d(net, W_conv_2)
        net = self.batch_norm(net)
        net = tf.add(net, orig)
        net = tf.nn.relu(net)

        return net

    def construct_net(self, planes):
        # NCHW format
        # batch, 17 channels, 19 x 19
        x_planes = tf.reshape(planes, [-1, 18, 19, 19])

        # Input convolution
        flow = self.conv_block(x_planes, filter_size=3,
                               input_channels=18,
                               output_channels=self.RESIDUAL_FILTERS)
        # Residual tower
        for _ in range(0, self.RESIDUAL_BLOCKS):
            flow = self.residual_block(flow, self.RESIDUAL_FILTERS)

        # Policy head
        conv_pol = self.conv_block(flow, filter_size=1,
                                   input_channels=self.RESIDUAL_FILTERS,
                                   output_channels=2)
        h_conv_pol_flat = tf.reshape(conv_pol, [-1, 2 * 19 * 19])
        W_fc1 = weight_variable([2 * 19 * 19, (19 * 19) + 1])
        b_fc1 = bias_variable([(19 * 19) + 1])
        #self.addWeightToCollection(W_fc1)
        #self.addWeightToCollection(b_fc1)
        h_fc1 = tf.add(tf.matmul(h_conv_pol_flat, W_fc1), b_fc1)

        # Value head
        conv_val = self.conv_block(flow, filter_size=1,
                                   input_channels=self.RESIDUAL_FILTERS,
                                   output_channels=1)
        h_conv_val_flat = tf.reshape(conv_val, [-1, 19 * 19])
        W_fc2 = weight_variable([19 * 19, 256])
        b_fc2 = bias_variable([256])
        #self.addWeightToCollection(W_fc2)
       # self.addWeightToCollection(b_fc2)
        h_fc2 = tf.nn.relu(tf.add(tf.matmul(h_conv_val_flat, W_fc2), b_fc2))
        W_fc3 = weight_variable([256, 1])
        b_fc3 = bias_variable([1])
       # self.addWeightToCollection(W_fc3)
       # self.addWeightToCollection(b_fc3)
        h_fc3 = tf.nn.tanh(tf.add(tf.matmul(h_fc2, W_fc3), b_fc3))

        return h_fc1, h_fc3

    def lossAndOptConstruction(self):
        cross = tf.nn.softmax_cross_entropy_with_logits(logits=self.policyNet
                                                        ,labels=self.porbPlaceHolder)
        self.lossP = lossP = tf.reduce_mean(cross)
        self.lossV = lossV = tf.reduce_mean(tf.squared_difference(self.valueNet,self.valuePlaceHolder))
        reg = tf.contrib.layers.l2_regularizer(scale=0.0001)
        regVar = tf.get_collection(tf.GraphKeys.WEIGHTS)
        regm = tf.contrib.layers.apply_regularization(reg, regVar)
        self.lossT = 1.0 * lossP + 1.0 * lossV + regm
        #self.opt = opt = tf.train.AdamOptimizer(learning_rate=0.0001 )
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.opt = opt = tf.train.MomentumOptimizer(momentum=0.9,learning_rate=0.01,
                                                        use_nesterov=True).minimize(self.lossT)


        #correct_prediction = \
        #    tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
        #correct_prediction = tf.cast(correct_prediction, tf.float32)
        #self.accuracy = tf.reduce_mean(correct_prediction)

        # Summary part
        #self.test_writer = tf.summary.FileWriter(
        #    os.path.join(os.getcwd(),
        #                 self.logbase + "/test"), self.session.graph)
        #self.train_writer = tf.summary.FileWriter(
        #    os.path.join(os.getcwd(),
        #                self.logbase + "/train"), self.session.graph)

        # Build checkpoint saver
        #self.saver = tf.train.Saver()




if __name__ == "__main__":
    leelaNet = LeelaZeroNet()