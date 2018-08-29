#coding=utf-8
import argparse
import os
import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 防止运行时出现莫名其妙的Warning

FLAGS = None

Output_channel=N

def deepnn(x):
    # input输入是10000×1 
    inputs = tf.reshape(x, [-1, 10000, 1])

    # 第一个卷积层 
    W_conv1 = weight_variable([9, 1, 1, N]) #卷积核是 9*1, 输入通道1 输出通道N
    b_conv1 = bias_variable([N])
    h_conv1 = tf.nn.relu(conv1d(inputs, W_conv1) + b_conv1)  # ReLU

    # 第二个卷积层 - 输出 N 张feature maps
    W_conv2 = weight_variable([9, 1, N, N])
    b_conv2 = bias_variable([N])
    h_conv2 = tf.nn.relu(conv1d(h_conv1, W_conv2) + b_conv2)  # ReLU
    
    # 第三个卷积层 - 输出 N 张feature maps
    W_conv3 = weight_variable([9, 1, N, N])
    b_conv3 = bias_variable([N])
    h_conv3 = tf.nn.relu(conv1d(h_conv2, W_conv3) + b_conv3)  # ReLU

    # Dropout
    keep_prob = tf.placeholder(tf.float32)
    h_conv3_drop = tf.nn.dropout(h_conv3, keep_prob)

    # 全连接层1 - 经过两轮的降采样后，10000*1的输入变成了10000*1*9 个（特征图）.
    # 现在将这些feature maps变成4000个features（扁平化）.
    W_fc1 = weight_variable([10000* 1 * 9, 4000])
    b_fc1 = bias_variable([4000])
    h_conv3_drop_flat = tf.reshape(h_conv3_drop, [-1, 10000 * 1 * 9])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_drop_flat, W_fc1) + b_fc1)

    # Dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    return h_fc1_drop, keep_prob  #结果和占位符都要输出


def conv1d(x, W):
    # 步长为1，进行零填充. 权重W也就是filter
    return tf.nn.conv1d(x, W, strides=[1, 1, 1], padding='SAME')

def weight_variable(shape):
    # 生成权重，形状为参数shape
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    # 生成偏置，形状为参数shape
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def main(_):
    # 加载数据集 要把输入改成批次
    inputs =..
    #mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    
    #输入和输出
    x = tf.placeholder(tf.float32, [None, 10000])
    y_ = tf.placeholder(tf.float32, [None, 4000])

    y_conv, keep_prob = deepnn(x) # 这里依然是网络的结构吧，不是输出

    #这里的是还没有化为0 1的值
    cross_entropy = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y_conv) ) #计算交叉熵, sigmoid_cross_entropy.. 内部会对 logits 使用sigmoid操作
   
    #训练参数
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) #梯度下降的优化器,参数是学习速率i
    
    #根据阈值 转换成 0 1 模型
    # _y应该是 -1*4000 形状的
    #准确度 怎么算呢 用来评价参数，不会影响模型的训练
    correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y_conv, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #运行
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer()) #初始化变量
        #运行的批次
        for i in range(50):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                #每次在计算图中计算准确率
                train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
                print 'step', i, 'training accuracy', train_accuracy
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        
        #准确率是在验证集上使用的，没有drop层
        test_accuracy = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
        print 'test accuracy', test_accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='mnist_data',
                        help='Directory for storing input data.')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
