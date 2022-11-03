import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.compat.v1.disable_eager_execution()

FLAGS = tf.compat.v1.flags.FLAGS
tf.compat.v1.flags.DEFINE_integer('is_train', 1, '指定程序是预测还是训练')


def full_connected():
    # 获取真实的数据
    mnist = input_data.read_data_sets('../data/mnist/', one_hot=True)

    # 1. 建立数据的占位符  x[None, 784]   y_true [None,10]
    with tf.compat.v1.variable_scope('data'):
        x = tf.compat.v1.placeholder(tf.float32, [None, 784])
        y_true = tf.compat.v1.placeholder(tf.int32, [None, 10])

    # 2. 建立一个全连接层的神经网络  w [784,10]   b [10]
    with tf.compat.v1.variable_scope('fc_model'):
        # 随机初始化权重和偏置
        weight = tf.compat.v1.Variable(tf.compat.v1.random_normal([784, 10], mean=0.0, stddev=1.0), name='w')
        bias = tf.compat.v1.Variable(tf.compat.v1.constant(0.0, shape=[10]), name='bias')

        # 预测None个样本的输出结果matrix   [None,784]  * [784, 10] + [10] = [None,10]
        y_predict = tf.compat.v1.matmul(x, weight) + bias

    # 3. 求出所有样本的损失，然后求平均值
    with tf.compat.v1.variable_scope('soft_cross'):
        # 求平均交叉熵损失
        loss = tf.compat.v1.reduce_mean(
            tf.compat.v1.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict))

    # 4. 梯度下降求出损失
    with tf.compat.v1.variable_scope('optimizer'):
        # 0.1 为学习率，minmizer( loss )表示最小化损失值
        train_op = tf.compat.v1.train.GradientDescentOptimizer(0.1).minimize(loss)

    # 5. 计算准确率
    with tf.compat.v1.variable_scope('acc'):
        equal_list = tf.compat.v1.equal(tf.argmax(y_true, 1), tf.argmax(y_predict, 1))
        # equal_list就有None个样本，  [1,0,...,...]
        accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    # 收集变量，当额数字值收集
    tf.compat.v1.summary.scalar('losses', loss)
    tf.compat.v1.summary.scalar('acc', accuracy)
    # 高维度变量收集
    tf.compat.v1.summary.histogram('weights', weight)
    tf.compat.v1.summary.histogram('biases', bias)

    # 初始化变量op
    init_op = tf.compat.v1.global_variables_initializer()

    # 定义一个合并变量的op
    merged = tf.compat.v1.summary.merge_all()

    # 定义保存模型的op
    saver = tf.compat.v1.train.Saver()

    # 开启会话训练
    with tf.compat.v1.Session() as sess:
        # 初始化变量
        sess.run(init_op)

        # 建立events文件，然后写入
        fileWriter = tf.compat.v1.summary.FileWriter('../tmp/summary/test/', graph=sess.graph)

        if FLAGS.is_train == 1:
            # 迭代步数去训练，更新参数预测
            for i in range(2000):
                # 取出真实存在的特征值和目标值
                mnist_x, mnist_y = mnist.train.next_batch(50)

                # 运行train_op训练
                sess.run(train_op, feed_dict={x: mnist_x, y_true: mnist_y})

                # 写入每步训练的值
                summary = sess.run(fetches=merged, feed_dict={x: mnist_x, y_true: mnist_y})
                fileWriter.add_summary(summary, i)

                print(f'训练第{i}步， 准确率为{sess.run(accuracy, feed_dict={x: mnist_x, y_true: mnist_y})}')

            # 保存模型
            saver.save(sess, '../tmp/ckpt/fc_model')
        else:
            # 如果是0， 做出预测
            # 加载模型
            saver.restore(sess, '../tmp/ckpt/fc_model')

            for i in range(100):
                # 每次测试一张图片
                x_test, y_test = mnist.test.next_batch(1)

                print(f'第{i}张图片， 手写数字图片目标是{tf.argmax(y_test, 1).eval()}, '
                      f'预测的结果是{tf.argmax(sess.run(y_predict, feed_dict={x: x_test, y_true: y_test}), 1).eval()}')

    return None


# 定义一个初始化权重的函数
def weight_variables(shape):
    w = tf.compat.v1.Variable(tf.compat.v1.random_normal(shape=shape, mean=0.0, stddev=1.0))
    return w


# 定义一个初始化偏置的函数
def bias_variables(shape):
    b = tf.compat.v1.Variable(tf.compat.v1.constant(0.0, shape=shape))
    return b


def model():
    """
    自定义的卷积模型
    :return:
    """
    # 1. 建立数据的占位符  x[None, 784]   y_true [None,10]
    with tf.compat.v1.variable_scope('data'):
        x = tf.compat.v1.placeholder(tf.float32, [None, 784])
        y_true = tf.compat.v1.placeholder(tf.int32, [None, 10])

    # 2. 一卷积层： 5*5*1,32个filters， strides=1， 激活：relu   ，池化
    with tf.compat.v1.variable_scope('conv1'):
        # 随机初始化权重， 偏置[32]
        # height, width, 输入channel, 输出channel
        w_conv1 = weight_variables([5, 5, 1, 32])
        b_conv1 = bias_variables([32])

        # 对x进行形状的改变[None,784] ---> [None,28,28,1]
        x_reshape = tf.compat.v1.reshape(x, [-1, 28, 28, 1])

        # 进行卷积,激活，维度从[None,28,28,1] ---> [None,28,28,32]
        x_relu1 = tf.compat.v1.nn.relu(
            tf.compat.v1.nn.conv2d(x_reshape, w_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)

        # 池化 2*2,strides = 2, [None,28,28,32] ---> [None,14,14,32]
        x_pool1 = tf.compat.v1.nn.max_pool(x_relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 3. 二卷积层： 5*5*32,64个filters， strides=1， 激活：relu   ，池化
    with tf.compat.v1.variable_scope('conv2'):
        # 随机初始化权重， 偏置[64]
        w_conv2 = weight_variables([5, 5, 32, 64])
        b_conv2 = bias_variables([64])

        # 卷积、激活、池化
        # [None, 14,14,32]  ---> [None,14,14,64]
        x_relu2 = tf.compat.v1.nn.relu(
            tf.compat.v1.nn.conv2d(x_pool1, w_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)

        # 池化 2*2, strides = 2, [None,14,14,64] ---> [None,7,7,64]
        x_pool2 = tf.compat.v1.nn.max_pool(x_relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 4. 全连接层 [None,7,7,64] ---> [None, 7*7*64] ---> *[7*7*64, 10] ---> [None,10]
    with tf.compat.v1.variable_scope('fc'):
        # 随机初始化权重， 偏置[10]
        w_tf = weight_variables([7 * 7 * 64, 10])
        b_tf = bias_variables([10])

        # 修改形状
        x_tf_reshape = tf.compat.v1.reshape(x_pool2, [-1, 7 * 7 * 64])

        # 进行矩阵运算得出每个样本的10个结果
        y_predict = tf.compat.v1.matmul(x_tf_reshape, w_tf) + b_tf

    return x, y_true, y_predict


def conv_fc():
    # 获取真实的数据
    mnist = input_data.read_data_sets('../data/mnist/', one_hot=True)

    # 定义模型，得出输出
    x, y_true, y_predict = model()

    # 进行交叉熵损失计算
    # 3. 求出所有样本的损失，然后求平均值
    with tf.compat.v1.variable_scope('soft_cross'):
        # 求平均交叉熵损失
        loss = tf.compat.v1.reduce_mean(
            tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_predict))

    # 4. 梯度下降求出损失
    with tf.compat.v1.variable_scope('optimizer'):
        # 0.1 为学习率，minmizer( loss )表示最小化损失值
        train_op = tf.compat.v1.train.GradientDescentOptimizer(0.001).minimize(loss)

    # 5. 计算准确率
    with tf.compat.v1.variable_scope('acc'):
        equal_list = tf.compat.v1.equal(tf.argmax(y_true, 1), tf.argmax(y_predict, 1))
        # equal_list就有None个样本，  [1,0,...,...]
        accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    # 初始化变量op
    init_op = tf.compat.v1.global_variables_initializer()
    # 开启会话运行
    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        # 循环训练
        for i in range(1000):
            # 取出真实存在的特征值和目标值
            mnist_x, mnist_y = mnist.train.next_batch(50)

            # 运行train_op
            sess.run(train_op, feed_dict={x: mnist_x, y_true: mnist_y})
            print(f'训练第{i}步， 准确率为{sess.run(accuracy, feed_dict={x: mnist_x, y_true: mnist_y})}')

    return None


if __name__ == '__main__':
    # full_connected()
    conv_fc()
