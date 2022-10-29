import tensorflow as tf
import os

tf.compat.v1.disable_v2_behavior()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# 创建一张图，包含了一组op（操作）和tensor（张量），上下文环境
# op： 只要使用tensorflow的API定义的函数都是OP
# tensor： 就指代的是数据
# g = tf.compat.v1.Graph()
#
# # print( '图：', g ) #  <tensorflow.python.framework.ops.Graph object at 0x0000021A5A297FD0>
#
# with g.as_default():
#     c = tf.compat.v1.constant(11.0)
#     # print( c.graph ) # <tensorflow.python.framework.ops.Graph object at 0x0000021A5A297FD0>
#
# # 实现一个加法运算
# a = tf.compat.v1.constant(5.0)
# b = tf.compat.v1.constant(6.0)
#
# sum1 = tf.compat.v1.add( a,b )
# # print( sum1 ) # tf.Tensor(11.0, shape=(), dtype=float32)
#
# # 默认的这张图，相当于给程序分配一段内存
# graph = tf.compat.v1.get_default_graph()
# print( graph ) # <tensorflow.python.framework.ops.Graph object at 0x000001C0E7F82A30>

# 只能运行一张图，Session中不指定图，则使用默认的图
# with tf.compat.v1.Session( config=tf.compat.v1.ConfigProto(log_device_placement=True) ) as sess:
#     print(sess.run(sum1)) # 11.0
#     print(a.graph) # <tensorflow.python.framework.ops.Graph object at 0x000001C29B43DFA0>
#     print(sum1.graph) # <tensorflow.python.framework.ops.Graph object at 0x000001C29B43DFA0>
#     print(sess.graph) # <tensorflow.python.framework.ops.Graph object at 0x000001C29B43DFA0>


### 变量op
# 1. 变量op能够持久化保存，普通张量op是不行的
# 2. 当定义一个变量op的时候，一定要在会话当中运行初始化op

# a = tf.compat.v1.constant(3.0,name='a') # Tensor("Const:0", shape=(5,), dtype=int32)
# b = tf.compat.v1.constant(4.0,name='b')
#
# sum1 = tf.add( a,b ,name='add')
#
# # 随机初始化一个2行3列的矩阵，且平均值为0，标准差为1
# var = tf.compat.v1.Variable(tf.compat.v1.random_normal([2, 3], mean=0.0, stddev=1.0), name='variable') # <tf.Variable 'Variable:0' shape=(2, 3) dtype=float32_ref>
#
# print(a, var)
# # 必须做一步显示的初始化op
# init_op = tf.compat.v1.global_variables_initializer()
#
# with tf.compat.v1.Session() as sess:
#     # 必须运行初始化op
#     sess.run(init_op)
#     # 把程序的图结构写入事件文件
#     file_writer = tf.compat.v1.summary.FileWriter('../tmp/summary/test/', graph= sess.graph )
#
#     # [array([1, 2, 3, 4, 5]), array([[ 0.19624615, -1.6842716 , -0.16053599],
#     #        [-0.97141266,  0.8143034 ,  0.20694952]], dtype=float32)]
#     print(sess.run([sum1, var]))


"""
1. 训练参数的问题： trainable
    学习率和步数的设置
    
2. 添加权重参数，损失值等在tensorboard观察的情况： 
    1. 收集变量。
    2.合并变量写入事件文件

3. 定义命令行参数
    1. 首先定义有哪些参数需要在运行时指定
    2. 程序当中获取定义命令行参数
"""

#  定义命令行参数
tf.compat.v1.flags.DEFINE_integer('max_step', 100, '模型训练的步数')
tf.compat.v1.flags.DEFINE_string('model_dir', ' ', '模型文件的加载路径')


def myregression():
    """
    自定义的线性回归算法
    :return:  None
    """
    with tf.compat.v1.variable_scope('data'):
        # 1. 准备数据
        # 样本特征值
        x = tf.compat.v1.random_normal([100, 1], mean=1.75, stddev=0.5, name='x_data')
        # 目标值
        y_true = tf.compat.v1.matmul(x, [[0.7]]) + 0.8

    with tf.compat.v1.variable_scope('model'):
        # 2. 建立线性回归模型， 1个特征，一个权重，一个偏重
        # 随机给一个权重和偏置的值，让他去计算损失，然后在当前状态下优化
        # 用变量定义才能优化
        weight = tf.compat.v1.Variable(tf.compat.v1.random_normal([1, 1], mean=0.0, stddev=1.0), name='w')
        bias = tf.compat.v1.Variable(0.0, name='b')

        y_predict = tf.compat.v1.matmul(x, weight) + bias

    with tf.compat.v1.variable_scope('loss'):
        # 3. 建立损失函数，均方误差
        loss = tf.compat.v1.reduce_mean(tf.compat.v1.square(y_true - y_predict))

    with tf.compat.v1.variable_scope('optimizer'):
        # 4. 梯度下降优化损失， learning_rate: 0-1
        train_op = tf.compat.v1.train.GradientDescentOptimizer(0.1).minimize(loss)

    # 1. 收集tensor
    tf.compat.v1.summary.scalar('losses', loss)
    tf.compat.v1.summary.histogram('weights', weight)
    tf.compat.v1.summary.histogram('biases', bias)

    # 定义合并tensor的op
    merged = tf.compat.v1.summary.merge_all()

    # 定义一个初始化变量的op
    init_op = tf.compat.v1.global_variables_initializer()

    # 定义一个模型的保存和加载的op
    saver = tf.compat.v1.train.Saver()

    # 5. 会话运行程序
    with tf.compat.v1.Session() as sess:
        # 初始化变量
        sess.run(init_op)

        # 打印随机最先初始化的权重和偏置
        print(f"随机初始化的参数权重为：${weight.eval()}, 偏置为${bias.eval()}")
        file_writer = tf.compat.v1.summary.FileWriter('../tmp/summary/test/', graph=sess.graph)

        # 在模型训练之前，加载保存的模型，从上一次训练的结果开始训练
        if os.path.exists('../tmp/ckpt/checkpoint'):
            saver.restore(sess, '../tmp/ckpt/model')
        # 循环训练，运行优化
        for i in range(500):
            sess.run(train_op)
            # 运行合并的tensor
            summary = sess.run(merged)
            file_writer.add_summary(summary, i)
            print(f"第{i + 1}次的参数权重为：{weight.eval()}, 偏置为{bias.eval()}")

        # 全部训练完成后，保存模型
        saver.save(sess, '../tmp/ckpt/model')
    return None


if __name__ == '__main__':
    myregression()
