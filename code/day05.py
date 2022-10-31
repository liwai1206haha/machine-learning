import os

import tensorflow as tf

tf.compat.v1.disable_eager_execution()
"""
模拟以下同步先处理数据，然后才能取数据训练
 
"""
# 1. 自定义队列
# Q = tf.compat.v1.FIFOQueue(3, tf.float32)
#
# # 放一些数据
# enq_many = Q.enqueue_many([[0.1, 0.2, 0.3], ])
#
# # 2. 定义一些处理数据的逻辑
# out_q = Q.dequeue()
#
# data = out_q + 1
#
# en_q = Q.enqueue(data)
#
# with tf.compat.v1.Session() as sess:
#     # 初始化队列
#     sess.run(enq_many)
#     # 处理数据
#     for i in range(100):
#         sess.run(en_q)
#
#     for i in range(Q.size().eval()):
#         print(sess.run(Q.dequeue()))


'''
模拟异步子线程，存入样本。 主线程， 读取样本
'''


# 1. 定义一个队列
# Q = tf.compat.v1.FIFOQueue(1000, tf.float32)
#
# # 2.定义要做的事情
# var = tf.compat.v1.Variable(0.0)
# # 实现变量op自增
# data = tf.compat.v1.assign_add(var, tf.constant(1.0))
#
# en_q = Q.enqueue(data)
#
# # 3. 定义队列管理器op，指定多少个子线程，子线程该做什么事情。   执行en_q操作，2个线程
# qr = tf.compat.v1.train.QueueRunner(Q, enqueue_ops=[en_q] * 2)
#
# # 初始化变量的op
# init_op = tf.compat.v1.global_variables_initializer()
#
# with tf.compat.v1.Session() as sess:
#     # 初始化变量
#     sess.run(init_op)
#
#     # 开启线程管理器
#     coord = tf.train.Coordinator()
#
#     # 真正开启子线程
#     threads = qr.create_threads(sess, coord= coord, start=True)
#
#     # 主线程，不断读取数据训练
#     for i in range(300):
#         print(sess.run(Q.dequeue()))
#
#     # 回收线程
#     coord.request_stop()
#     coord.join( threads )


# 批处理大小，跟队列，数据的数量没有影响，之决定这批次取多少数据
def csvread(filelist):
    """
    读取csv文件
    :param filelist: 文件路径+名字的列表
    :return:  读取的内容
    """
    # 1. 构造文件的队列
    file_queue = tf.compat.v1.train.string_input_producer(filelist)

    # 2. 构造csv阅读器读取队列数据（ 按一行 ）
    reader = tf.compat.v1.TextLineReader()

    key, value = reader.read(file_queue)
    # print(value) # Tensor("ReaderReadV2:1", shape=(), dtype=string)

    # 3. 对每行的内容进行解码
    # record_defaults： 指定每一个样本的每一列的类型，指定默认值[['None'],[4.0]]
    records = [['None'], ['None']]
    example, label = tf.compat.v1.decode_csv(value, record_defaults=records)

    # 4. 想要读取多个数据，就需要批处理
    example_batch, label_batch = tf.compat.v1.train.batch([example, label], batch_size=9, num_threads=1, capacity=9)
    return example_batch, label_batch


def picread(fileList):
    """
    读取图片
    :param fileList:
    :return:
    """
    # 1. 构造文件队列
    file_queue = tf.compat.v1.train.string_input_producer(fileList)

    # 2.构造阅读器去读取图片的内容（ 默认读取一张图片）
    reader = tf.compat.v1.WholeFileReader()

    key, value = reader.read(file_queue)
    print(value)  # Tensor("ReaderReadV2:1", shape=(), dtype=string)
    # 3. 对读取的图片数据进行解码
    dog_image = tf.compat.v1.image.decode_jpeg(value)
    print(dog_image)  # Tensor("DecodeJpeg:0", shape=(None, None, None), dtype=uint8)

    # 4. 处理图片的大小（ 统一大小 ）
    image_resize = tf.compat.v1.image.resize_images(dog_image, [200, 200])
    print(image_resize)

    # 注意，一定要把样本的形状固定
    image_resize.set_shape([200, 200, 3])
    print(image_resize)
    # 6. 进行批处理
    image_batch = tf.compat.v1.train.batch([image_resize], batch_size=20, num_threads=1, capacity=20)
    print(image_batch)
    return image_batch


# 定义cifar的数据的命令行参数
FLAGS = tf.compat.v1.flags.FLAGS
tf.compat.v1.flags.DEFINE_string('cifar_dir', '../data/cifar-10-batches-bin/', '文件的目录')
tf.compat.v1.flags.DEFINE_string('cifar_tfrecords', '../tmp/cifar.tfrecords', '存进tfrecords的目录')


class CifarRead(object):
    """
    完成读取二进制文件，写进tfrecords，读取tfrecords
    """

    def __init__(self, fileList):
        # 文件列表
        self.file_list = fileList

        # 定义读取的图片的一些属性
        self.height = 32
        self.width = 32
        self.channel = 3

        # 二进制文件每张图片的子节
        self.label_bytes = 1
        self.image_bytes = self.height * self.width * self.channel
        self.bytes = self.label_bytes + self.image_bytes

    def read_and_decode(self):
        # 1. 构造文件队列
        file_queue = tf.compat.v1.train.string_input_producer(self.file_list)

        # 2. 构造二进制文件读取器，读取内容，读取的大小为每个样本的字节数
        reader = tf.compat.v1.FixedLengthRecordReader(self.bytes)

        key, value = reader.read(file_queue)

        # 3. 解码内容,二进制内容解码
        label_imgae = tf.compat.v1.decode_raw(value, tf.uint8)
        print(label_imgae)  # Tensor("DecodeRaw:0", shape=(None,), dtype=uint89)

        # 4. 分割出图片和标签数据，切除特征值和目标值
        label = tf.compat.v1.cast(tf.slice(label_imgae, [0], [self.label_bytes]), tf.int32)
        image = tf.slice(label_imgae, [self.label_bytes], [self.image_bytes])

        # 5. 可以对图片的特征数据进行形状的改变
        image_reshape = tf.reshape(image, [self.height, self.width, self.channel])
        print(image_reshape)  # Tensor("Reshape:0", shape=(32, 32, 3), dtype=uint8)

        # 6. 批处理数据
        image_batch, label_batch = tf.compat.v1.train.batch([image_reshape, label], batch_size=10, num_threads=1,
                                                            capacity=10)
        # Tensor("batch:0", shape=(10, 32, 32, 3), dtype=uint8)
        # Tensor("batch:1", shape=(10, 1), dtype=int32)
        print(image_batch, label_batch)

        return image_batch, label_batch

    def write_to_tfrecords(self, image_batch, label_batch):
        """
        将图片的特征值与目标值存入tfrecords
        :param image_batch:  10张图片的特征值
        :param label_batch:  10张图片的目标值
        :return: None
        """
        # 1. 建立TFRecords存储器
        writer = tf.compat.v1.python_io.TFRecordWriter(FLAGS.cifar_tfrecords)

        # 2. 循环将所有样本写入文件，每张图片样本都要构造example协议
        for i in range(10):
            # 取出第i个样本写入文件，每张图片样本都要构造example协议
            image = image_batch[i].eval().tostring()
            label = int(label_batch[i].eval()[0])

            # 构造一个样本的example
            example = tf.compat.v1.train.Example(features=tf.compat.v1.train.Features(feature={
                'image': tf.compat.v1.train.Feature(bytes_list=tf.compat.v1.train.BytesList(value=[image])),
                'label': tf.compat.v1.train.Feature(int64_list=tf.compat.v1.train.Int64List(value=[label])),
            }))

            # 写入单独的样本
            writer.write(example.SerializeToString())

        writer.close()
        return None

    def read_from_tfrecords(self):
        # 1. 构造文件队列
        file_queue = tf.compat.v1.train.string_input_producer([FLAGS.cifar_tfrecords])

        # 2. 构造文件阅读器，读取内容`Example`
        reader = tf.compat.v1.TFRecordReader()

        key, value = reader.read(file_queue)

        # 3. 解析example
        features = tf.compat.v1.parse_single_example(value, features={
            'image': tf.compat.v1.FixedLenFeature([], tf.string),
            'label': tf.compat.v1.FixedLenFeature([], tf.int64),  # 只支持转int64
        })

        # 4.解码内容,如果读取的内容格式是string则需要解码，如果是int634或int32则不需要解码
        image = tf.compat.v1.decode_raw(features['image'], tf.uint8)
        # 指定image的大小，以便于批处理
        image_reshape = tf.reshape(image, [self.height, self.width, self.channel])

        label = tf.compat.v1.cast(features['label'], tf.int32)

        # 5.批处理
        image_batch, label_batch = tf.compat.v1.train.batch([image_reshape, label], batch_size=10, num_threads=1,
                                                            capacity=10)

        return image_batch, label_batch


if __name__ == '__main__':
    # 1. 找到文件、放入列表
    fileNames = os.listdir(FLAGS.cifar_dir)
    # print( fileNames ) # ['A.csv', 'B.csv', 'C.csv']
    fileList = [os.path.join(FLAGS.cifar_dir, fileName) for fileName in fileNames if fileName[-3:] == 'bin']
    cf = CifarRead(fileList)
    # image_batch, label_batch = cf.read_and_decode()
    image_batch, label_batch = cf.read_from_tfrecords()

    # 开启会话运行结果
    with tf.compat.v1.Session() as sess:
        # 定义一个线程协调器
        coord = tf.compat.v1.train.Coordinator()

        # 开启读文件的线程
        threads = tf.compat.v1.train.start_queue_runners(sess, coord=coord)

        # 存进tfrecords文件
        # print('开始存储')
        # cf.write_to_tfrecords( image_batch, label_batch )
        # print('结束存储')

        # 打印读取的内容
        print(sess.run([image_batch, label_batch]))

        # 回收子线程
        coord.request_stop()
        coord.join(threads)
