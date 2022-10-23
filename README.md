# machine-learning
机器学习

可用数据集: kaggle、scikit-learn、UCI
 
## 特征提取
特征抽取就是对文本等数据进行特征值化
1. 字典特征数据抽取: 把字典中一些类别的数据，分别进行转换成特征
2. 文本特征抽取：对文本数据进行特征值化
3. 中文特征抽取，
   第一种方式：先使用jieba进行分词再特征化
   第二种方式： tf: 词的频率
              idf：逆文档频率  log（总文档数量/该词出现的文档数量）
   

## 归一化
多个特征同等重要的时候，需要归一化，使得某一个特征不会对结果的影响太大。
![img.png](img.png)

## 标准化
如果出现异常点，由于具有一定的数据量，少量的异常点对于平均值的影响并不大，从而方差改变较小

## 特征预处理：对数据进行处理。
通过特定的统计方法，将数据转换成算法要求的数据



