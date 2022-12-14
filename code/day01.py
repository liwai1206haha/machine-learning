# 引入字典特征抽取
from sklearn.feature_extraction import DictVectorizer
# 文本特征提取
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# 用于中文分词
import jieba
# 归一化处理和标准化处理
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
# 特征选择
from sklearn.feature_selection import VarianceThreshold
# 主成分分析
from sklearn.decomposition import PCA
import numpy as np


def dictvec():
    '''
    字典特征抽取
    :return:  None
    '''
    # 实例化,sparse=True表示使用压缩矩阵，等于False表示使用原始的二维矩阵
    dict = DictVectorizer(sparse=False)

    # 调用fit_transform,将数据转换为特征值
    data = dict.fit_transform([
        {'City': '北京', 'temperature': 100},
        {'City': '上海', 'temperature': 60},
        {'City': '深圳', 'temperature': 30}
    ])

    print(dict.get_feature_names_out())  # ['City=上海' 'City=北京' 'City=深圳' 'temperature']
    print(data)
    '''[[  0.   1.   0. 100.]
         [  1.   0.   0.  60.]
         [  0.   0.   1.  30.]]'''

    # 将特征值转换成原数据
    print(dict.inverse_transform(data))
    return None


def countvec():
    '''
    对文本进行特征值化
    :return: None
    '''
    cv = CountVectorizer()
    # data = cv.fit_transform(['life is short,i like python', 'life is too long,i dislike python'])

    # 对于中文，默认不支持特征抽取.需要先进行分词，再进行抽取
    data = cv.fit_transform(['人生苦短，我喜欢 python', '人生漫长，不用 python'])
    # 统计所有文章中所有的词，作为一个元素不重复的列表，这叫做“词的列表”。
    # 注意，对单个英文字母不统计，因为不作为分类依据
    print(cv.get_feature_names_out())  # ['dislike' 'is' 'life' 'like' 'long' 'python' 'short' 'too']
    # 统计每篇文章，在词的列表里面每个词出现的次数
    print(data.toarray())  # [[0 1 1 1 0 1 1 0]
    #  [1 1 1 0 1 1 0 1]]
    return None


def cutword():
    # 利用jieba进行分词
    con1 = jieba.cut('今天很残酷，明天更残酷，后天很美好，但绝大部分是死在明天晚上，所以每个人不要放弃今天。')
    con2 = jieba.cut('我们看到的从很远星系来的光是在几百万年前发出的，这样当我们看到宇宙时，我们是在看他的过去。')
    con3 = jieba.cut('如果只用一种方式了解某样事物，你就不会真正了解他。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。')

    # 将分词结果转换为list
    content1 = list(con1)
    content2 = list(con2)
    content3 = list(con3)

    # 将列表中的每个值都用空格连接
    c1 = ' '.join(content1)
    c2 = ' '.join(content2)
    c3 = ' '.join(content3)

    return c1, c2, c3


def hanzivec():
    '''
    中文特征值化
    :return:None
    '''
    c1, c2, c3 = cutword()
    print(c1, c2, c3, sep='\n')
    cv = CountVectorizer()
    # 对于中文，默认不支持特征抽取.需要先进行分词，再进行抽取
    data = cv.fit_transform([c1, c2, c3])
    print(cv.get_feature_names_out())
    print(data.toarray())
    return None


def tfidfvec():
    '''
    中文特征值化2
    :return:None
    '''
    c1, c2, c3 = cutword()
    # print( c1,c2,c3, sep='\n')
    tv = TfidfVectorizer()
    # 对于中文，默认不支持特征抽取.需要先进行分词，再进行抽取
    data = tv.fit_transform([c1, c2, c3])
    print(tv.get_feature_names_out())
    print(data.toarray())  # 重要性
    return None


def mm():
    '''
    归一化处理
    :return: None
    '''
    m = MinMaxScaler(feature_range=(2, 3))  # 缩放到每个特征的值都在2--3之间
    data = m.fit_transform([[90, 2, 10, 40], [60, 4, 15, 45], [75, 3, 13, 46]])
    print(data)
    return None


def stand():
    '''
    标准化缩放
    :return: None
    '''
    std = StandardScaler()
    data = std.fit_transform([[1., -1., 3.], [2., 4., 2.], [4., 6., -1.]])
    print(data)
    return None


def im():
    '''
    缺失值处理
    :return:  None
    '''
    im = SimpleImputer(missing_values=np.nan, strategy='mean')  # 使用平均值填充nan值
    data = im.fit_transform([[1, 2], [np.nan, 3], [7, 6]])
    print(data)
    return None


def var():
    '''
    特征选择 -- 删除低方差的特征
    :return:
    '''
    var = VarianceThreshold(threshold=0.0)
    data = var.fit_transform([
        [0, 2, 0, 3],
        [0, 1, 4, 3],
        [0, 1, 1, 3]
    ])
    '''
    [[2 0]
     [1 4]
     [1 1]]
    '''
    print(data)
    return None


def pca():
    '''
    主成分分析进行特征降维
    :return:  None
    '''
    pca = PCA(n_components=0.9)
    data = pca.fit_transform([
        [2, 8, 4, 5],
        [6, 3, 0, 8],
        [5, 4, 9, 1],
    ])
    print(data)
    return None


if __name__ == '__main__':
    # dictvec()
    # countvec()
    # hanzivec()
    # tfidfvec()
    # mm()
    # stand()
    # im()
    # var()
    pca()
