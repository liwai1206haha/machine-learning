import pandas as pd
# 数据集
from sklearn.datasets import load_iris, fetch_20newsgroups, load_boston
# 分割测试集与训练集
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
# 特征标准化
from sklearn.preprocessing import StandardScaler
# k近邻算法
from sklearn.neighbors import KNeighborsClassifier
# 特征抽取
from sklearn.feature_extraction.text import TfidfVectorizer
# 朴素贝叶斯
from sklearn.naive_bayes import MultinomialNB
# 精确率和召回率
from sklearn.metrics import classification_report
# 决策树
from sklearn.tree import DecisionTreeClassifier, export_graphviz

'''
分类数据集
'''
# li = load_iris()  # 获得数据

# print("获取特征值")
# print(li.data)
# print( "目标值")
# print(li.target)
# print( li.DESCR)

# 注意返回值，训练集的特征值，测试机的特征值，训练集的目标值，测试集的目标值
# x_train, x_test, y_train, y_test = train_test_split(li.data, li.target, test_size=0.25)  # 25%的数据作为测试集

# print("训练集特征值和目标值：", x_train, y_train)
# print("测试集特征值和目标值：", x_test, y_test)

# news = fetch_20newsgroups( subset='all')
# print( news.data)
# print( news.target)


"""
回归数据集
"""


# lb = load_boston()
# print("特征值", lb.data)
# print("目标值", lb.target)


def knncls():
    """
    K-近邻预测用户签到位置
    :return: None
    """
    # 读取数据
    data = pd.read_csv('../data/FBlocation/train.csv')
    # 处理数据
    # 1. 缩小数据规模，查询数据筛选
    data = data.query('x>1.0 & x <1.25 & y>2.5 & y<2.75')
    # print( data )

    # 处理时间的数据
    time_value = pd.to_datetime(data['time'], unit='s')
    print(time_value)

    # 把日期格式转换成字典格式
    time_value = pd.DatetimeIndex(time_value)
    print(time_value)

    # 构造一些特征
    data['day'] = time_value.day
    data['hour'] = time_value.hour
    data['weekday'] = time_value.weekday

    # 把时间戳和row_id 特征删除
    data = data.drop(['time'], axis=1)
    # print( data )

    # 把签到数量少于n个的目标位置删除
    # 1. 统计数量
    place_count = data.groupby('place_id').count()
    # 2. 筛选数据，并重置索引,place_count的每一列都是count值
    tf = place_count[place_count.row_id > 3].reset_index()
    # 3. 筛选出data中place_id在tf.place_id中的列
    data = data[data['place_id'].isin(tf.place_id)]

    # 取出数据中的特征值和目标值
    y = data['place_id']
    x = data.drop(['place_id', 'row_id'], axis=1)

    # 进行数据的分割训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # 特征工程（ 标准化）
    std = StandardScaler()

    # 对测试集和训练集的特征值进行标准化
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)

    # 进行算法流程
    # knn = KNeighborsClassifier(n_neighbors=5)  # 5个邻近值

    # 处理并训练数据，得出模型
    # knn.fit(x_train, y_train)
    #
    # # 得出预测结果
    # y_predict = knn.predict(x_test)
    # print('预测的目标签到位置为：', y_predict)
    #
    # # 得出准确率,先对x_test进行预测。得出预测值，再与y_test比较
    # print('预测的准确率:', knn.score(x_test, y_test))

    # 构造一些参数的值进行搜索
    param = {'n_neighbors': [3, 5, 10]}

    # 进行网格搜索
    knn = KNeighborsClassifier()
    gc = GridSearchCV(knn, param_grid=param, cv=2)  # 使用二折交叉验证

    gc.fit(x_train, y_train)

    # 预测准确率
    print('测试集上的准确率：', gc.score(x_test, y_test))
    print('在交叉验证中最好的结果：', gc.best_score_)
    print('选择最好的模型是：', gc.best_estimator_)
    print('每个超参数每次交叉验证的结果：', gc.cv_results_)

    return None


def naviebayes():
    """
    朴贝叶斯进行文本分类
    :return:  None
    """
    news = fetch_20newsgroups(subset='all')

    # 进行数据分割
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25)

    # 对数据进行特征抽取
    tf = TfidfVectorizer()

    # 以训练集当中的词的列表进行每篇文章重要性统计
    x_train = tf.fit_transform(x_train)
    print(tf.get_feature_names_out())

    x_test = tf.transform(x_test)

    # 进行朴素贝叶斯算法的预测,alpha: 拉普拉斯平滑系数
    mlt = MultinomialNB(alpha=1.0)
    print(x_train.toarray())

    mlt.fit(x_train, y_train)

    y_predict = mlt.predict(x_test)
    print('预测的文章类别为：', y_predict)

    # 准确率
    print('准确率为：', mlt.score(x_test, y_test))

    # 准确率和召回率               真实目标值  预测目标值    分类类别
    print(classification_report(y_test, y_predict, target_names=news.target_names))

    return None


def decision():
    """
    决策树对泰坦尼克号进行预测生死
    :return:  None
    """
    # 获取数据
    file_path = '../data/titanic/train.csv'
    titan = pd.read_csv(file_path)

    # 处理数据，找出特征值与目标值
    x = titan[['Pclass', 'Age', 'Sex']]
    y = titan['Survived']
    # print(x)
    # 缺失值处理
    x['Age'].fillna(x['Age'].mean(), inplace=True)

    # 分割数据
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # 进行数据处理（特征工程） 特征 》 类别 》 one-hot编码
    dict = DictVectorizer(sparse=False)
    # x_train.to_dict( orient='records')): 将x_train转换为字典,然后才能进行字典的特征抽取
    x_train = dict.fit_transform(x_train.to_dict(orient='records'))
    # print(dict.get_feature_names_out())

    x_test = dict.transform(x_test.to_dict(orient='records'))
    # print(x_train)

    # 使用决策树进行预测
    # dec = DecisionTreeClassifier( max_depth=8 ) # 决策树的最大深度
    # # 训练模型
    # dec.fit(x_train, y_train)
    #
    # # 预测准确率
    # print('准确率:', dec.score(x_test, y_test))

    # 导出决策树的结构
    # export_graphviz(dec,
    #                 out_file='./tree.dot',
    #                 feature_names=['年龄', 'plass=1st', 'pclass=2nd', 'pclass=3rd', '女性', '男性'])

    # 使用决策森林进行预测
    rf = RandomForestClassifier()
    param = {'n_estimators': [120, 200, 300, 500, 800, 1200], 'max_depth': [5, 8, 15, 25, 30]}

    # 网格搜索与交叉验证, 2折
    gc = GridSearchCV(rf, param_grid=param, cv=2)
    gc.fit(x_train, y_train)

    print('准确率', gc.score(x_test, y_test))
    print('查看选择的参数模型:', gc.best_params_)


if __name__ == '__main__':
    # knncls()
    # naviebayes()
    decision()
