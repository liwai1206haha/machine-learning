# 导入数据
import joblib
from sklearn.datasets import load_boston
# 引入线性回归中的线性回归算法和梯度下降算法
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# 均方误差
from sklearn.metrics import mean_squared_error ,classification_report
import pandas as pd
import numpy as np


def myliner():
    """
    线性回归直接预测房子价格
    :return: None
    """
    # 获取数据
    lb = load_boston()

    # 分割数据集
    x_train, x_test, y_train, y_test = train_test_split(lb.data, lb.target, test_size=0.25)
    # print(y_train, y_test)

    # 进行标准化处理
    # 特征值和目标值都必须进行标准化处理，并且不能使用同一个标准化实例对象
    std_x = StandardScaler()

    x_train = std_x.fit_transform(x_train)
    x_test = std_x.transform(x_test)

    # 目标值
    std_y = StandardScaler()

    y_train = std_y.fit_transform(y_train.reshape(-1, 1))
    y_test = std_y.transform(y_test.reshape(-1, 1))

    # estimator预测
    # 正规方程求解方式预测结果
    lr = LinearRegression()
    # 训练模型
    lr.fit(x_train, y_train)
    # 回归系数
    print('回归系数', lr.coef_)

    '''
        保存训练好的模型
    '''
    joblib.dump(lr, '../tmp/LinearRegression.pkl')
    # 使用保存的模型进行预测
    lr = joblib.load('../tmp/LinearRegression.pkl')

    # 预测测试集的房子价格,将标准化的数据转换成标准化之前的数据
    y_lr_predict = std_y.inverse_transform(lr.predict(x_test).reshape(-1, 1))
    print('正规方程测试集里面每个样本的预测价格：', y_lr_predict)
    print('正规方程的均方误差：', mean_squared_error(std_y.inverse_transform(y_test.reshape(-1, 1)), y_lr_predict))

    print('*' * 100)

    # 梯度下降求解方式预测结果
    sgd = SGDRegressor()
    # 训练模型
    sgd.fit(x_train, y_train)
    # 回归系数
    print('回归系数', sgd.coef_)
    # 预测测试集的房子价格,将标准化的数据转换成标准化之前的数据
    y_sgd_predict = std_y.inverse_transform(sgd.predict(x_test).reshape(-1, 1))
    print('梯度下降测试集里面每个样本的预测价格：', y_sgd_predict)
    print('梯度下降的均方误差：', mean_squared_error(std_y.inverse_transform(y_test.reshape(-1, 1)), y_sgd_predict))

    # 岭回归求解方式预测结果
    rd = Ridge()
    # 训练模型
    rd.fit(x_train, y_train)
    # 回归系数
    print('回归系数', rd.coef_)
    # 预测测试集的房子价格,将标准化的数据转换成标准化之前的数据
    y_rd_predict = std_y.inverse_transform(rd.predict(x_test).reshape(-1, 1))
    print('岭回归测试集里面每个样本的预测价格：', y_rd_predict)
    print('岭回归的均方误差：', mean_squared_error(std_y.inverse_transform(y_test.reshape(-1, 1)), y_rd_predict))

    return None


def logistic():
    """
    逻辑回归判断是否得癌症
    :return: None
    """
    # 分割数据集
    column = ['Sample code number',
              'Clump Thickness',
              'Uniformity of Cell Size',
              'Uniformity of Cell Shape',
              'Marginal Adhesion',
              'Single Epithelial Cell Size',
              'Bare Nuclei',
              'Bland Chromatin',
              'Normal Nucleoli',
              'Mitoses',
              'Class']

    # 读取数据
    file_path = '../data/cancer/breast-cancer-wisconsin.data'
    data = pd.read_csv(file_path, names=column)
    print( data )

    # 替换空参数
    data = data.replace( to_replace='?', value=np.nan )
    data = data.dropna()

    x_train, x_test, y_train, y_test = train_test_split(data[column[1:10]], data[column[10]], test_size=0.25)

    # 标准化
    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)

    # 使用逻辑回归模型进行预测
    lg = LogisticRegression(C=1.0)
    # 训练模型
    lg.fit(x_train, y_train)
    y_predict = lg.predict( x_test )
    # 打印回归参数
    print('回归参数', lg.coef_)
    print('准确率：', lg.score(x_test, y_test))
    print( '召回率', classification_report(y_test, y_predict, labels=[2,4] ,target_names=['良性','恶行']))


if __name__ == '__main__':
    # myliner()
    logistic()