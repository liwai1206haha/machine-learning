# 导入数据
import joblib
from sklearn.datasets import load_boston
# 引入线性回归中的线性回归算法和梯度下降算法
from sklearn.linear_model import LinearRegression, SGDRegressor,Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error  # 均方误差


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
    joblib.dump( lr, '../tmp/LinearRegression.pkl')
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


if __name__ == '__main__':
    myliner()
