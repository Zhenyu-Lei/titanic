# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from myUtils import *
import os
import pandas as pd
from torch import nn


class my_model(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.layer1 = nn.Linear(in_features, in_features)
        self.sigmoid1 = nn.Sigmoid()
        self.layer2 = nn.Linear(in_features, 1)
        self.sigmoid2 = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.sigmoid1(x)
        x = self.layer2(x)
        x = self.sigmoid2(x)
        return x


def get_net(in_features):
    return my_model(in_features)


def main():
    # 读取数据
    train_data, test_data = readData('./data/train.csv', './data/test.csv')

    if not os.path.exists("./data/oneHot_all_feature.csv"):
        print("不存在oneHot_all_feature.csv文件，请先使用dataAnalysis.py生成")
        return

    all_features = pd.read_csv("./data/oneHot_all_feature.csv")

    print("开始训练")

    n_train = train_data.shape[0]  # 样本个数

    # 将pandas数据类型转为pytorch 数据类型tensor
    train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
    test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
    train_labels = torch.tensor(train_data['Survived'].values, dtype=torch.float32).reshape(-1, 1)

    # 进行训练
    # 模型选择
    k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 16
    loss = nn.BCELoss()
    net = get_net(test_features.shape[1])
    train_l, valid_l = k_fold(net, loss, k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
    print(f'{k}-折验证：平均训练rmse：{float(train_l):f},'f'平均验证rmse：{float(valid_l):f}')
    # 训练集验证集划分
    # X_train, y_train, X_valid, y_valid = get_k_fold_data(5, 1, train_features, train_labels)
    # train_l, valid_l = train(net, loss, X_train, y_train, X_valid, y_valid, num_epochs, lr, weight_decay, batch_size)
    # print(f'平均训练log rmse：{float(train_l[-1]):f},平均验证log rmse：{float(valid_l[-1]):f}')

    # detach(将GPU上数据转到CPU上),numpy(将tensor数据转为numpy数据)
    preds = net(test_features).detach()
    preds = torch.where(preds < 0.5, torch.tensor(0), torch.tensor(1))
    preds = preds.numpy()
    # reshape(1,-1):将数据变为一行的数据，大小为1×n(-1意思是根据大小自行计算另一维度),然后是[[]]型数据，取第0维出来
    test_data['Survived'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['PassengerId'], test_data['Survived']], axis=1)
    submission.to_csv('submission.csv', index=False)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
