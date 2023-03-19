from torch import nn
from myUtils import readData
from myUtils import correlation
import pandas as pd
from myUtils import pandas_plot
from myUtils import pandas_plot_judgeNull
from sklearn.ensemble import RandomForestRegressor


def set_missing_age(df):
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    # age非空的的词条
    known_age = age_df[age_df['Age'].notnull()].values
    unknown_age = age_df[age_df['Age'].isnull()].values

    y = known_age[:, 0]
    x = known_age[:, 1:]

    # n_estimators集成随机树的个数，n_job并行度
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(x, y)

    predictedAges = rfr.predict(unknown_age[:, 1:])
    df.loc[df['Age'].isnull(), 'Age'] = predictedAges


def main():
    # 读取数据
    train_data, test_data = readData('./data/train.csv', './data/test.csv')

    object_features = train_data.dtypes[train_data.dtypes == 'object'].index

    # 对数值对象进行相关性分析
    # correlation(train_data, "./data")
    # 由于数据量不大，将所有数字类型的数据都进行保留
    numeric_main_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
    # 对object对象(离散值)进行分析
    # pd.set_option('display.max_columns', 1000)
    # pd.set_option('display.width', 1000)
    # pd.set_option('display.max_colwidth', 1000)
    # pandas_plot_judgeNull(train_data['Survived'], train_data['Cabin'])

    object_main_features = ['Sex', 'Embarked', 'Cabin']
    all_features_list = numeric_main_features + object_main_features
    # 合并特征
    all_features = pd.concat((train_data[all_features_list], test_data[all_features_list]))

    train_object = train_data[object_features]
    print(train_object.describe())
    print(all_features.describe())

    # 先将Fare的缺失值填充
    all_features['Fare'] = all_features['Fare'].fillna(all_features['Fare'].mean())

    # 处理Age属性
    set_missing_age(all_features)

    # 处理cabin属性：分为null和not null两种
    all_features.loc[all_features['Cabin'].notnull(), 'Cabin'] = 'Yes'
    all_features.loc[all_features['Cabin'].isnull(), 'Cabin'] = 'No'

    # 将缺失值变为均值
    # 将特征缩放到0均值和单位方差来归一化
    # 将非object数据变为均值为0，方差为1
    all_features[numeric_main_features] = all_features[numeric_main_features].apply(
        lambda x: (x - x.mean()) / (x.std()))
    # 将数值数据中not number的数据用0填充(其实是均值，修改了分布均值为0)
    all_features[numeric_main_features] = all_features[numeric_main_features].fillna(0)

    # 处理离散值，用独热编码替换它们
    all_features = pd.get_dummies(all_features, dummy_na=True)
    # print(all_features[:4].values)
    all_features.to_csv("./data/oneHot_all_feature.csv", index=False, encoding="utf-8")

    print("数据生成完成")


if __name__ == '__main__':
    main()
