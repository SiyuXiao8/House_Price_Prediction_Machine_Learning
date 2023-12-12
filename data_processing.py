"""
Authors: Siyu Xiao, Shi Zeng, Huijuan Wang
Description: Kaggle House prediction using machine learning.
Before running the model file, this file should be run first to process the train and test data
"""

import warnings

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

from scipy.stats import skew

# Visualization
import seaborn as sns
from matplotlib import pyplot as plt

color = sns.color_palette()
sns.set_style('darkgrid')

# ignore warnings


def ignore_warn(*args, **kwargs):
    pass


warnings.warn = ignore_warn

train_set = pd.read_csv("train.csv")
test_set = pd.read_csv("test.csv")

# remove Id from the dataset, since it is not related
train_set.drop('Id', axis=1, inplace=True)
test_set.drop('Id', axis=1, inplace=True)

# break the data into two category - numerical, categorical
num_features = []
cate_features = []
for col in test_set.columns:
    if test_set[col].dtype == 'object':
        cate_features.append(col)
    else:
        num_features.append(col)
print('number of numeric features:', len(num_features))
print('number of categorical features:', len(cate_features))

# Feature engineering part

# Oberserve that ’TotalBsmtSF’、'GrLiveArea's graph，we can see that there's outlier：
sns.scatterplot(x='TotalBsmtSF', y='SalePrice', data=train_set)
# plt.show()

# remove the outlier
train = train_set.drop(train_set[(train_set['TotalBsmtSF'] > 6000) & (
    train_set['SalePrice'] < 200000)].index)
sns.scatterplot(x='TotalBsmtSF', y='SalePrice', data=train)
# plt.show()

# same procedure applys to 'GrLivArea'：
sns.scatterplot(x='GrLivArea', y='SalePrice', data=train)

# remove the outlier
train = train.drop(train[(train['GrLivArea'] > 4000) &
                   (train['SalePrice'] < 200000)].index)
sns.scatterplot(x='GrLivArea', y='SalePrice', data=train)
# plt.show()

# Null value processing
# Check the amount of missing values in train set
print('The shape of training data:', train.shape)
train_missing = train.isnull().sum()
train_missing = train_missing.drop(
    train_missing[train_missing == 0].index).sort_values(ascending=False)
# print(train_missing)


# Check the missing value in the test set
print('The shape of testing data:', test_set.shape)
test_missing = test_set.isnull().sum()
test_missing = test_missing.drop(
    test_missing[test_missing == 0].index).sort_values(ascending=False)
# print(train_missing)

# Fill it with none
none_lists = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
              'GarageCond', 'BsmtFinType1',
              'BsmtFinType2', 'BsmtCond', 'BsmtExposure', 'BsmtQual', 'MasVnrType']
for col in none_lists:
    train[col] = train[col].fillna('None')
    test_set[col] = test_set[col].fillna('None')

# Fill with the most freq element
most_lists = ['MSZoning', 'Exterior1st', 'Exterior2nd',
              'SaleType', 'KitchenQual', 'Electrical']
for col in most_lists:
    train[col] = train[col].fillna(train[col].mode()[0])
    test_set[col] = test_set[col].fillna(
        train[col].mode()[0])

train['Functional'] = train['Functional'].fillna('Typ')
test_set['Functional'] = test_set['Functional'].fillna('Typ')

# Delete useless feature
train.drop('Utilities', axis=1, inplace=True)
test_set.drop('Utilities', axis=1, inplace=True)

# Numberical null value processing
# Fill with zeros
zero_lists = ['GarageYrBlt', 'MasVnrArea', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
              'GarageCars', 'GarageArea',
              'TotalBsmtSF']
for col in zero_lists:
    train[col] = train[col].fillna(0)
    test_set[col] = test_set[col].fillna(0)

# Fill with average val
train['LotFrontage'] = train.groupby(
    'Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

# train['LotFrontage'] = train.groupby(
#     'Neighborhood')['LotFrontage'].apply(lambda x: x.fillna(x.median()))

for ind in test_set['LotFrontage'][test_set['LotFrontage'].isnull().values == True].index:
    x = test_set['Neighborhood'].iloc[ind]
    test_set['LotFrontage'].iloc[ind] = train.groupby(
        'Neighborhood')['LotFrontage'].median()[x]

# Check whether there is more null val
print(train.isnull().sum().any())
print(test_set.isnull().sum().any())

# LabelEncoder
# Remove 'Utilities'
cate_features.remove('Utilities')
print('The number of categorical features:', len(cate_features))

for col in cate_features:
    train[col] = train[col].astype(str)
    test_set[col] = test_set[col].astype(str)
le_features = ['Street', 'Alley', 'LotShape', 'LandContour', 'LandSlope', 'HouseStyle', 'RoofMatl', 'Exterior1st',
               'Exterior2nd', 'ExterQual',
               'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
               'HeatingQC', 'CentralAir',
               'KitchenQual', 'Functional', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive',
               'PoolQC', 'Fence']
for col in le_features:
    encoder = LabelEncoder()
    value_train = set(train[col].unique())
    value_test = set(test_set[col].unique())
    value_list = list(value_train | value_test)
    encoder.fit(value_list)
    train[col] = encoder.transform(train[col])
    test_set[col] = encoder.transform(test_set[col])

# process skew features
skewness = train[num_features].apply(
    lambda x: skew(x)).sort_values(ascending=False)
skewness = skewness[skewness > 0.5]
skew_features = skewness.index
print(skewness)

# engineer new features

# 1 - remodeled, 0 - did not remodel
train['IsRemod'] = 1
train['IsRemod'].loc[train['YearBuilt'] ==
                     train['YearRemodAdd']] = 0
# difference b/w year remodeled and year built
train['BltRemodDiff'] = train['YearRemodAdd'] - \
    train['YearBuilt']
# Basement's unfinished ratio
train['BsmtUnfRatio'] = 0
train['BsmtUnfRatio'].loc[train['TotalBsmtSF'] !=
                          0] = train['BsmtUnfSF'] / train['TotalBsmtSF']
# total square footage
train['TotalSF'] = train['TotalBsmtSF'] + \
    train['1stFlrSF'] + train['2ndFlrSF']
# Same procedure for testing dataset
test_set['IsRemod'] = 1
test_set['IsRemod'].loc[test_set['YearBuilt'] ==
                        test_set['YearRemodAdd']] = 0
test_set['BltRemodDiff'] = test_set['YearRemodAdd'] - \
    test_set['YearBuilt']
test_set['BsmtUnfRatio'] = 0
test_set['BsmtUnfRatio'].loc[test_set['TotalBsmtSF'] != 0] = test_set['BsmtUnfSF'] / test_set[
    'TotalBsmtSF']
test_set['TotalSF'] = test_set['TotalBsmtSF'] + \
    test_set['1stFlrSF'] + test_set['2ndFlrSF']

# process other features
dummy_features = list(set(cate_features).difference(set(le_features)))
print(dummy_features)

# combine both training and testing set：
all_data = pd.concat((train.drop('SalePrice', axis=1),
                     test_set)).reset_index(drop=True)
all_data = pd.get_dummies(all_data, drop_first=True)

# save the test and train set after processing
trainset = all_data[:1458]
y = train['SalePrice']
trainset['SalePrice'] = y.values
testset = all_data[1458:]
print('The shape of training data:', trainset.shape)
print('The shape of testing data:', testset.shape)

# save to our local machine
trainset.to_csv('train_data_after_process.csv', index=False)
testset.to_csv('test_data_after_process.csv', index=False)
