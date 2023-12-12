"""
Authors: Siyu Xiao, Shi Zeng, Huijuan Wang
Description: Kaggle House prediction using machine learning.
This file contains implementation of the algorithms that train the predictive model
"""
from scipy.stats import skew, kurtosis, norm
import numpy as np
import pandas as pd


import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

# model
from sklearn.linear_model import LassoCV, ElasticNetCV,  RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from mlxtend.regressor import StackingCVRegressor
# from sklearn.svm import SVR
import lightgbm as lgb
import xgboost as xgb

# neural network
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
# from keras.utils import np_utils
# from keras import KerasRegressor

# model related
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error

# ignore warnings
import warnings


def ignore_warn(*args, **kwargs):
    pass


warnings.warn = ignore_warn

# read train and test csv file
train = pd.read_csv('train_data_after_process.csv')
test = pd.read_csv('test_data_after_process.csv')

# print('The shape of training data:', train.shape)
# print('The shape of testing data:', test.shape)
# # check for null/missing values
# print('missing values: ', train.isnull().sum().any())

# checking for skewness of saleprice
y_price = train['SalePrice']
y = train['SalePrice']
# print('Skewness of target:', y.skew())
# print('kurtosis of target:', y.kurtosis())
sns.distplot(y, fit=norm)
# plt.show()

# right skewed
y = np.log1p(y)
# print('Skewness of target:', y.skew())
# print('kurtosis of target:', y.kurtosis())
sns.distplot(y, fit=norm)
# plt.show()


train = train.drop('SalePrice', axis=1)
# check if both dataset have equal dimension
# print('The shape of training data:', train.shape)
# print('The length of y:', len(y))
# print('The shape of testing data:', test.shape)

# 10-fold validation method
n_folds = 10


def rmse_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=20)
    rmse = np.sqrt(-cross_val_score(model, train.values, y,
                   scoring='neg_mean_squared_error', cv=kf))
    return (rmse)


# NN - keras
model1 = Sequential()

model1.add(Dense(128, input_shape=(159,)))
model1.add(Activation('relu'))
model1.add(Dropout(0.5))

model1.add(Dense(64))
model1.add(Activation('relu'))
model1.add(Dropout(0.5))

model1.add(Dense(16))
model1.add(Activation('relu'))
model1.add(Dropout(0.5))

model1.add(Dense(1))

model1.compile(loss='mean_squared_error', metrics=['mse'], optimizer='adam')

train1 = np.asarray(train).astype('float32')
history = model1.fit(train1, np.ravel(y_price), epochs=500, verbose=2)

# transform into an submission file to kaggle
# Plotting the Accuracy Metrics
# fig = plt.figure()
# plt.subplot(2, 1, 1)
# plt.plot(history.history['mse'])
# plt.plot(history.history['loss'])
# plt.title('Model Accuracy')
# plt.ylabel('Mean Square Error')
# plt.xlabel('Loss')
# plt.legend(['Mean Square Error', 'Loss'], loc='lower right')
# fig

# plt.show()
# train = np.asarray(train).astype('float32')
# pred = model1.fit(train, y)

# pred = pred.reshape(1459,)
# list_a = []
# for i in range(1461, 2920):
#     list_a.append(i)

# submission_df = pd.DataFrame({'Id': list_a, 'SalePrice': pred})
# submission_df.to_csv('Sample_Submission_NN.csv', index=False)


# Parameter Tuning
# Lasso
lasso_alpha = [0.00005, 0.0001, 0.0002, 0.0005, 0.001,
               0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
lasso = make_pipeline(RobustScaler(), LassoCV(
    alphas=lasso_alpha, random_state=2))

# ElasticNet
enet_beta = [0.1, 0.2, 0.5, 0.6, 0.8, 0.9]
enet_alpha = [0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01]
ENet = make_pipeline(RobustScaler(), ElasticNetCV(
    l1_ratio=enet_beta, alphas=enet_alpha, random_state=12))

# Ridge
ridge_alpha = [0.00005, 0.0001, 0.0002, 0.0005, 0.001,
               0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=ridge_alpha))

# Gradient Boosting
gbr_params = {'loss': 'huber',
              'criterion': 'friedman_mse',
              'learning_rate': 0.1,
              'n_estimators': 600,
              'max_depth': 4,
              'subsample': 0.6,
              'min_samples_split': 20,
              'min_samples_leaf': 5,
              'max_features': 0.6,
              'random_state': 32,
              'alpha': 0.5}
gbr = GradientBoostingRegressor(**gbr_params)

# LightGBM
lgbr_params = {'learning_rate': 0.01,
               'n_estimators': 1850,
               'max_depth': 4,
               'num_leaves': 20,
               'subsample': 0.6,
               'colsample_bytree': 0.6,
               'min_child_weight': 0.001,
               'min_child_samples': 21,
               'random_state': 42,
               'reg_alpha': 0,
               'reg_lambda': 0.05}
lgbr = lgb.LGBMRegressor(verbose=-1, **lgbr_params)

# XGBoost
xgbr_params = {'learning_rate': 0.01,
               'n_estimators': 3000,
               'max_depth': 5,
               'subsample': 0.6,
               'colsample_bytree': 0.7,
               'min_child_weight': 3,
               'seed': 52,
               'gamma': 0,
               'reg_alpha': 0,
               'reg_lambda': 1}
xgbr = xgb.XGBRegressor(**xgbr_params)

models_name = ['Lasso', 'ElasticNet', 'Ridge',
               'Gradient Boosting', 'LightGBM', 'XGBoost']
models = [lasso, ENet, ridge, gbr, lgbr, xgbr]

for i, model in enumerate(models):
    score = rmse_cv(model)
    print('{} RMSE score with 10-fold : {}({})'.format(
        models_name[i], score.mean(), score.std()))

# stacking model parameter tuning
stack_model = StackingCVRegressor(regressors=(lasso, ENet, ridge, gbr, lgbr, xgbr), meta_regressor=lasso,
                                  use_features_in_secondary=True)

# Lasso
lasso_trained = lasso.fit(np.array(train), np.array(y))

# ElasticNet
ENet_trained = ENet.fit(np.array(train), np.array(y))

# Ridge
rid_trained = ridge.fit(np.array(train), np.array(y))

# Gradient Boosting
gbr_trained = gbr.fit(np.array(train), np.array(y))

# LightGBM
lgbr_trained = lgbr.fit(np.array(train), np.array(y))

# XGBoost
xgbr_trained = xgbr.fit(np.array(train), np.array(y))

# Stacking
stack_model_trained = stack_model.fit(np.array(train), np.array(y))


def rmse(y, y_preds):
    return np.sqrt(mean_squared_error(y, y_preds))


# adding stacking model
models.append(stack_model)
models_name.append('Stacking_model')
for i, model in enumerate(models):
    y_preds = model.predict(np.array(train))
    model_score = rmse(y, y_preds)
    print('RMSE of {}: {}'.format(models_name[i], model_score))

# submission file to kaggle
# sample_submission = pd.read_csv('sample_submission.csv')
# for i, model in enumerate(models):
#   preds = model.predict(np.array(test))
#   submission = pd.DataFrame({'Id': sample_submission['Id'], 'SalePrice': np.expm1(preds)})
#   submission.to_csv('House_Price_submission_'+models_name[i]+'_optimation.csv', index=False)
#   print('{} finished.'.format(models_name[i]))


# Blending
# average blending
preds_in_train = np.zeros((len(y), len(models)))
for i, model in enumerate(models):
    preds_in_train[:, i] = model.predict(np.array(train))
average_preds_in_train = preds_in_train.mean(axis=1)
average_score = rmse(y, average_preds_in_train)
print('RMSE of average model on training data:', average_score)

# weighted blending
model_weights = [0.15, 0.12, 0.08, 0.08, 0.12, 0.15, 0.3]
weight_preds_in_train = np.matmul(preds_in_train, model_weights)
weight_score = rmse(y, weight_preds_in_train)
print('RMSE of weight model on training data:', weight_score)
# weighted blending is better


# Below is the holdout method, and process of obtaining the graph
# # Spliting traning set and testing set, holdout method
# X_train, X_val, y_train, y_val = train_test_split(train, y, test_size=0.2, random_state=42)
# print("Training set shape (X_train):", X_train.shape)
# print("Validation set shape (X_val):", X_val.shape)
# print("Training target shape (y_train):", y_train.shape)
# print("Validation target shape (y_val):", y_val.shape)


# # training neural net model
# nn_model = Sequential()
# nn_model.add(Dense(128, input_shape=(X_train.shape[1],)))
# nn_model.add(Activation('relu'))
# nn_model.add(Dropout(0.5))
# nn_model.add(Dense(64))
# nn_model.add(Activation('relu'))
# nn_model.add(Dropout(0.5))
# nn_model.add(Dense(16))
# nn_model.add(Activation('relu'))
# nn_model.add(Dropout(0.5))
# nn_model.add(Dense(1))
# nn_model.compile(loss='mean_squared_error', metrics=['mse'], optimizer='adam')

# # transform the data type
# X_train_nn = np.asarray(X_train).astype('float32')
# y_train_nn = np.asarray(y_train).astype('float32')

# # training
# history = nn_model.fit(X_train_nn, y_train_nn, epochs=500, verbose=2)

# # predict sale price on testing set
# X_val_nn = np.asarray(X_val).astype('float32')
# y_val_pred_nn = nn_model.predict(X_val_nn).flatten()
# y_val_pred_nn = np.expm1(y_val_pred_nn)  # 转换回原始尺度

# # color list
# colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow']

# # perpare canvas
# plt.figure(figsize=(12, 8))

# # train every model
# for i, model in enumerate(models):
#     # training model
#     model.fit(X_train, y_train)

#     # predict on testing set
#     y_val_pred_log = model.predict(X_val)
#     y_val_pred = np.expm1(y_val_pred_log)

#     # scatter plot
#     plt.scatter(np.expm1(y_val), y_val_pred, alpha=0.7, color=colors[i], label=models_name[i])

# # scatter plot for neural net
# plt.scatter(np.expm1(y_val), y_val_pred_nn, alpha=0.7, color='black', label='Neural Network')


# # actual sale price plot
# plt.plot([np.expm1(y_val).min(), np.expm1(y_val).max()], [np.expm1(y_val).min(), np.expm1(y_val).max()], 'r--')

# # labels and title
# plt.xlabel('Actual SalePrice')
# plt.ylabel('Predicted SalePrice')
# plt.title('Comparison of Model Predictions')
# plt.legend()

# # showing the graph
# plt.show()
