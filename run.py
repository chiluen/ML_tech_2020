import numpy as np
import pandas as pd
import ipdb
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser

from model.xgb import model_xgb_reg, model_xgb_cls
from utils import rmse, binary_cls_error

"""
To do:
1. 把model建置好
2. 把normalization的事情搞好
3. cross validation等等看要怎麼和testing data接起來
"""



parser = ArgumentParser()
parser.add_argument("-n", action='store_true', help='Do normalization or not')
parser.add_argument("-train", action='store_true')
parser.add_argument("-test", action='store_true')
parser.add_argument("-all")
args = parser.parse_args()

#----Path----#
PREPROCESSED_DATA = "./train_preprocess.csv"
RESULT_ROOT = "./result"

#----Data load----#
train_d = pd.read_csv(PREPROCESSED_DATA)

#幾個column要拿掉
revenue = train_d.pop('revenue')
canceled_list = train_d.pop('is_canceled')
date_time = train_d.pop('arrival_date')
train_label_df = pd.concat([revenue, canceled_list, date_time], axis=1)  #把這三者組在一起

#這個要寫成一個function, 每一次train_test_split要用
"""
if args.n:
    print('Do normalization')
    from sklearn.preprocessing import StandardScaler, MaxAbsScaler
    z_score_scaler = StandardScaler()
    maxab_scaler = MaxAbsScaler()

    #部份column不做normalize的原因：原本的column太過於sparse（大部分為0）
    #只要部份column的mean不會到太大(超過10), 就保留它原本的數值
    z_score_scaler.fit(train_d[['lead_time','arrival_date_week_number']])
    maxab_scaler.fit(train_d[['days_in_waiting_list']])
    train_d[['lead_time','arrival_date_week_number']] = z_score_scaler.transform(train_d[['lead_time','arrival_date_week_number']])
    train_d[['days_in_waiting_list']] = maxab_scaler.transform(train_d[['days_in_waiting_list']])
"""


X_train, X_test, y_train, y_test = train_test_split(train_d, train_label_df, test_size=0.2)

#----Training for total revenue(adr * number of people)----#

print("Start to train for total revenue")
model_reg = model_xgb_reg()
model_reg.train(X_train, y_train['revenue'])

#Eval
preds = model_reg.predict(X_test)
error = rmse(y_test['revenue'], preds)
print("Rmse for E_val: {}".format(error))

#Ein
preds = model_reg.predict(X_train)
error = rmse(y_train['revenue'], preds)
print("Rmse for E_in: {}".format(error))

#----Training for is_canceled----#

print("Start to train for is_canceled")
model_cls = model_xgb_cls()
model_cls.train(X_train, y_train['is_canceled'])

#Eval
preds = model_cls.predict(X_test)
error = binary_cls_error(y_test['is_canceled'], preds)
print("Rmse for E_val: {}".format(error))

#Ein
preds = model_cls.predict(X_train)
error = binary_cls_error(y_train['is_canceled'], preds)
print("Rmse for E_in: {}".format(error))
