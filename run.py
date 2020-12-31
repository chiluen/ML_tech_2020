import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser

from model.xgb import model_xgb
from utils import rmse


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

#----Training----#

#train for total revenue(adr * number of people)
model = model_xgb()

X_train, X_test, y_train, y_test = train_test_split(train_d, revenue, test_size=0.2)
model.train(X_train, y_train)

print("Start to train")
preds = model.predict(X_test)
error = rmse(y_train, preds)

print("Rmse: {}".format(error))




