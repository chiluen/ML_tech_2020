import numpy as np
import pandas as pd
#import ipdb
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser

from model.xgb import model_xgb_reg, model_xgb_cls
from utils import rmse, binary_cls_error, mae

"""
To do:
2. 把normalization的事情搞好
3. cross validation等等看要怎麼和testing data接起來
4. 把test data的preprocess做好
"""



parser = ArgumentParser()
parser.add_argument("-n", action='store_true', help='Do normalization or not')
parser.add_argument("-train", action='store_true')
parser.add_argument("-test", action='store_true')
parser.add_argument("-all")
args = parser.parse_args()

#----Path----#
PREPROCESSED_DATA = "./train_preprocess.csv"
PREPROCESSED_TEST_DATA = "./test_preprocess.csv"
DATA_ROOT = "./Data"
RESULT_ROOT = "./result"

#----Data load----#
train_d = pd.read_csv(PREPROCESSED_DATA)
train_d_label = pd.read_csv(DATA_ROOT + "/train_label.csv")
print(train_d.shape)

#幾個column要拿掉
revenue = train_d.pop('revenue')
canceled_list = train_d.pop('is_canceled')
date_time = train_d.pop('arrival_date')
print(train_d.shape)

train_label_df = pd.concat([revenue, canceled_list, date_time], axis=1)  #把這三者組在一起

print(train_label_df.shape)
print(train_label_df[0:10])
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
"""
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(train_d)
"""
X_train, X_test, y_train, y_test = train_test_split(train_d, train_label_df, test_size=0.3)

#----Training for total revenue(adr * number of people)----#

print("Start to train for total revenue")

model_reg = model_xgb_reg(learning_rate = 0.05, #原本設0.1
                          n_estimators = 700, #原本設100
                          max_depth = 5, #原本設5
                          min_child_weight = 2, #原本設1
                          gamma = 0.5 #原本設0
                          )
model_reg.train(X_train, y_train['revenue'])
#print(y_train['revenue'].shape)
"""
SVR modek testing
"""
"""
from sklearn.svm import SVR as SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
#n_samples, n_features = 10, 5
#rng = np.random.RandomState(0)
#y = rng.randn(n_samples)
#X = rng.randn(n_samples, n_features)
model_reg = SVR(C=1.0, epsilon=0.0, verbose=1)
#model_reg = SVR(kernel='rbf', C=1e3, gamma=0.1, verbose=True, max_iter=10000)
model_reg.fit(X_train[:], y_train['revenue'][:])
#regr.fit(X, y)
"""
"""
Random Forest
"""
"""
from sklearn.ensemble import RandomForestRegressor

model_reg = RandomForestRegressor(max_depth=20, random_state=0, n_estimators=200)
model_reg.fit(X_train, y_train['revenue'])
"""
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
print("Cls_error for E_val: {}".format(error))

#Ein
preds = model_cls.predict(X_train)
error = binary_cls_error(y_train['is_canceled'], preds)
print("Cls_error for E_in: {}".format(error))


#----Predict for final quantize revenue----#

##這邊我沒有分valid, 因為如果要做quantize revenue預測, valid應該要用"時間"來切, 就簡單做training set的testing
model_reg.train(train_d, train_label_df['revenue'])
revenue_predict = model_reg.predict(train_d)
canceled_predict = model_cls.predict(train_d)

profit_list = []
profit_per_day = 0
now_date = train_label_df['arrival_date'][0]
for i in range(train_d.shape[0]):
    if now_date == train_label_df['arrival_date'][i]:
        profit_per_day += revenue_predict[i] * abs(canceled_predict[i] - 1)
    else:
        now_date = train_label_df['arrival_date'][i]
        profit_list.append(profit_per_day)
        #profit_per_day = 0
        profit_per_day = revenue_predict[i] * abs(canceled_predict[i] - 1)
        
profit_list.append(profit_per_day)

quantized_profit_list = []
for i in range(len(profit_list)):
    if profit_list[i] < 10000:
        quantized_profit_list.append(0)
    elif 10000 <= profit_list[i] < 20000:
        quantized_profit_list.append(1)
    elif 20000 <= profit_list[i] < 30000:
        quantized_profit_list.append(2)
    elif 30000 <= profit_list[i] < 40000:
        quantized_profit_list.append(3)
    elif 40000 <= profit_list[i] < 50000:
        quantized_profit_list.append(4)
    elif 50000 <= profit_list[i] < 60000:
        quantized_profit_list.append(5)
    elif 60000 <= profit_list[i] < 70000:
        quantized_profit_list.append(6)
    elif 70000 <= profit_list[i] < 80000:
        quantized_profit_list.append(7)
    elif 80000 <= profit_list[i] < 90000:
        quantized_profit_list.append(8)
    else:
        quantized_profit_list.append(9)

error = mae(train_d_label['label'], quantized_profit_list)
print("Mae error for final result: {}".format(error))


if args.test:
    test_d = pd.read_csv(PREPROCESSED_TEST_DATA)
    output_ans = pd.read_csv(DATA_ROOT+'/test_nolabel.csv')
    test_date_time = test_d.pop('arrival_date')
    test_d = test_d[train_d.keys()] #把column排序

    revenue_predict = model_reg.predict(test_d)
    canceled_predict = model_cls.predict(test_d)

    
    profit_list = []
    profit_per_day = 0
    now_date = test_date_time[0]
    for i in range(test_d.shape[0]):
        if now_date == test_date_time[i]:
            profit_per_day += revenue_predict[i] * abs(canceled_predict[i] - 1)
        else:
            now_date = test_date_time[i]
            profit_list.append(profit_per_day)
            #profit_per_day = 0
            profit_per_day = revenue_predict[i] * abs(canceled_predict[i] - 1)
    profit_list.append(profit_per_day)

    quantized_profit_list = []
    for i in range(len(profit_list)):
        if profit_list[i] < 10000:
            quantized_profit_list.append(0)
        elif 10000 <= profit_list[i] < 20000:
            quantized_profit_list.append(1)
        elif 20000 <= profit_list[i] < 30000:
            quantized_profit_list.append(2)
        elif 30000 <= profit_list[i] < 40000:
            quantized_profit_list.append(3)
        elif 40000 <= profit_list[i] < 50000:
            quantized_profit_list.append(4)
        elif 50000 <= profit_list[i] < 60000:
            quantized_profit_list.append(5)
        elif 60000 <= profit_list[i] < 70000:
            quantized_profit_list.append(6)
        elif 70000 <= profit_list[i] < 80000:
            quantized_profit_list.append(7)
        elif 80000 <= profit_list[i] < 90000:
            quantized_profit_list.append(8)
        else:
            quantized_profit_list.append(9)
    output_ans['label'] = quantized_profit_list
    output_ans.to_csv('./submit.csv',index=False)
    
    print(test_d.shape)
    print(revenue_predict.shape)
    print("done")