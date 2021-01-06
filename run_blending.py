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
BLENDING_NUMBER = 5

#----Data load----#
train_d = pd.read_csv(PREPROCESSED_DATA)
train_d_label = pd.read_csv(DATA_ROOT + "/train_label.csv")

#幾個column要拿掉
adr = train_d.pop('adr')
canceled_list = train_d.pop('is_canceled')
date_time = train_d.pop('arrival_date')

train_label_df = pd.concat([adr, canceled_list, date_time], axis=1)  #把這三者組在一起


X_train, X_test, y_train, y_test = train_test_split(train_d, train_label_df, test_size=0.01)

#----Training for total revenue(adr * number of people)----#
print("Start to train for total adr")
n_estimators_list = [100, 300, 500, 700, 900]
max_depth_list = [4,5,6,7,8]
model_reg_list = []
for i in range(BLENDING_NUMBER):
    model_reg = model_xgb_reg(learning_rate = 0.065, #原本設0.1
                            n_estimators = n_estimators_list[i], #原本設100
                            max_depth = max_depth_list[i], #原本設5
                            min_child_weight = 2, #原本設1
                            gamma = 0 #原本設0
                            )
    model_reg.fit(X_train, y_train['adr'])


    #Eval
    preds = model_reg.predict(X_test)
    error = rmse(y_test['adr'], preds)
    print("Rmse for E_val: {}".format(error))

    #Ein
    preds = model_reg.predict(X_train)
    error = rmse(y_train['adr'], preds)
    print("Rmse for E_in: {}".format(error))
    print("Finish model {}".format(i+1))
    model_reg_list.append(model_reg)

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


#----Predict for final quantize adr----#

##這邊我沒有分valid, 因為如果要做quantize revenue預測, valid應該要用"時間"來切, 就簡單做training set的testing
model_reg.fit(train_d, train_label_df['adr'])
adr_predict = model_reg.predict(train_d)
canceled_predict = model_cls.predict(train_d)

revenue_predict = adr_predict * (train_d['stays_in_week_nights'] + train_d['stays_in_weekend_nights'])

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
    test_date_time = test_d.pop('arrival_date')
    test_d = test_d[train_d.keys()] #把column排序 
    output_ans = pd.read_csv(DATA_ROOT+'/test_nolabel.csv')
    quantized_profit_list_total = np.zeros(output_ans.shape[0]) 

    for i in range(BLENDING_NUMBER):
        test_d = pd.read_csv(PREPROCESSED_TEST_DATA)
        test_date_time = test_d.pop('arrival_date')
        test_d = test_d[train_d.keys()] #把column排序

        adr_predict = model_reg_list[i].predict(test_d)
        canceled_predict = model_cls.predict(test_d)

        revenue_predict = adr_predict * (test_d['stays_in_week_nights'] + test_d['stays_in_weekend_nights'])

        profit_list = []
        profit_per_day = 0
        now_date = test_date_time[0]
        for i in range(test_d.shape[0]):
            if now_date == test_date_time[i]:
                profit_per_day += revenue_predict[i] * abs(canceled_predict[i] - 1)
            else:
                now_date = test_date_time[i]
                profit_list.append(profit_per_day)
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

        quantized_profit_list_total += np.array(quantized_profit_list)

    
    #把vote後的答案取平均
    quantized_profit_list_total = quantized_profit_list_total / BLENDING_NUMBER

    output_ans['label'] = quantized_profit_list_total
    output_ans.to_csv('./submit.csv',index=False)
    
    print(test_d.shape)
    print(revenue_predict.shape)
    print("done")