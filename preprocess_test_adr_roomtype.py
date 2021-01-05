import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from argparse import ArgumentParser

parser = ArgumentParser()
args = parser.parse_args()

test_d = pd.read_csv("./Data/test.csv")


def strTodatetime(datestr, format):
    return datetime.datetime.strptime(datestr, format)

hotel_label = []
arrival_date = []
country_label = []
#revenue = []

for i in range(test_d.shape[0]):

    """
    整理既有Data
    """
    #hotel
    if test_d["hotel"][i] == 'Resort Hotel':
        hotel_label.append(0)
    else:
        hotel_label.append(1)
    
    #date
    arr_date = str(test_d['arrival_date_year'][i]) + str(test_d['arrival_date_month'][i])+ str(test_d['arrival_date_day_of_month'][i])
    arrival_date.append(strTodatetime(arr_date,"%Y%B%d"))
        
    #Country, 先把'PRT'設為1, 其餘為0
    if test_d["country"][i] == 'PRT':
        country_label.append(1)
    else:
        country_label.append(0)
            
    if i % 10000 == 0:
        print('{} finished'.format(round(i/test_d.shape[0], 2)))

test_d = test_d.join(pd.get_dummies(test_d.meal, prefix='meal')) #meal for one hot
test_d = test_d.join(pd.get_dummies(test_d.market_segment, prefix='segment')) #market_segment for one hot
test_d = test_d.join(pd.get_dummies(test_d.assigned_room_type, prefix = 'assigned_room'))
test_d = test_d.join(pd.get_dummies(test_d.deposit_type, prefix = 'deposit_type'))
test_d = test_d.join(pd.get_dummies(test_d.customer_type, prefix = 'customer_type'))
test_d = test_d.join(pd.get_dummies(test_d.distribution_channel, prefix = 'distribution_channel'))
test_d = test_d.join(pd.get_dummies(test_d.reserved_room_type, prefix = 'reserved_room_type'))

test_d['hotel_label'] = hotel_label
test_d['arrival_date'] = arrival_date
test_d['country_label'] = country_label

#Drop column
drop_list = ['meal','market_segment','assigned_room_type','deposit_type','customer_type','distribution_channel','reserved_room_type',
             'hotel','country','arrival_date_year','arrival_date_month','arrival_date_day_of_month',
             'ID','company','agent']
test_d = test_d.drop(drop_list, axis=1)

#補na值
for i in range(test_d.shape[0]):
    if np.isnan(test_d['children'][i]):
        test_d['children'][i] = 0

#test_d缺少幾個train_d有的column
#分別為segment_Undefined/ assigned_room_L/ distribution_channel_Undefined
segment_Undefined = np.zeros(test_d.shape[0])
assigned_room_L = np.zeros(test_d.shape[0])
distribution_channel_Undefined = np.zeros(test_d.shape[0])
reserved_room_type_L = np.zeros(test_d.shape[0])

test_d['segment_Undefined'] = segment_Undefined
test_d['assigned_room_L'] = assigned_room_L
test_d['distribution_channel_Undefined'] = distribution_channel_Undefined
test_d['reserved_room_type_L'] = reserved_room_type_L


test_d.to_csv('./test_preprocess.csv',index=False)