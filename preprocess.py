import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from argparse import ArgumentParser

parser = ArgumentParser()
#parser.add_argument("-n", action='store_true', help='Do normalization or not')
args = parser.parse_args()

train_d = pd.read_csv("./Data/train.csv")

def strTodatetime(datestr, format):
    return datetime.datetime.strptime(datestr, format)

hotel_label = []
arrival_date = []
country_label = []
revenue = []

for i in range(train_d.shape[0]):

    """
    整理既有Data
    """
    #hotel
    if train_d["hotel"][i] == 'Resort Hotel':
        hotel_label.append(0)
    else:
        hotel_label.append(1)
    
    #date
    #import ipdb; ipdb.set_trace()
    arr_date = str(train_d['arrival_date_year'][i]) + str(train_d['arrival_date_month'][i])+ str(train_d['arrival_date_day_of_month'][i])
    arrival_date.append(strTodatetime(arr_date,"%Y%B%d"))
        
    #Country, 先把'PRT'設為1, 其餘為0
    if train_d["country"][i] == 'PRT':
        country_label.append(1)
    else:
        country_label.append(0)
        
    """
    新增Column
    """
    revenue.append(train_d['adr'][i] * (train_d['stays_in_week_nights'][i] + train_d['stays_in_weekend_nights'][i]))
    
    if i % 10000 == 0:
        print('{} finished'.format(round(i/train_d.shape[0], 2)))

train_d = train_d.join(pd.get_dummies(train_d.meal, prefix='meal')) #meal for one hot
train_d = train_d.join(pd.get_dummies(train_d.market_segment, prefix='segment')) #market_segment for one hot
train_d = train_d.join(pd.get_dummies(train_d.assigned_room_type, prefix = 'assigned_room'))
train_d = train_d.join(pd.get_dummies(train_d.deposit_type, prefix = 'deposit_type'))
train_d = train_d.join(pd.get_dummies(train_d.customer_type, prefix = 'customer_type'))
train_d = train_d.join(pd.get_dummies(train_d.distribution_channel, prefix = 'distribution_channel'))

train_d['hotel_label'] = hotel_label
train_d['arrival_date'] = arrival_date
train_d['country_label'] = country_label
train_d['revenue'] = revenue

#Drop column
drop_list = ['meal','market_segment','assigned_room_type','deposit_type','customer_type',
             'hotel','country','arrival_date_year','arrival_date_month','arrival_date_day_of_month',
             'adr','reservation_status','reservation_status_date','ID',
             'distribution_channel','company','agent','reserved_room_type']
train_d = train_d.drop(drop_list, axis=1)

#補na值
for i in range(train_d.shape[0]):
    if np.isnan(train_d['children'][i]):
        train_d['children'][i] = 0

#會包含
train_d.to_csv('./train_preprocess.csv',index=False)


