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
#country_label = []
revenue = []

countries = ['PRT', 'GBR', 'FRA', 'ESP', 'DEU']

'''
#drop data which has no country
train_d = train_d.drop(['agent','company'],axis=1)
train_d = train_d.dropna(axis = 0)
'''

#Deal with Outliers
train_d.loc[train_d.lead_time > 500,'lead_time'] = 500
train_d.loc[train_d.days_in_waiting_list > 0,'days_in_waiting_list'] = 1
train_d.loc[train_d.stays_in_weekend_nights >=  5,'stays_in_weekend_nights'] = 5
train_d.loc[train_d.adults > 4,'adults'] = 4
train_d.loc[train_d.previous_bookings_not_canceled > 0,'previous_bookings_not_canceled'] = 1
train_d.loc[train_d.previous_cancellations > 0,'previous_cancellations'] = 1
train_d.loc[train_d.stays_in_week_nights > 10,'stays_in_week_nights'] = 10
train_d.loc[train_d.booking_changes > 5,'booking_changes']=  5
train_d.loc[train_d.babies > 8,'babies'] = 0
train_d.loc[train_d.required_car_parking_spaces > 5,'required_car_parking_spaces'] = 0
train_d.loc[train_d.children > 8,'children']  = 0
#combine children and babies together as kids
train_d['kids'] = train_d.children + train_d.babies
#combine total mumbers by adding kids and adults
train_d['total_members'] = train_d.kids + train_d.adults

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

    ''' 
    #Country, 先把'PRT'設為1, 其餘為0
    if train_d["country"][i] == 'PRT':
        country_label.append(1)
    else:
        country_label.append(0)
    '''

    if train_d["country"][i] not in countries:
        train_d['country'][i] = 'Others'
        
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
train_d = train_d.join(pd.get_dummies(train_d.reserved_room_type, prefix = 'reserved_room_type'))
train_d = train_d.join(pd.get_dummies(train_d.country, prefix = 'country'))

train_d['hotel_label'] = hotel_label
train_d['arrival_date'] = arrival_date
#train_d['country_label'] = country_label
#train_d['revenue'] = revenue

#Drop column
drop_list = ['meal','market_segment','assigned_room_type','deposit_type','customer_type','distribution_channel','reserved_room_type',
             'hotel','country','arrival_date_year','arrival_date_month','arrival_date_day_of_month',
             'reservation_status','reservation_status_date','ID',
             'company','agent']
train_d = train_d.drop(drop_list, axis=1)

#補na值
for i in range(train_d.shape[0]):
    if np.isnan(train_d['children'][i]):
        train_d['children'][i] = 0

#會包含
train_d.to_csv('./train_preprocess.csv',index=False)


