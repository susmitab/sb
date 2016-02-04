import pandas
import numpy as np
import re
import datetime
from datetime import timedelta
from sklearn.ensemble import RandomForestClassifier



train =  pandas.read_csv("train.csv", parse_dates=['Dates'])
test = pandas.read_csv("test.csv", parse_dates=['Dates'])


def feature_engineering(data):
    data['Day'] = data['Dates'].dt.day
    data['Month'] = data['Dates'].dt.month
    data['Year'] = data['Dates'].dt.year
    data['Hour'] = data['Dates'].dt.hour
    data['Minute'] = data['Dates'].dt.minute
    data['DayOfWeek'] = data['Dates'].dt.dayofweek
    data['WeekOfYear'] = data['Dates'].dt.weekofyear
    return data
#    
train = feature_engineering(train)
test = feature_engineering(test)

#Clean dataset
train = train.drop(['Descript', 'Resolution', 'Address'], axis = 1)
test = test.drop(['Address'], axis = 1)

#Define predictors
predictors= ['Year','WeekOfYear','DayOfWeek','Hour','Minute', 'X','Y']

#Select algorith & fit
alg = RandomForestClassifier(n_estimators=20)
alg.fit(train[predictors], train["Category"])
##alg.score(crime_sf[predictors], crime_sf["Category"])
#
predictions = alg.predict(test[predictors])

#Make submission file for kaggle
y = train['Category'].astype('category')
submit = pandas.DataFrame({'Id': test.Id.tolist()})

for category in y.cat.categories:
    submit[category] = np.where(predictions == category, 1, 0)
    
submit.to_csv('kaggle_SFOcrime.csv', index = False)    




