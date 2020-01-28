# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#from tensorflow.python import keras
#from tensorflow.python.keras.models import Sequential
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
from sklearn import preprocessing

#from keras.callbacks import ModelCheckpoint
#from keras.models import Sequential
#from keras.layers import Dense, Activation, Flatten


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os

train = pd.read_csv('../input/train_V2.csv')
test = pd.read_csv('../input/test_V2.csv')

train['TotalMoving'] = train['rideDistance'] + train['walkDistance'] + train['swimDistance'] 
test['TotalMoving'] = test['rideDistance'] + test['walkDistance'] + test['swimDistance']

train['NormToMov'] = train['TotalMoving']/train['matchDuration']
test['NormToMov'] = test['TotalMoving']/test['matchDuration']


#train['MeaningfulDamage'] = np.where(train['kills']+train['assists'] == 0,0,train['damageDealt']/(train['kills']+train['assists']))
#test['MeaningfulDamage'] = np.where(test['kills']+test['assists'] == 0,0,test['damageDealt']/(test['kills']+test['assists']))


train.drop(2744604,inplace = True) #WinPercPlace is null
train['killsWithoutMoving'] = ((train['kills']>20) & (train['walkDistance'] < 2000))
train.drop(train[train['killsWithoutMoving']==True].index,inplace = True)
train.drop(train[train['TotalMoving'] == 0].index,inplace = True )
train.drop(train[(train['headshotKills'] == train['kills']) & (train['kills']>5)].index,inplace = True)
train.drop(train[train['longestKill']>=200].index, inplace = True)
train.drop(train[train['swimDistance']>=500].index,inplace = True)
train.drop(train[train['weaponsAcquired'] >= 50].index,inplace = True)
train.drop(train[train['heals'] >= 50].index, inplace = True)
train.drop(train[train['boosts']>= 20].index,inplace = True)
train.drop(train[train['roadKills'] > 3].index, inplace=True)
train.drop(train[train['kills'] > 20].index, inplace=True)
train.drop(train[train['matchDuration']<1000].index,inplace=True)
train.drop(train[train['killStreaks']>5].index,inplace = True)
train.drop(train[train['matchType'] == 'crashfpp'].index,inplace = True)
train.drop(train[train['matchType'] == 'normal-squad'].index,inplace = True)
train.drop(train[train['matchType'] == 'crashfpp'].index,inplace = True)
train.drop(train[train['matchType'] == 'normal-solo'].index,inplace = True)
train.drop(train[train['matchType'] == 'normal-duo'].index,inplace = True)
train.drop(train[train['matchType'] == 'normal-squad-fpp'].index,inplace = True)
train.drop(train[train['matchType'] == 'normal-duo-fpp'].index,inplace = True)
train.drop(train[train['matchType'] == 'normal-solo-fpp'].index,inplace = True)
train.drop(train[train['matchType'] == 'flarefpp'].index,inplace = True)
train.drop(train[train['revives'] > 15].index,inplace=True)



#detecting outliers refering https://www.kaggle.com/carlolepelaars/pubg-data-exploration-rf-funny-gifs
print("Done for Outlier")
"""train['matchDuration'].value_counts().sort_index().plot['winPlacePerc'ind].line()
train['matchDuration'][train['DBNOs']>10].value_counts().sort_index().plot.line()
train['killStreaks'][train['killStreaks']>3].value_counts().sort_index().plot.bar()
train['matchType'][train['matchType'] == 'normal-squad-fpp'].value_counts().plot.bar()
train['revives'][train['revives']>6].value_counts().plot.bar()
train['vehicleDestroys'].value_counts().sort_index().plot.bar()
train['numGroups'].value_counts().sort_index().plot.line()"""


#tr_copy = train.copy()
#tr_copy[['groupId','Id','winPlacePerc']].set_index('groupId').sort_index()
#tr_copy['winPlacePerc'].groupby(train['groupId'])
#tr_copy[['killPlace','winPlacePerc']].set_index('killPlace').sort_index()
#tr_copy.groupby('groupId').groups.keys()
aggregations = {
    'kills':{
        'total_kills' : 'sum'
    },
    'assists':{
        'total_assists' : 'sum',
        'average_assists' : 'mean'
    },
    'NormToMov':{
        'total_NormToMov' : 'sum',
        'average_NormToMov' : 'mean'
    },
    'boosts' : {
        'total_boosts' : 'sum',
        'average_boosts' : 'mean'
    },
    'weaponsAcquired' : {
        'total_weaponsAcquired' : 'sum',
        'average_weaponsAcquired' : 'mean'
    },
    'killPlace' : {
        'max_killPlace' : 'max',
        'min_killPlace' : 'min',
        'average_killPlace' : 'mean'
    },
    'damageDealt' : {
        'mean_damageDealt' : 'mean',
        'total_damageDealt' : 'sum'
    },
    'DBNOs' : {
        'total_DBNOs' : 'sum'
    }
}
grouped = train.groupby('groupId',as_index=False).agg(aggregations)
#grouped : has groupId so know which data to match later
X_2 = grouped.set_index('groupId')
#X_2 : delete groupId to use scaler

scaler = preprocessing.MinMaxScaler(feature_range = (-1,1),copy = False).fit(X_2)
scaler.transform(X_2)
#use scaler

grouped_y = train.groupby('groupId',as_index=False).agg({'winPlacePerc' : 'mean'})
y = grouped_y['winPlacePerc']
#y : to use as one dimensional y, delete groupId on grouped_y

"""k = 35
corrmat = X_2.corr()
cols = corrmat.nlargest(k,'winPlacePerc').index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
f, ax = plt.subplots(figsize=(14,10))
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt = '.2f',annot_kws={'size':8},yticklabels = cols.values,xticklabels=cols.values)
plt.show"""

train_X, temp_X, train_y, temp_y = train_test_split(X_2,y,random_state = 0,test_size=0.2)
test_X,val_X,test_y,val_y = train_test_split(temp_X,temp_y,random_state = 0, test_size = 0.5)

model_1 = XGBRegressor(n_estimator = 1000, learning_rate = 0.5, max_depth = 7)
model_1.fit(train_X,train_y, early_stopping_rounds = 20, eval_set = [(test_X,test_y)],verbose = False)
preds = model_1.predict(val_X)
print(" MAE is :",mean_absolute_error(val_y,preds))
#done for training

grouped_test = test.groupby('groupId',as_index=False).agg(aggregations)
T = grouped_test.set_index('groupId')
scaler.transform(T)
val_predictions = model_1.predict(T)
T.columns = T.columns.droplevel(level = 0)
T['winPlacePerc'] = val_predictions
T.head()
T.info()
df = pd.merge(test,T,on='groupId',how='right')
df.info()

#submission_model = XGBRegressor(n_estimator = 1000,learning_rate = 0.5)
#submission_model.fit(train_X,train_y,early_stopping_rounds = 5, eval_set = [(test_X,test_y)],verbose = False)
#print(val_predictions)
#val_predictions.info()
submission = pd.DataFrame(
        {'Id' : df.Id , 'winPlacePerc' : df.winPlacePerc },
        columns = ['Id', 'winPlacePerc'])
submission.to_csv('submission.csv',index = False)
#do it for later. 
#NaN in data should be replaced by others.
#ax = sns.heatmap(train, linewidth = 1)
#plt.show()

# Any results you write to the current directory are saved as output.
