#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sys, os
import pandas as pd
import math

from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import cross_val_score, GridSearchCV

def make_date_player_ids(df):

    # create temporary data structures 
    one_year_df = pd.DataFrame()
    date_id = np.array([])
    obp = np.array([])
    age = np.array([])
    fut_obp = np.array([])
    fut_pa = np.array([])
    pa = np.array([])

    # iterate through the years 2016-2020 making separate indices for the player
        #  for each year and setting the age, obp, pa, fut_obp, fut_pa
        #  where fut_* is denoting the stat for the **NEXT** year
    # this approach sacrifices the ability to model change in obp over time
        # in exchange for providing more data for the model to incorporate
    for i in range(16,21):

        # set temporary data arrays before adding to dataframe
        temp_date_id = np.array(df['playerid'], dtype='int')
        temp_age = np.array(df['age'], dtype='int')
        temp_fut_obp = np.array(df['OBP_{}'.format(i+1)])
        temp_age = temp_age - (365 * (22-i))
        temp_obp = np.array(df['OBP_{}'.format(i)])
        temp_pa = np.array(df['PA_{}'.format(i)])
        temp_fut_pa = np.array(df['PA_{}'.format(i+1)])

        # create new player ids for each year
        maxID = (df['playerid'].max())
        min_new_ID = np.power(10,(math.floor(math.log10(maxID)+1)))
        for j in range(len(temp_date_id)):
            temp_date_id[j] = temp_date_id[j] + (2000 + i)*(min_new_ID)

        # append temporary data to the main structure
        date_id = np.hstack((date_id, temp_date_id))
        obp = np.hstack((obp, temp_obp))
        age = np.hstack((age, temp_age))
        fut_obp = np.hstack((fut_obp, temp_fut_obp))
        fut_pa = np.hstack((fut_pa, temp_fut_pa))
        pa = np.hstack((pa, temp_pa))

    # make new dataframe from the restructured data
    new_df = pd.DataFrame({"id": date_id, "age": age, "pa": pa, "obp": obp, "fut_obp": fut_obp, "fut_pa": fut_pa}, index=date_id)

    zero_df = new_df.loc[(new_df['obp']==0) & (new_df['id']>202000000)]
    print(zero_df)
    zero_df = new_df.loc[new_df['obp']==0]
    print(zero_df)
    for i in (zero_df.index):
        if i > 201700000:
#            print(new_df.at[i-100000, 'obp'])
            new_df.at[i,'obp'] = new_df.at[i-100000,'obp']
    zero_df = new_df.loc[(new_df['obp']==0) & (new_df['id']>202000000)]
    print(zero_df)
    #print(new_df)
    # reduce dataframe to only cases where the pa for the year are > 30
        # (note this does not incorporate the 2021 pa data)
#    new_df = new_df.drop(new_df['id'].loc[(new_df['pa']<=30)])
    print(new_df)
    
    return new_df

def regres(df):

    # create a list of hidden_layer_sizes to check
    layers = []
    
    for i in range(1,5):
        for j in np.logspace(0.8,1.3,10,endpoint=True,dtype=int):
            if i == 1:
                layers.append((j, ))
            elif i < j:
                layers.append((j, i))

    parameters = {"hidden_layer_sizes":layers}
  

    data = df[['age','pa','obp','fut_obp']].to_numpy()
    print(data)
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    print(data)
    X = data[:, 0:3]
    y = data[:,3]
    regr = MLPRegressor(random_state=1, max_iter=5000).fit(X, y )
    #grid = GridSearchCV(regr, parameters)
    #grid.fit(X,y)
    #print(grid.best_index_, grid.best_score_, layers[grid.best_index_])
    #print(grid.cv_results_)
    #X_train, X_test, y_train, y_test = train_test_split(X, y,
    #                                                    random_state=1)
    #regr = MLPRegressor(random_state=1, max_iter=5000, hidden_layer_sizes=layers[grid.best_index_]).fit(X, y )
#   # regr = LinearRegression().fit(X, y )
    #print(scaler.mean_)
    df = df.loc[df['id']>202000000]
    data = df.loc[df['id'] > 202000000][['age','pa','obp','fut_obp']].to_numpy()
    print(data)
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    print(data)
    X = data[:, 0:3]
    y = data[:,3]
    df['predict'] = (np.sqrt(scaler.var_[2])*regr.predict(X)+scaler.mean_[2])
    print(df.loc[df['obp']==0])

    for i in df.loc[df['obp']==0].index:
        df.at[i,'predict'] = np.random.normal(loc=scaler.mean_[2], scale=scaler.var_[2])
    print(df.loc[df['obp']==0])
    print(regr.score(X, y))

    return df

def main():

    df = pd.read_csv('obp.csv')
    df = df.fillna(0)
    df['birth_date'] = pd.to_datetime(df['birth_date'])
    df['age'] = df['birth_date'].apply(lambda x: pd.Timedelta(pd.Timestamp('2022-04-01') - x).days)
    new_df = make_date_player_ids(df) 
    new_df = regres(new_df)
    plt.plot(np.arange(1000)*0.5/(1000), np.arange(1000)*0.5/(1000) )
    plt.plot(new_df['predict'],new_df['fut_obp'], 'bo', linestyle='None' )
   

    print(new_df.sort_values(by='predict', ascending=False))
    print(new_df.sort_values(by='fut_obp', ascending=False))
    plt.show()
    

if __name__ == "__main__":
    main()
