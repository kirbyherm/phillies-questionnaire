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

# script, first, second = sys.argv

def make_date_player_ids(df):

    one_year_df = pd.DataFrame()
    date_id = np.array([])
    obp = np.array([])
    age = np.array([])
    fut_obp = np.array([])
    fut_pa = np.array([])
    pa = np.array([])
    for i in range(16,21):
        temp_date_id = np.array(df['playerid'], dtype='int')
        temp_age = np.array(df['age'], dtype='int')
        temp_fut_obp = np.array(df['OBP_{}'.format(i+1)])
        temp_age = temp_age - (365 * (22-i))
        temp_obp = np.array(df['OBP_{}'.format(i)])
        temp_pa = np.array(df['PA_{}'.format(i)])
        temp_fut_pa = np.array(df['PA_{}'.format(i+1)])
        maxID = (df['playerid'].max())
        min_new_ID = np.power(10,(math.floor(math.log10(maxID)+1)))
        for j in range(len(temp_date_id)):
            temp_date_id[j] = temp_date_id[j] + (2000 + i)*(min_new_ID)
        date_id = np.hstack((date_id, temp_date_id))
        obp = np.hstack((obp, temp_obp))
        age = np.hstack((age, temp_age))
        fut_obp = np.hstack((fut_obp, temp_fut_obp))
        fut_pa = np.hstack((fut_pa, temp_fut_pa))
        pa = np.hstack((pa, temp_pa))
    new_df = pd.DataFrame({"id": date_id, "age": age, "pa": pa, "obp": obp, "fut_obp": fut_obp, "fut_pa": fut_pa}, index=date_id)
    print(new_df)
    new_df = new_df.drop(new_df['id'].loc[(new_df['pa']<=20) | (new_df['fut_pa']<=20)])
    print(new_df)
    
    return new_df

def regres(df):

    data = df[['age','pa','obp','fut_obp']].to_numpy()
    print(data)
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    print(data)
    X = data[:, 0:3]
    y = data[:,3]
#    X_train, X_test, y_train, y_test = train_test_split(X, y,
#                                                        random_state=1)
    regr = MLPRegressor(random_state=1, max_iter=500).fit(X, y )
    print(scaler.mean_)
    df['predict'] = (np.sqrt(scaler.var_[2])*regr.predict(X)+scaler.mean_[2])
    print(regr.score(X, y))

    return df

def main():

    df = pd.read_csv('obp.csv')
    df = df.fillna(0)
    df['birth_date'] = pd.to_datetime(df['birth_date'])
    df['age'] = df['birth_date'].apply(lambda x: pd.Timedelta(pd.Timestamp('2022-04-01') - x).days)
    new_df = make_date_player_ids(df) 
    regres(new_df)
    

    print(new_df.sort_values(by='predict', ascending=False))
    print(new_df.sort_values(by='fut_obp', ascending=False))
    

if __name__ == "__main__":
    main()
