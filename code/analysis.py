#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sys, os
import pandas as pd
import math

from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import r2_score

#pd.set_option('display.max_rows', 600)
#pd.set_option('display.max_columns', 600)

def make_date_player_ids(df):

    # create temporary data structures 
    one_year_df = pd.DataFrame()
    player_id = np.array([])
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
        temp_player_id = np.array(df['playerid'], dtype='int')
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
        player_id = np.hstack((player_id, temp_player_id))
        obp = np.hstack((obp, temp_obp))
        age = np.hstack((age, temp_age))
        fut_obp = np.hstack((fut_obp, temp_fut_obp))
        fut_pa = np.hstack((fut_pa, temp_fut_pa))
        pa = np.hstack((pa, temp_pa))

    # make new dataframe from the restructured data
    new_df = pd.DataFrame({"id": date_id, "playerid": player_id, "age": age, "pa": pa, "obp": obp, "fut_obp": fut_obp, "fut_pa": fut_pa}, index=date_id)

    # in order to gain ***some*** data to predict the 2021 players who didn't play in 2020, pull most recent data forward
        # i know this is cut on ***obp==0*** and not ***pa==0*** but obp==0 is not useful for predicting anything either
    zero_df = new_df.loc[(new_df['obp']==0) & (new_df['id']>202000000)]
    # iterate through the rows, stopping when the most recent nonzero OBP is found
    for i in (zero_df.index):
        # starting in 2017, check back a year for data
        for j in range(1,5): 
            if new_df.at[i-j*100000, 'obp'] > 0:
                new_df.at[i,'obp'] = new_df.at[i-j*100000,'obp']
                new_df.at[i,'pa'] = new_df.at[i-j*100000,'pa']
                break
    
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
  
    # remove pre-2020 limited data from df
    df = df.drop(df['id'].loc[(df['obp']==0) & (df['id']<202000000) | (df['pa']<=30) & (df['id']<202000000)])

    # transform to standardized data for regression
    data = df[['age','pa','obp','fut_obp']].to_numpy()
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    X = data[:, 0:3]
    y = data[:,3]


    # below is several regressions i attempted in order to find a good model
        # MLP and Linear Regressions both proved less useful than SVR

#    # grid search for ideal layer size for MLP
#    regr = MLPRegressor(random_state=1, max_iter=5000).fit(X, y )
#    grid = GridSearchCV(regr, parameters)
#    grid.fit(X,y)
#    print(grid.cv_results_)
#    print("summarized results:\n grid best index, grid best score, hidden_layer_size")
#    print(grid.best_index_, grid.best_score_, layers[grid.best_index_])
#    regr = MLPRegressor(random_state=1, max_iter=5000, hidden_layer_sizes=layers[grid.best_index_]).fit(X, y )

    # split training data into train test samples
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=1)

    # run only the determined best case for the MLP
    regr = MLPRegressor(random_state=1, max_iter=5000, hidden_layer_sizes=(15,2)).fit(X_train,y_train )
    print("MLP r2: ", regr.score(X_test,y_test))


    regr = LinearRegression().fit(X_train,y_train)
    print("Linear r2: ",regr.score(X_test,y_test))

    # run SVR fit on past data
    regr = SVR().fit(X_train,y_train)

    print("SVR r2: ", regr.score(X_test,y_test))
    
    # SVR performs best so we continue with that

    # construct the data for the prediction
    df = df.loc[df['id']>202000000]
    data = df.loc[df['id'] > 202000000][['age','pa','obp','fut_obp']].to_numpy()
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    X = data[:, 0:3]
    y = data[:,3]

    # inverse transform (by hand) the results of the prediction
    df['predict'] = (np.sqrt(scaler.var_[2])*regr.predict(X)+scaler.mean_[2])

    # sample obp from normal with mean == training data mean, and variance == training data var
    for i in df.loc[(df['obp']==0)].index:
        np.random.seed(int(i))
        df.at[i,'predict'] = np.random.normal(loc=scaler.mean_[2], scale=scaler.var_[2])


    # display r2 results for the "good" data
    no_zero_df = df.loc[df['obp']>0]
    print("R**2 excluding sampled 2021 predictions: ", r2_score(no_zero_df['fut_obp'],no_zero_df['predict']))

    # display r2 results for the "good" data
    no_zero_df = df.loc[(df['obp']>0) & (df['pa']>50)]
    print("R**2 excluding sampled 2021 predictions, and pa < 50: ", r2_score(no_zero_df['fut_obp'],no_zero_df['predict']))

    # set index to playerid (used for join)
    df = df.set_index('playerid')

    return df, scaler.mean_[2]

def main():

    # load data
    df = pd.read_csv('obp.csv')
    # convert nans to 0
    df = df.fillna(0)
    # convert birth date to date
    df['birth_date'] = pd.to_datetime(df['birth_date'])
    # calculate age
    df['age'] = df['birth_date'].apply(lambda x: pd.Timedelta(pd.Timestamp('2022-04-01') - x).days)
    # construct collapsed dataset
    new_df = make_date_player_ids(df) 
    # regress data
    new_df, scaler_mean = regres(new_df)
    # join regressed data back into main df
    df = df.join(new_df.loc[new_df['id'] > 202000000]['predict'], on='playerid')
    # calculate score for individual
    df['score'] = (df['OBP_21'] - df['predict'])**2
    # show overall r2
    print("R**2 including sampled 2021 predictions: ", 1- df['score'].sum() / ((df['OBP_21']-df['OBP_21'].mean())**2).sum())
    # write output to file
    df.to_csv('predict.csv')

    # show results
    plt.plot(np.arange(1000)*0.5/(1000), np.arange(1000)*0.5/(1000), label='OBP_21==predict' )
    plt.axhline(scaler_mean, color='g', linestyle='dashed', label='OBP_20 mean (includes \"futurized\" OBP)' )
    plt.plot(df.loc[df['PA_20']>0]['OBP_21'],df.loc[df['PA_20']>0]['predict'], 'bo', linestyle='None' , label='available 2020 data')
    plt.plot(df.loc[df['PA_20']==0]['OBP_21'],df.loc[df['PA_20']==0]['predict'], 'rx', linestyle='None', label='sampled 2021 prediction' )
    plt.plot(df.loc[df['PA_20']>50]['OBP_21'],df.loc[df['PA_20']>50]['predict'], 'ks', linestyle='None', fillstyle='none', label='PA_20>50 data' )
    plt.ylabel('predicted 2021 OBP')
    plt.xlabel('actual 2021 OBP')
    plt.legend()   
    plt.savefig('results.png')

    df = df.loc[df['PA_20'] >50 ]
    print("R**2 excluding PA_20 <= 50 predictions: ", 1- df['score'].sum() / ((df['OBP_21']-df['OBP_21'].mean())**2).sum())

if __name__ == "__main__":
    main()
