#!/usr/bin/env python
# coding: utf-8

import os
import dask
import dask.dataframe as dd
import pandas as pd
import time
import dask_ml
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from scipy_utils import make_cluster
from dask.distributed import Client
from dask_ml.model_selection import GridSearchCV as GridSearchCV_dask
from multiprocessing import Process, freeze_support
from sklearn.externals import joblib

class NYDataframe():
    def __init__(self, url, parameters):
        self.parameters = parameters
    def NYCflight():
        df = dd.read_csv(os.path.join('data', 'nycflights', '*.csv'),
                    parse_dates={'Date': [0, 1, 2]},
                    dtype={'TailNum': str,
                            'CRSElapsedTime': float,
                            'Cancelled': bool})


        df_dry = df.query("Cancelled == False")
        df_dry = df_dry.drop(['TailNum','Cancelled','TaxiIn','TaxiOut'], axis=1)
        df_dry = df_dry.set_index('Date')
        pd_df = df_dry.compute()
        pd_df = pd_df.select_dtypes(exclude = 'object')

        pd_df.fillna(value=0, inplace=True)

        return pd_df

    def train_test(pd_df, test_size, random_state):
        X_df = pd_df.drop(['DepDelay'],axis=1)
        y = pd_df.DepDelay
        test_size = test_size
        X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test 

        
    def RandomForestSkLearn(param_grid, X_train, X_test, y_train, y_test):
        estimator = RandomForestRegressor()

        param_grid = param_grid
        grid_search_random_forest = GridSearchCV(estimator, param_grid, cv=2, n_jobs=-1)
        grid_search_random_forest.fit(X_train, y_train)
        r_2 = grid_search_random_forest.score(X_test, y_test)
        mse = mean_squared_error(grid_search_random_forest.predict(X_test), y_test)
        return r_2, mse

    def RandomForestDask(param_grid, X_train, X_test, y_train, y_test):

        cluster = make_cluster()
        cluster
        client = Client(cluster)
        client
        dask_X_train = dd.from_pandas(X_train, npartitions=3)        # preprocess data
        dask_y_train = dd.from_pandas(y_train, npartitions=3)

        dask_X_test = dd.from_pandas(X_test, npartitions=3)        
        dask_y_test = dd.from_pandas(y_test, npartitions=3)

        estimator = RandomForestRegressor()
        param_grid = param_grid

        grid_search_dask = GridSearchCV_dask(estimator, param_grid, cv=2, n_jobs=-1)
        with joblib.parallel_backend("dask", scatter=[dask_X_train, dask_y_train]):
            grid_search_dask.fit(dask_X_train, dask_y_train)
        grid_search_dask.score(dask_X_test, dask_y_test)
        r_2 = grid_search_dask.best_estimator_.score(dask_X_test, dask_y_test)
        mse = mean_squared_error(grid_search_dask.best_estimator_.predict(X_test), y_test)
        return r_2, mse, 
    



