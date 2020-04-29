import numpy as np
import dask
import dask_ml
import dask_ml.xgboost
import time
import joblib
import dask.dataframe as dd
from sklearn.model_selection import GridSearchCV
from scipy_utils import make_cluster
from dask.distributed import Client
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV



class MyXGBooster():
    def __init__(self):
        pass 
    def daskbooster( X_train, X_test, y_train, y_test):
        
        cluster = make_cluster()
        cluster

        client = Client(cluster)
        client


        df_train = dd.from_pandas(X_train, npartitions=3)        # preprocess data
        labels_train = dd.from_pandas(y_train, npartitions=3)

        params = {'objective': 'reg:squarederror',
                'colsample_bytree':0.4,
                'learning_rate':0.07,
                'min_child_weight':1.5,
                'max_depth':5,
                'seed':42,
                'n_jobs':-1,
                'max_depth': 10}
        bst = dask_ml.xgboost.train(client, params, df_train, labels_train)

        #bst 
        y_test_dask = dd.from_pandas(y_test, npartitions=3)
        X_test_dask = dd.from_pandas(X_test, npartitions=3)
        predictions = dask_ml.xgboost.predict(client, bst, X_test_dask)
        mse = mean_squared_error(predictions.compute(), y_test_dask)

        return mse
    
    def skbooster(param_grid, X_train, X_test, y_train, y_test):
        estimator = XGBRegressor(objective= 'reg:squarederror', \
        random_state=42)
        parameters = param_grid
        grid_search_xgbregressor = GridSearchCV( estimator=estimator, \
                                    param_grid=parameters,\
                                    scoring = 'neg_mean_squared_error',\
                                    n_jobs = -1,\
                                    cv = 2,\
                                    verbose=True)
        grid_search_xgbregressor.fit(X_train, y_train)
        grid_search_xgbregressor.score(X_test, y_test)
        r_2 = grid_search_xgbregressor.best_estimator_.score(X_test, y_test)
        predictions = grid_search_xgbregressor.best_estimator_.predict(X_test)
        mse = mean_squared_error(predictions, y_test)
        return r_2, mse