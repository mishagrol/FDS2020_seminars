import time
import json
import argparse
import pandas as pd
import numpy as np
from loader import Download, Flights
from randomforest import NYDataframe
from xgboostreg import MyXGBooster


def main():
    parser = argparse.ArgumentParser(description='Args parser')
    parser.add_argument('--parameters', default='parameters.json', dest='config', type=str)
    parser.add_argument('--random_state', default=42, dest='random_state', action='store', type=int)
    parser.add_argument('--test_size', default=0.3, dest='test_size', action='store', type=float)
    parser.add_argument('--results_path', default='results.csv', dest='results_path', type=str)


    args = parser.parse_args()
    test_size = args.test_size
    random_state = args.random_state
    result_csv = str(args.results_path)

    with open(args.config) as json_file:
        parameters = json.load(json_file)

    data_dir = parameters['Working_dir']
    param_grid_rf = parameters['models']['RandomForest'][0]
    param_grid_xgb = parameters['models']['XGBoost'][0]
    url = parameters['URL']
    row_numbers = parameters['Row_number']

    #Load csv format data

    Flights(url, row_numbers)

    #Load dask.dataframe and feature engineering

    pd_df = NYDataframe.NYCflight()
    print('Dataset download----------DONE')

    X_train, X_test, y_train, y_test = NYDataframe.train_test(pd_df, test_size, random_state)

    ## Random Forest and GridSearchCV
    empty_df = pd.DataFrame(columns=['R^2','MSE', 'Times'], index=['RF_SKLearn', 'DASK_RF', 'DASK_XGBoost', 'XGBoostGCV'])
    time_start = time.time()
    param_grid = param_grid_rf
    r_2, mse = NYDataframe.RandomForestSkLearn(param_grid, X_train, X_test, y_train, y_test)
    time_result = time.time() - time_start
    empty_df.iloc[0] = r_2, mse, time_result 
    print('RandomForestSKlearn ---------- DONE')
    
    time_start = time.time()
    r_2, mse = NYDataframe.RandomForestDask(param_grid, X_train, X_test, y_train, y_test)
    time_result = time.time() - time_start
    empty_df.iloc[1] = r_2, mse, time_result 
    print('RandomForestDASK-ML ---------- DONE')

    #XGBooster and GridSearchCV
    time_start = time.time()
    param_grid = param_grid_xgb
    mse = MyXGBooster.daskbooster(X_train, X_test, y_train, y_test)
    time_result = time.time() - time_start
    empty_df.iloc[2] = 0., mse, time_result 
    print('XGBooster-DASK-ML ---------- DONE')  
    
    
    time_start = time.time()
    r_2, mse = MyXGBooster.skbooster(param_grid, X_train, X_test, y_train, y_test)
    time_result = time.time() - time_start
    empty_df.iloc[3] = r_2, mse, time_result 
    empty_df = empty_df.round(2)
    pd.DataFrame.to_csv(empty_df, result_csv, sep=',')
    #save to csv
if __name__ == '__main__':
    # freeze_support() here if program needs to be frozen
    main()  