# Parallel Data Analysis with Dask with NYCflight dataset

To repeat results use `ZHORES` with python3, 
In terminal press `module add python/anaconda3`
and `sh run.sh`

Here we compare results of:
* RandomForest SkLearn GridSearchCV
* RandomForest Dask-ml GridSearchCV
* XGBoost SkLearn GridSearchCV
* XGBoost Dask-ml 

It is possible to change parameters in `parameters.json` for:
* GridSearchCV HyperParameters search
* URL's of Dataset
* Number of rows for `csv` files
* Setup Working Directory

It is possible to use flags with `main.py` file:
* `--test_size`
* `random_state`
* `--results_path` to csv file
* `--parameters` path to `JSON` file

|model|R^2|MSE|Time(s)|
|---|---|---|---|
|Random Forest SKleran GS| 0.71  | 245.1  |  35.81 |
| Random Forest DASK-ml GS  |0.71   |  244.9 | 43.82  |
|  DASK-ml XGBoost |  NaN | 524.8  | 28.6  |
|  XGBoost GridSearchCV |  0.97 |  23.4 | 694.12  |
