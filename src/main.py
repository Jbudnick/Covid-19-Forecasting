'''
See notebooks/EDA.ipynb for plots
'''
import pandas as pd
import numpy as np
import datetime

from scipy.interpolate import make_interp_spline

import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.basemap import Basemap
matplotlib.rcParams.update({'font.size': 16})
plt.style.use('fivethirtyeight')
plt.close('all')

#Define minimum threshold of cases per 1 million people in each state to begin training data on.
#Threshold is the minimum value at which data is output; Used to reduce misleading predictions
#(low new cases count and low social distancing parameters before pandemic)
threshold = 100

'''
Import scripts from other .py files
'''
from reg_model_class import reg_model
from data_clean_script import clean_data, replace_initial_values, replace_with_moving_averages, load_and_clean_data, create_spline, convert_to_date


def state_plot(state, df):
    fig, axes = plt.subplots(8, 1, figsize=(12, 15))
    for i, ax in enumerate(axes, 2):
        query = df[df['state'] == state]['days_elapsed']
        x = query.values
        y = covid_df.loc[query.index].iloc[:, i]
        ax.plot(x, y)
    fig.show()


def series_to_supervised(data, columns, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('%s(t-%d)' % (columns[j], i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('%s(t)' % (columns[j])) for j in range(n_vars)]
        else:
            names += [('%s(t+%d)' % (columns[j], i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def generate_prediction_df(level, test_df, test_targs, columns, predictions=20):
    '''
    This function generates a prediction array based on the level of social distancing. The predetermined sets
    are determined as follows: High: Minimum values for each column (except residential),
    # Medium: inbetween values, Low: 1 (Back to Normal)
    '''
    levelDict = {'High': [0.34, 0.5, 0.36, 0.295, 0.4, 1.3, 0.385],
                 'Medium': [0.6, 0.8, 0.7, 0.7, 0.75, 1.1, 0.7],
                 'Low': [1, 1, 1, 1, 1, 0.9, 1]
                 }

    if level not in levelDict.keys():
        pred_params = level
    else:
        pred_params = levelDict[level]

    pred_df = rf_model.X.loc[:, 'days_elapsed(t)':].copy()
    pred_df['moving_average'] = rf_model.y
    pred_df.columns = columns
    last_recorded_day = int(test_df['days_elapsed(t)'].max())
    for i in range(last_recorded_day + 1, last_recorded_day + predictions + 1):
        pred_df_row = pd.DataFrame([i] + pred_params).T
        pred_df_row.columns = columns[:-1]
        pred_df = pred_df.append(pred_df_row, sort=True)
    pred_df = pred_df[columns]
    prediction_ts = series_to_supervised(
        pred_df, columns=columns, n_in=0, n_out=21, dropnan=False)
    prediction_ts = prediction_ts[prediction_ts['days_elapsed(t)'] <= 82]

    #Filling in Nan in former time series fields for moving average with the average values seen - to fix later
    avg_mov_average = prediction_ts.iloc[:6, -1].mean()
    prediction_ts.fillna(avg_mov_average, inplace=True)
    prediction_ts.drop('moving_average(t+20)', axis=1, inplace=True)
    return prediction_ts


if __name__ == '__main__':
    #Load data, select only New York for now
    covid_df = load_and_clean_data()
    mask1 = (covid_df['state'] == 'New York')
    NY_df = covid_df[mask1]
    y = NY_df.pop('New_Cases_per_pop')
    X = NY_df.iloc[:, 1: -1]

    #Calculate moving average, use as target variable instead of raw new cases/pop
    smooth_x, smooth_y = create_spline(X['days_elapsed'], y)
    mov_avg_df = pd.DataFrame([smooth_x, smooth_y]).T
    mov_avg_df.columns = ('days_elapsed', 'moving_average')
    mov_avg_df = mov_avg_df[mov_avg_df['moving_average'] >= threshold]
    revised_df = NY_df.merge(mov_avg_df, on='days_elapsed').iloc[:, 1:]
    revised_df = replace_with_moving_averages(
        revised_df, revised_df.columns[1:-1])

    #Only one state is currently considered in this study, no need to compare pop_density
    revised_df.drop('pop_density', axis=1, inplace=True)

    #Create time series dataframe, fit it into model and evaluate
    values = revised_df.values
    ts_frame_data = series_to_supervised(values, revised_df.columns, 20, 1)
    ts_y = ts_frame_data.pop('moving_average(t)')
    ts_x = ts_frame_data
    rf_model = reg_model(ts_x, ts_y)
    rf_model.rand_forest(n_trees='optimize')
    rf_model.evaluate_model(print_err_metric=True)

    #Plots in notebooks/EDA.ipynb
