import numpy as np
import pandas as pd

import datetime

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from scipy.interpolate import make_interp_spline
from src.data_clean_script import replace_initial_values, replace_with_moving_averages, load_and_clean_data, create_spline, convert_to_date, fill_na_with_surround
from src.Misc_functions import fill_blank_known_ts, populate_predictions

from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt
import matplotlib

from matplotlib.dates import (DAILY, DateFormatter,
                              rrulewrapper, RRuleLocator, drange)

class reg_model(object):
    def __init__(self, X, y, train_test_split=0.75, ts = False, normalized = False):
        '''
            Parameters:
                X (Pandas DataFrame): Data to be used for regression model - before train/test split
                y (Series): Target values for X
                train_test_split (int/float): Days elapsed value to separate dataset into train/testing set. If int, will use days_elapsed number. If float, will separate based on percentage (0.80 = 80% of data will go into training, 20% will be testing)
        '''
        self.X = X
        self.y = y
        if type(train_test_split) is float:
            days = X['days_elapsed(t)']
            train_test_split = int(((days.max() - days.min()) * train_test_split) + days.min())
        elif train_test_split >= self.X['days_elapsed(t)'].max():
            train_test_split = self.X['days_elapsed(t)'].max() - 10
        if type(normalized) is float:
            print('Train/Test Split on day', train_test_split, 'after state reached',
                  normalized * 100, "% of maximum daily new cases.")
        else:
            print('Train/Test Split on', convert_to_date(train_test_split))
        self.train_test_split = train_test_split
        train_mask = self.X['days_elapsed(t)'] < train_test_split
        holdout_mask = self.X['days_elapsed(t)'] >= train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = self.X[
            train_mask], self.X[holdout_mask].copy(), self.y[train_mask], self.y[holdout_mask]
        self.error_metric = None

    def rand_forest(self, n_trees=100):
        '''
        Applies random forest to reg_model object.
        '''
        if n_trees == 'optimize':
            '''
            If set to optimize, will take a selection of 1 to max_trees and uses number that minimizes error in training set.
            This can be plotted by uncommenting out the plt.plot(n, error) line.
            '''
            max_trees = 100
            n = np.arange(1, max_trees + 1, 1)
            error = []
            for each in n:
                self.model = RandomForestRegressor(
                    n_estimators=each, n_jobs=-1, random_state=1)
                self.model.fit(self.X_train, self.y_train)
                self.error_metric = 'rmse'
                error.append(self.evaluate_model())
            #plt.plot(n, error)
            n_trees = n[error.index(min(error))]
        self.model = RandomForestRegressor(
            n_estimators=n_trees, random_state=None)
        self.model.fit(self.X_train, self.y_train)
        self.error_metric = 'rmse'

        #Removes leakage from test set
        new_state_idx = self.X_test['pop_density(t)'].drop_duplicates(
        ).index.values
        min_idx = new_state_idx[0] + 1
        for i in new_state_idx[1:]:
            self.X_test.loc[min_idx: i - 1, 'New_Cases_per_pop(t-21)': 'New_Cases_per_pop(t-1)'] = 0
            fill_blank_known_ts(self.X_test, None, min_idx, row_end=i)
            populate_predictions(self.X_test, preds=pd.DataFrame(), model=self.model, start_row=min_idx - 1, end_row=i)[0]
            min_idx = i + 1
        self.X_test.loc[min_idx:, 'New_Cases_per_pop(t-21)': 'New_Cases_per_pop(t-1)'] = 0
        fill_blank_known_ts(self.X_test, None, min_idx, row_end= 'all')
        populate_predictions(self.X_test, preds=pd.DataFrame(), model=self.model, start_row=min_idx - 1, end_row= 'all')[0]

    def evaluate_model(self, print_err_metric=False):
        '''
        Determine validity of model on test set.
        '''
        self.y_hat = self.model.predict(self.X_test)
        self.predicted_vals_df = pd.DataFrame(self.y_test)
        self.predicted_vals_df['days_elapsed(t)'] = self.X_test['days_elapsed(t)']
        self.predicted_vals_df['y_hat'] = self.y_hat
        self.predicted_vals_df.sort_index(inplace=True)
        if self.error_metric == 'rmse':
            rmse = np.sqrt(mean_squared_error(self.y_test, self.y_hat))
            if print_err_metric:
                print('rmse:', rmse)
            return rmse
        elif self.error_metric == 'rss':
            rss = np.mean((self.y_test - self.y_hat)**2)
            if print_err_metric:
                print('rss: ', rss)
            return rss

    def plot_model(self, threshold=100, save_name=None, xvar='days_elapsed(t)', convDate=True):
        '''
        Use smoothed generates data using moving average. 
        Convdate converts days elapsed into date
        '''
        register_matplotlib_converters()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(self.X_test.loc[:, xvar].apply(convert_to_date),
               self.y_test, color='blue', label="Test Data")
        ax.bar(self.X_train.loc[:, xvar].apply(convert_to_date),
               self.y_train, color='red', label="Training Data")
        x = pd.DataFrame(self.X[xvar]).iloc[:, 0].apply(convert_to_date)
        y = self.y
        ax.plot_date(x, y, c='green', label='Moving Average - 7 days',
                        xdate=True, marker='', ls='-')
        fig.autofmt_xdate()

        rule = rrulewrapper(DAILY, interval=7)
        loc = RRuleLocator(rule)
        formatter = DateFormatter('%y/%m/%d')
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_tick_params(rotation=30, labelsize=14)
        ax.set_xlabel('Date')
        try:
            x_thresh = convert_to_date(self.train_test_split)
            ax.axvline(x_thresh, label='Threshold', color='black', ls='--')
        except:
            pass
        ax.legend()
        ax.set_ylabel('Daily Cases per 1 Million Population')
        ax.set_title('COVID-19 New Cases (Training Set')
        fig.tight_layout()
        if save_name != None:
            fig.savefig('images/{}'.format(save_name), dpi = 300)
