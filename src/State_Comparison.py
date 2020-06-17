from src.reg_model_class import reg_model
from src.data_clean_script import replace_initial_values, replace_with_moving_averages, load_and_clean_data, create_spline, convert_to_date, fill_na_with_surround, convert_to_moving_avg_df
from src.Misc_functions import series_to_supervised, generate_prediction_df, normalize_days

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas.plotting import register_matplotlib_converters

import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker

class Comparable_States(object):
    '''
    To get predictions on a state, similar states in population density will be needed to compare. 
    This class generates and stores a dataframe of states, population densities, and Recovery Factor*
    
    *Recovery Factor is a measure of how well the state has recovered from the pandemic, measured as the greatest
    number of 7 day moving averages of new cases divided by the most recent 7 day moving average.
    Will return states that exceed this number and plus or minus the specified popululation density difference
    to compared state.
    '''

    def __init__(self, covid_df):
        self.covid_df = covid_df
        self.master_pop_density_df = self.make_master_pop_dens_df()

    def make_master_pop_dens_df(self):
        '''
        Makes a dataframe of population densities and recovery factors
        '''
        all_states = self.covid_df['state'].unique()
        pop_density = self.covid_df[['state', 'pop_density']]
        pop_density_df = pop_density.set_index('state')
        pop_density_df.loc[:, 'pop_density'] = pop_density_df.apply(lambda x: round(x, 5))
        pop_density_df.drop_duplicates(inplace = True)
        
        max_cases = self.covid_df[['state', 'New_Cases_per_pop']].groupby('state').max()
        max_cases.drop_duplicates(inplace = True)

        most_recent_day = self.covid_df['days_elapsed'].max()
        recent_cases = self.covid_df[self.covid_df['days_elapsed'] == most_recent_day][['state', 'New_Cases_per_pop']]
        #If any state has less than 0.01 recent cases, replaces with 0.01 so a recovery factor will be generated
        recent_cases['New_Cases_per_pop'] = recent_cases['New_Cases_per_pop'].where(recent_cases['New_Cases_per_pop'] > 0.01, 0.01)
        recent_cases.set_index('state', inplace=True)
        recent_cases.drop_duplicates(inplace=True)

        Recovery_df = max_cases / recent_cases
        Recovery_df.rename(columns={'New_Cases_per_pop': 'Recovery Factor'}, inplace=True)

        self.master_pop_density_df = pop_density_df.merge(Recovery_df, on='state').sort_values('pop_density')
        self.master_pop_density_df.sort_values('pop_density', inplace=True)
        return self.master_pop_density_df

    def get_similar_states(self, state_to_predict, recovery_factor_min=1.2, pop_density_tolerance=25):
        self.state_to_predict = state_to_predict
        '''
        Gets states similar in population densities to state_to_predict

            Parameters:
                state_to_predict (str): Specify state to focus on for getting predictions
                recovery_factor_min (float): Minimum value of recovery to retrieve data for training set
                pop_density_tolerance (int): Include states that meet recovery_factor_min and are plus or minus this number in population density
            Returns:
                DataFrame of states that meet the specified recovery_factor_min and pop_density_tolerance
        '''
        state_pop_dens = self.master_pop_density_df.loc[state_to_predict, 'pop_density']
        mask1 = self.master_pop_density_df['pop_density'] > state_pop_dens - \
            pop_density_tolerance
        mask2 = self.master_pop_density_df['pop_density'] < state_pop_dens + \
            pop_density_tolerance
        mask3 = self.master_pop_density_df['Recovery Factor'] > recovery_factor_min
        return self.master_pop_density_df[mask1 & mask2 & mask3]


def state_analysis(covid_df, state, create_indiv_rf=False, print_err=False, normalize_day=False):
    '''
    Generates time series DataFrame for state
        Parameters:
            covid_df (Pandas DataFrame)
            state (str): State to analyze
            create_indiv_rf (bool): True to create rf model for individual state
            print_err (bool): for use with the prior parameter, show error metric of rf
            normalize_day (bool): Currently broken
        Returns:
            ts_x (Pandas DataFrame): Time series of x values for specified state
            ts_y (Pandas DataFrame): Time series of y values for specified state
    '''
    #Create time series dataframe, fit it into model and evaluate
    days_to_lag = 21
    state_ts_df = covid_df[covid_df['state'] == state].copy()
    values = state_ts_df.values
    num_cols = len(state_ts_df.columns)
    ts_frame_data = series_to_supervised(values, state_ts_df.columns, days_to_lag, 1)
    ts_frame_data = ts_frame_data.iloc[:,
                                       num_cols-1:-num_cols + 1:num_cols].join(ts_frame_data.iloc[:, -num_cols:])
    ts_frame_data.index.name = state
    ts_y = ts_frame_data.pop('New_Cases_per_pop(t)')
    ts_x = ts_frame_data
    if create_indiv_rf == True:
        ts_x_rf = ts_x.drop('state(t)', axis=1)
        rf_model = reg_model(ts_x_rf, ts_y)
        rf_model.rand_forest(n_trees=100)
        rf_model.evaluate_model(print_err_metric=print_err)
        return ts_x, ts_y, rf_model
    else:
        return ts_x, ts_y


class Combined_State_Analysis(reg_model):
    '''
    Provide list of state_dfs to combined into one model to create a dataset for training. Use the Comparable_States
    class to generate similar states for better results before calling predictions class.
    '''

    def __init__(self, covid_df, state_list, min_days = 45, train_test_split = 0.8, print_err=False, normalize_day=False):
        '''
            Parameters:
                covid_df (Pandas DataFrame): dataset with moving average, etc 
                state_list (list): List of similar states
                min_days (int): Filter out days before pandemic
                train_test_split (float/int): Split between train and test data
                print_err (bool): To print error metric
                normalize_day (bool): (currently broken - to fix later)
        '''
        register_matplotlib_converters()
        self.covid_df = covid_df
        self.state_list = state_list
        X_df_list = [state_analysis(self.covid_df, state=state)[0] for state in state_list]
        y_df_list = [state_analysis(self.covid_df, state=state)[1] for state in state_list]
        if len(X_df_list) == 1:
            self.X = X_df_list[0]
            self.y = y_df_list[0]
        else:
            self.X = X_df_list[0].append(X_df_list[1:])
            self.y = y_df_list[0].append(y_df_list[1:])
        self.X_rf = self.X[self.X['days_elapsed(t)'] >= min_days]
        self.y_rf = self.y[self.X['days_elapsed(t)'] >= min_days]
        X_rf = self.X_rf.drop('state(t)', axis = 1)
        self.rf = reg_model(X_rf, self.y_rf, train_test_split)
        self.rf.rand_forest()
        self.evaluate = self.rf.evaluate_model(print_err_metric=print_err)

    def print_err(self, print_err):
        self.rf.evaluate_model(print_err=True)

    def get_feature_importances(self, exclude_time_lag = True):
        features = self.rf.X.columns
        feature_importances = self.rf.model.feature_importances_
        feature_importances = pd.DataFrame(feature_importances, features)
        if exclude_time_lag == True:
            feat_imp_df = feature_importances.loc['days_elapsed(t)':, :]
        else:
            feat_imp_df = feature_importances
        return feat_imp_df.sort_values(0, ascending = False)

class Predictions(Combined_State_Analysis):
    '''
    Use results from Comparable_States and Combined_State_Analysis to come up with predictions for state
    '''

    def __init__(self, covid_df, state_to_predict, similar_states, Comb_St_Analysis):
        '''
            Parameters:
                covid_df (Pandas DataFrame)
                state_to_predict (str): State focus for predictions
                similar_states (list): List of states similar in population density to draw insight from
                Comb_St_Analysis (Combined_States_Analysis object): Object generated using similar states and regression model
            Returns:
                Most methods generate predictions in the form of plots
        '''
        self.state = state_to_predict
        self.similar_states = similar_states
        self.State_Compile = Comb_St_Analysis
        self.similar_df = Comb_St_Analysis.X.copy()
        self.similar_df['New_Cases_per_pop(t)'] = Comb_St_Analysis.y

        self.pop_densities = self.similar_df['pop_density(t)'].unique()
        self.State_Analysis_X, self.State_Analysis_y = state_analysis(
            covid_df, state=state_to_predict, print_err=False, normalize_day=False)[0], state_analysis(
            covid_df, state=state_to_predict, print_err=False, normalize_day=False)[1]

    def get_social_distancing_estimates(self, analysis=False):
        '''
        This method gets the minimum and maximum social distancing levels for all states in the training set based
        on the maximum and minimum amounts observed on the training interval.
        '''
        min_vals = self.similar_df.min(
        ).loc['retail_and_recreation(t)':'driving(t)']
        max_vals = self.similar_df.max(
        ).loc['retail_and_recreation(t)':'driving(t)']
        max_SD = list(min_vals[:5])
        max_SD.extend([max_vals[5], min_vals[6]])
        min_SD = list(max_vals[:5])
        min_SD.extend([min_vals[5], max_vals[6]])
        if analysis == False:
            return min_SD, max_SD
        if analysis == True:
            high, low = max_SD, min_SD
            columns = ['Retail/Recreation %', 'Grocery/Pharmacy %', 'Parks %',
                       'Transit Stations %', 'Workplaces %', 'Residential %', 'Driving %']
            SD_Table = round(pd.DataFrame(
                [np.array(high), np.array(low)], columns=columns) * 100, 2)
            SD_Table[''] = ['High', 'Low']
            SD_Table.set_index('', inplace=True)
            return SD_Table

    def plot_similar_states(self, save=None):
        fig, ax = plt.subplots(figsize=(14, 7))
        x = self.State_Analysis_X['days_elapsed(t)'].apply(convert_to_date)
        y = self.State_Analysis_y.values
        ax.plot(x.values, y, label=self.state, ls='--', c = 'steelblue')
        for i, pop_density in enumerate(self.pop_densities):
            state_df = self.similar_df[self.similar_df['pop_density(t)']
                                       == pop_density]
            x = state_df.loc[:, 'days_elapsed(t)']
            y = state_df.loc[:, 'New_Cases_per_pop(t)']
            ax.plot(x.apply(convert_to_date), y,
                    label=self.similar_states[i])
        ax.axvline(convert_to_date(self.State_Compile.rf.train_test_split), linestyle='-.', lw='0.7',
                color='black', label='Train/Test Split')
        ax.legend()
        ax.set_title(
            'States Similar to {} in Population Density'.format(self.state))
        ax.set_xlabel('Date')
        ax.set_ylabel('New Cases/Day Per 1M Pop')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(7))
        fig.autofmt_xdate(rotation=30)
        fig.tight_layout()
        if save != None:
            fig.savefig(save, dpi = 300)

    def plot_pred_vs_actual(self, save=None):
        fig, ax = plt.subplots(figsize=(14, 7))
        State_Analysis_X = self.State_Analysis_X.drop('state(t)', axis = 1)
        ax.plot(State_Analysis_X['days_elapsed(t)'].apply(
            convert_to_date), self.State_Compile.rf.model.predict(State_Analysis_X), label='Model Predictions', c = 'black', ls = '--')
        ax.plot(State_Analysis_X['days_elapsed(t)'].apply(
            convert_to_date), self.State_Analysis_y.values, label='Actually  Observed', c = 'steelblue')
        ax.set_ylim(0)
        ax.legend()
        ax.set_title('Model Performance for {}'.format(self.state))
        ax.set_xlabel('Date')
        ax.set_ylabel('New Cases/Day Per 1M Pop')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(7))
        fig.autofmt_xdate(rotation=30)
        fig.tight_layout()
        if save != None:
            fig.savefig(save, dpi = 300)

    def forecast_to_future(self, SD_delay = 10, save=None):
        '''
        SD_delay(int): If social distancing parameters were set using a delayed moving average, the level of delay should be entered here 
        as well, so the prediction matrix is generated with it as well.
        '''
        min_SD, max_SD = self.get_social_distancing_estimates()
        high_pred = generate_prediction_df(
            max_SD, self.State_Analysis_X, self.State_Analysis_y, predictions=21, rf=self.State_Compile.rf)
        fig, ax = plt.subplots(figsize=(14, 7))
        x = high_pred[0]['days_elapsed(t)']
        y = high_pred[1]
        most_recent_day = self.State_Analysis_X['days_elapsed(t)'].max()
        ax.plot(x[x < most_recent_day].apply(convert_to_date),
                y[:len(x[x < most_recent_day])], label='Past Data', c='black')
        ax.plot(x[x >= most_recent_day - 1].apply(convert_to_date),
                y[-len(x[x >= most_recent_day - 1]):], label='Low Public Activity', c='lime', ls='-.')

        low_pred = generate_prediction_df(
            min_SD, self.State_Analysis_X, self.State_Analysis_y, SD_delay = SD_delay, predictions=21, rf=self.State_Compile.rf)
        x = low_pred[0]['days_elapsed(t)']
        y = low_pred[1]
        ax.plot(x[x >= most_recent_day].apply(convert_to_date),
                y[-len(x[x >= most_recent_day]):], label='High Public Activity', c='tomato', ls='-.')
        ax.legend()
        ax.set_title('Future Predicted Daily New Cases for {}'.format(self.state))
        ax.set_xlabel('Date')
        ax.set_ylabel('New Cases/Day Per 1M Pop')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(7))
        fig.autofmt_xdate(rotation=30)
        fig.tight_layout()
        plt.show()
        if save != None:
            fig.savefig(save, dpi = 300)
