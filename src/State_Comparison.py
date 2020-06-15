from src.reg_model_class import reg_model
from src.data_clean_script import replace_initial_values, replace_with_moving_averages, load_and_clean_data, create_spline, convert_to_date, fill_na_with_surround, get_moving_avg_df
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
    To get predictions on a state, similar states in population density  will be needed to compare. Use the
    non-moving average covid_df for import - a moving average will be applied automatically.
    This class generates and stores a dataframe of states, population densities, and Recovery Factor*
    '''

    def __init__(self):
        self.master_pop_density_df = self.make_master_pop_dens_df()

    def make_master_pop_dens_df(self,most_recent_day=104):
        covid_df = load_and_clean_data(use_internet = False)
        all_states = covid_df['state'].unique()
        pop_density = covid_df[['state', 'pop_density']].drop_duplicates()
        pop_density_df = pop_density.set_index('state')
        self.master_covid_df = pd.DataFrame(
            get_moving_avg_df(covid_df, state=all_states[0]))
        self.master_covid_df['state'] = all_states[0]
        for state in all_states[1:]:
            state_df = get_moving_avg_df(covid_df, state=state)
            state_df['state'] = state
            self.master_covid_df = self.master_covid_df.append(state_df)
        max_cases = self.master_covid_df[[
            'state', 'Daily_Cases_per_pop']].groupby('state').max()
        recent_cases = self.master_covid_df[self.master_covid_df['days_elapsed'] == most_recent_day][[
            'state', 'Daily_Cases_per_pop']]  # .groupby('state').max()
        recent_cases['Daily_Cases_per_pop'] = recent_cases['Daily_Cases_per_pop'].where(
            recent_cases['Daily_Cases_per_pop'] > 0.01, 0.01)
        recent_cases.set_index('state', inplace=True)
        recent_cases.drop_duplicates(inplace=True)

        Recovery_df = max_cases / recent_cases
        Recovery_df.rename(
            columns={'Daily_Cases_per_pop': 'Recovery Factor'}, inplace=True)

        master_pop_density_df = pop_density_df.merge(
            Recovery_df, on='state').sort_values('pop_density')
        master_pop_density_df.sort_values('pop_density', inplace=True)
        self.master_pop_density_df = master_pop_density_df
        return self.master_pop_density_df

    def get_similar_states(self, state_to_predict, recovery_factor_min=1.2, pop_density_tolerance=25):
        self.state_to_predict = state_to_predict
        '''
        Recovery Factor is a measure of how well the state has recovered from the pandemic, measured as the greatest
        number of 7 day moving averages of new cases divided by the most recent 7 day moving average.
        Will return states that exceed this number and plus or minus the specified popululation density difference
        to compared state.
        
        Can be called automatically using pre-defined parameters by specifying state when initializing. To specify
        parameters, must be called on objects with them specified after initialization.
        '''
        state_pop_dens = self.master_pop_density_df.loc[state_to_predict, 'pop_density']
        mask1 = self.master_pop_density_df['pop_density'] > state_pop_dens - \
            pop_density_tolerance
        mask2 = self.master_pop_density_df['pop_density'] < state_pop_dens + \
            pop_density_tolerance
        mask3 = self.master_pop_density_df['Recovery Factor'] > recovery_factor_min
        return self.master_pop_density_df[mask1 & mask2 & mask3]


class Combined_State_Analysis(reg_model):
    '''
    Provide list of state_dfs to combined into one model to create a dataset for training. Use the Comparable_States
    class to generate similar states for better results before calling predictions class.
    '''

    def __init__(self, state_list, print_err=False, normalize_day = True):
        '''
        normalize_day is currently broken

        '''
        register_matplotlib_converters()
        covid_df = load_and_clean_data(use_internet = False)
        if normalize_day == True:
            state_dfs = normalize_days(state_list, covid_df)
            new_df = state_dfs[0].copy()
            if len(state_dfs) > 0:
                for each_df in state_dfs[1:]:
                    new_df = new_df.append(each_df)
            covid_df = new_df
            covid_df.rename(columns = {'days_since_start': 'days_elapsed'}, inplace = True)
        self.state_list = state_list
        X_df_list = [state_analysis(covid_df, state=state, print_err=False, mov_avg = True)[
            1] for state in state_list]
        y_df_list = [state_analysis(covid_df, state=state, print_err=False, mov_avg = True)[
            2] for state in state_list]
        if len(X_df_list) == 1:
            self.X = X_df_list[0]
            self.y = y_df_list[0]
        else:
            try:
                self.X = X_df_list[0].append(X_df_list[1:])
            except:
                breakpoint()
            self.y = y_df_list[0].append(y_df_list[1:])
        self.rf = reg_model(self.X, self.y)
        self.rf.rand_forest()
        self.evaluate = self.rf.evaluate_model(print_err_metric=print_err)

    def print_err(self, print_err):
        self.rf.evaluate_model(print_err=True)

    def get_feature_importances(self):
        features = self.rf.X.columns
        feature_importances = self.rf.model.feature_importances_
        return pd.DataFrame(feature_importances, index=features)


class Predictions(Combined_State_Analysis):
    '''
    Use results from Comparable_States and Combined_State_Analysis to come up with predictions for state
    '''

    def __init__(self, covid_df, state_to_predict, similar_states, Comb_St_Analysis, normalize = False):
        self.state = state_to_predict
        self.similar_states = similar_states
        self.State_Compile = Comb_St_Analysis
        self.similar_df = Comb_St_Analysis.X.copy()
        self.similar_df['New_Cases_per_pop(t)'] = Comb_St_Analysis.y

        self.pop_densities = self.similar_df['pop_density(t)'].unique()
        self.State_Analysis_X, self.State_Analysis_y = state_analysis(
            covid_df, state=state_to_predict, print_err=False, normalize_day=False)[1], state_analysis(
            covid_df, state=state_to_predict, print_err=False, normalize_day=False)[2]

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
        ax.axvline(convert_to_date(93), linestyle='-.', lw='0.7',
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
        ax.plot(self.State_Analysis_X['days_elapsed(t)'].apply(
            convert_to_date), self.State_Compile.rf.model.predict(self.State_Analysis_X), label='Model Predictions', c = 'black', ls = '--')
        ax.plot(self.State_Analysis_X['days_elapsed(t)'].apply(
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

    def forecast_to_future(self, save=None):
        min_SD, max_SD = self.get_social_distancing_estimates()
        high_pred = generate_prediction_df(
            max_SD, self.State_Analysis_X, self.State_Analysis_y, predictions=21, rf=self.State_Compile.rf)
        fig, ax = plt.subplots(figsize=(14, 7))
        labels = ['High Social Distancing', 'Low Social Distancing']
        x = high_pred[0]['days_elapsed(t)']
        y = high_pred[1]
        x = x[2:]
        ax.plot(x[x <= 103].apply(convert_to_date), y[:len(x[x <= 103])], label = 'Past Data', c = 'black')
        ax.plot(x[x >= 103].apply(convert_to_date), y[-len(x[x >= 103]):], label= 'Low Public Activity', c = 'lime', ls = '-.')

        low_pred = generate_prediction_df(
            min_SD, self.State_Analysis_X, self.State_Analysis_y, predictions=21, rf=self.State_Compile.rf)
        x = low_pred[0]['days_elapsed(t)']
        y = low_pred[1]
        ax.plot(x[x >= 103].apply(convert_to_date),
                y[-len(x[x >= 103]):], label='High Public Activity', c = 'tomato', ls = '-.')

        ax.legend()
        ax.set_title('Future Predicted Daily New Cases'.format(self.state))
        ax.set_xlabel('Date')
        ax.set_ylabel('New Cases/Day Per 1M Pop')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(7))
        fig.autofmt_xdate(rotation=30)
        fig.tight_layout()
        if save != None:
            fig.savefig(save, dpi = 300)

def state_analysis(covid_df, state='New York', print_err=False, normalize_day = False, mov_avg = True):
    '''
    Produces random forest model for specified state, returns tuple of model and time series dataframe
    '''
    if normalize_day == True:
        state_dfs = normalize_days([state], covid_df)
        new_df = state_dfs[0]
        if len(state_dfs) > 1:
            new_df.append([each_df for each_df in state_dfs[1:]])
        covid_df = new_df
        covid_df.rename(
            columns={'days_since_start': 'days_elapsed'}, inplace=True)
    if mov_avg == True:
        revised_df = get_moving_avg_df(covid_df, state=state)
    else:
        revised_df = covid_df.copy()

    #Create time series dataframe, fit it into model and evaluate
    values = revised_df.values
    num_cols = len(revised_df.columns)
    ts_frame_data = series_to_supervised(values, revised_df.columns, 21, 1)
    ts_frame_data = ts_frame_data.iloc[:,
                                        num_cols-1:-num_cols + 1:num_cols].join(ts_frame_data.iloc[:, -num_cols:])
    ts_frame_data.index.name = state
    try:
        ts_y = ts_frame_data.pop('New_Cases_per_pop(t)')
    except:
        ts_y = ts_frame_data.pop('Daily_Cases_per_pop(t)')
    ts_x = ts_frame_data
    if 'state(t)' in ts_x.columns:
        ts_x.drop('state(t)', axis = 1, inplace = True)

    #If there are any negative values in elapsed, it has been normalized.
    if ts_x[ts_x['days_elapsed(t)'] <0].shape[0] > 0:
        rf_model = reg_model(ts_x, ts_y, day_cutoff='auto')
    else:
        rf_model = reg_model(ts_x, ts_y)
    rf_model.rand_forest(n_trees=100)
    rf_model.evaluate_model(print_err_metric=print_err)
    return rf_model, ts_x, ts_y
