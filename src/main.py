'''
See notebooks/EDA.ipynb for plots

Import scripts from other .py files
'''
from src.State_Comparison import Comparable_States, Combined_State_Analysis, state_analysis, Predictions
from src.reg_model_class import reg_model
from src.data_clean_script import replace_initial_values, replace_with_moving_averages, load_and_clean_data, create_spline, convert_to_date, fill_na_with_surround, convert_to_moving_avg_df
from src.Misc_functions import series_to_supervised, generate_prediction_df, normalize_days

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

def state_plot(states, df):
    '''
    Plots data from a list of states into one figure.
        Parameters:
            states(list of strings)
            df (Pandas DataFrame where information to plot is)
        Returns:
            plot (fig)
    '''
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    for state in states:
        query = df[df['state'] == state]
        x = query['days_elapsed'].apply(convert_to_date).values
        y = query['New_Cases_per_pop'].values
        ax.plot(x, y, label = state)
        ax.set_xlabel('Date')
        ax.set_ylabel('New Daily Cases Per 1M Pop')
        ax.set_title('Daily New Cases Per State Over Time')
        ax.legend()
    fig.show()

if __name__ == '__main__':
    #Specify state to draw predictions for below, and similar state finding parameters
    state = 'North Carolina'
    min_recovery_factor = 1.2
    pop_density_tolerance = 25
    SD_delay = 10
    normalize_days = True

    raw_covid_df = load_and_clean_data(use_internet = True)
    covid_df = convert_to_moving_avg_df(raw_covid_df, SD_delay = SD_delay)
    Similar_States = Comparable_States(covid_df)
    sim_states_df = Similar_States.get_similar_states(
        state_to_predict=state, recovery_factor_min=min_recovery_factor, pop_density_tolerance = pop_density_tolerance)
    similar_states = sim_states_df.index.values
    
    if len(similar_states) == 0:
        print('No similar states found. Try to expand recovery_factor_min and pop_density_tolerance.')
    else:
        print("The Most similar states to {} that meet the comparable parameters are: {}. These will be used to predict for {}.".format(
            state, similar_states, state))
            #Investigate data leakage - test set still has time lagged values
        State_Compile = Combined_State_Analysis(covid_df, similar_states, min_days=0, print_err=True, normalize_day= normalize_days)
        feat_importances = State_Compile.get_feature_importances()
        print(feat_importances)
        Prediction_Insights = Predictions(covid_df, state, similar_states, State_Compile)

    #Plots in notebooks/EDA.ipynb
