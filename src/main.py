'''
See notebooks/EDA.ipynb for plots

Import scripts from other .py files
'''
from src.State_Comparison import Comparable_States, Combined_State_Analysis, state_analysis, Predictions
from src.reg_model_class import reg_model
from src.data_clean_script import clean_data, replace_initial_values, replace_with_moving_averages, load_and_clean_data, create_spline, convert_to_date, fill_na_with_surround, get_moving_avg_df
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

#Define minimum threshold of cases per 1 million people in each state to begin training data on.
#Threshold is the minimum value at which data is output; Used to reduce misleading predictions
#(low new cases count and low social distancing parameters before pandemic)
threshold = 450

def state_plot(state, df):
    fig, axes = plt.subplots(8, 1, figsize=(12, 15))
    for i, ax in enumerate(axes, 2):
        query = df[df['state'] == state]['days_elapsed']
        x = query.values
        y = covid_df.loc[query.index].iloc[:, i]
        ax.plot(x, y)
    fig.show()

if __name__ == '__main__':
    state = 'Minnesota'

    covid_df = load_and_clean_data()
    Similar_States_Init = Comparable_States()
    Similar_States_Init.make_master_pop_dens_df()
    sim_states_df = Similar_States_Init.get_similar_states(
        state_to_predict=state, recovery_factor_min=1.2, pop_density_tolerance=25)
    similar_states = sim_states_df.index.values
    State_Compile = Combined_State_Analysis(similar_states, print_err=True, normalize_day = False)
    State_Compile.get_feature_importances().T
    print("The Most similar states to {} that meet the comparable parameters are: {}. These will be used to predict for {}.".format(
        state, similar_states, state))
    Prediction_Insights = Predictions(covid_df, state, similar_states, State_Compile, normalize = False)
    #Plots in notebooks/EDA.ipynb
