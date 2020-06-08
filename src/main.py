'''
See notebooks/EDA.ipynb for plots

Import scripts from other .py files
'''
from src.State_Comparison import Comparable_States, Combined_State_Analysis, state_analysis, Predictions
from src.reg_model_class import reg_model
from src.data_clean_script import clean_data, replace_initial_values, replace_with_moving_averages, load_and_clean_data, create_spline, convert_to_date, fill_na_with_surround, get_moving_avg_df
from Misc_functions import series_to_supervised

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

def fill_diagonals(df, preds, model, start_row=31, n_interval=21):
    df.fillna(0, inplace=True)
    n_rows = df.shape[0]
    new_preds = list(preds.values)
    for row in range(start_row, n_rows)[:]:
        new_pred = model.predict(df[row:row + 1])[0]
        new_preds.append(new_pred)
        j = 0
        for col in range(n_interval-1, 0, -1):
            try:
                if df.iloc[row + j, col] == 0:
                    df.iloc[row + j, col] = new_pred
                j += 1
            except:
                continue
    new_pred = model.predict(df[-1:-2:-1])[0]
    new_preds.append(new_pred)
    return df, new_preds


def generate_prediction_df(level, total_x, total_y, rf, predictions=21):

    #Part 1: Expands time lagged Daily New Cases columns

    columns = ['days_elapsed(t)', 'retail_and_recreation(t)', 'grocery_and_pharmacy(t)',
               'parks(t)', 'transit_stations(t)', 'workplaces(t)', 'residential(t)', 'driving(t)', 'pop_density(t)']

    levelDict = {'High': [0.34, 0.5, 0.36, 0.295, 0.4, 1.3, 0.385],
                 'Medium': [0.6, 0.8, 0.7, 0.7, 0.75, 1.1, 0.7],
                 'Low': [1, 1, 1, 1, 1, 0.9, 1]
                 }
    if type(level) != str:
        pred_params = level
    else:
        pred_params = levelDict[level]
    pred_df = total_x.copy()
    last_recorded_day = int(pred_df['days_elapsed(t)'].max())
    pop_dens = pred_df['pop_density(t)'].mode().iloc[0]
    for i in range(last_recorded_day + 1, last_recorded_day + predictions + 1):
        pred_df_row = pd.DataFrame([i] + pred_params + [pop_dens]).T
        pred_df_row.columns = columns
        pred_df = pred_df.append(pred_df_row, sort=False)

    y_pred = total_y

    # Part 2: Fills in blank known new cases values
    n_rows = pred_df.shape[0]
    pred_df.fillna(0, inplace=True)
    row_start = pred_df.shape[0] - \
        pred_df[pred_df['Daily New Cases(t-1)'] == 0].count()[0]
    col_start = 20
    new_preds = list(y_pred.values)
    pred_df.iloc[row_start, col_start] = y_pred.values[-1]
    for row in range(row_start, n_rows):
        for col in range(col_start - 1, -1, -1):
            pred_df.iloc[row, col] = pred_df.iloc[row - 1, col + 1]

    #Part 3: Fills in rest of time lagged values for future t values, predicting based on prior predictions
    fill_diag_and_predictions = fill_diagonals(
        pred_df, y_pred.loc[:45], rf.model, start_row=row_start, n_interval=21)
    pred_df = fill_diag_and_predictions[0]
    pred_y = fill_diag_and_predictions[1][-pred_df.shape[0]:]
    return pred_df, pred_y

if __name__ == '__main__':
    state = 'Minnesota'

    covid_df = load_and_clean_data()
    Similar_States_Init = Comparable_States()
    Similar_States_Init.make_master_pop_dens_df()
    sim_states_df = Similar_States_Init.get_similar_states(
        state_to_predict=state, recovery_factor_min=1.2, pop_density_tolerance=25)
    similar_states = sim_states_df.index.values
    State_Compile = Combined_State_Analysis(similar_states, print_err=True)
    State_Compile.get_feature_importances().T
    print("The Most similar states to {} that meet the comparable parameters are: {}. These will be used to predict for {}.".format(
        state, similar_states, state))

    Prediction_Insights = Predictions(covid_df, state, similar_states, State_Compile)
    #Plots in notebooks/EDA.ipynb
