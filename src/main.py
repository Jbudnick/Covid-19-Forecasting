'''
See notebooks/EDA.ipynb for plots

Import scripts from other .py files
'''
from pandas.plotting import scatter_matrix
from src.State_Comparison import Comparable_States, Combined_State_Analysis, state_analysis, Predictions
from src.reg_model_class import reg_model
from src.data_clean_script import replace_initial_values, replace_with_moving_averages, load_and_clean_data, create_spline, convert_to_date, fill_na_with_surround, convert_to_moving_avg_df
from src.Misc_functions import series_to_supervised, generate_prediction_df, normalize_days, plot_normalized

import pandas as pd
import numpy as np
import datetime
from sklearn.inspection import plot_partial_dependence
from pycebox.ice import ice, ice_plot
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

def plot_part_dep(State_Compile, use_ice = False):
    state_to_predict = State_Compile.state_to_predict_analysis['state(t)'].unique()[
        0]
    state_to_predict = state_to_predict.replace(' ', '_')

    X = State_Compile.X_norm.drop('state(t)', axis=1)
    X.rename(columns={'parks(t)': 'Public Activity in Parks',
                      'residential(t)': 'Activity At Home'}, inplace=True)

    if use_ice == True:
        fig, axes = plt.subplots(3,3, figsize = (40,40))
        y = State_Compile.y_norm
        cols = X.columns
        cols_interest_ind = X.columns.get_loc('days_elapsed(t)')
        columns_of_interest = X.columns[cols_interest_ind:]
        for i, ax in enumerate(axes.flatten()):
            ice_df = ice(X, columns_of_interest[i], State_Compile.rf.model.predict)
            ice_plot(ice_df, alpha = 0.25, ax = ax)
            ax.set_title('ICE Curve')
            ax.set_xlabel(columns_of_interest[i])
            ax.set_ylabel('Esimated Daily Number of Cases/1M Pop')
        fig.savefig('images/{}ICE'.format(state_to_predict), dpi=300)

    else:
        X = X.loc[:, 'days_elapsed(t)':]
        fig = plot_partial_dependence(State_Compile.rf.model, X, X.columns)
        fig = plt.gcf()
        axes = fig.get_axes()
        for ax in axes:
            ax.set_ylabel('Estimate of Daily New Cases/1M Pop')
        fig.set_figwidth(30)
        fig.set_figheight(30)
        fig.savefig('images/{}part_dep'.format(state_to_predict), dpi=300)

def plot_feature_importances(State_Compile):
    fig, ax = plt.subplots(figsize=(14, 7))
    state_to_predict = State_Compile.state_to_predict_analysis['state(t)'].unique()[
        0]
    state_to_predict = state_to_predict.replace(' ', '_')
    feat_importance = pd.DataFrame(
        State_Compile.get_feature_importances().T.iloc[:, -9:].values.flatten()).T
    cols = State_Compile.get_feature_importances(
    ).T.iloc[:, -9:].columns.values
    feat_importance.columns = [
        x.replace('_', ' ').replace('(t)', '').title() for x in cols]

    colors = ['blue', 'blue', 'blue', 'blue',
              'blue', 'grey', 'grey', 'blue', 'grey']
    cols = ['Parks', 'Workplaces', 'Retail And Recreation', 'Residential',
            'Grocery And Pharmacy', 'Pop Density', 'Driving', 'Transit Stations',
            'Days Elapsed']
    colorDict = {col: color for col, color in zip(cols, colors)}
    feat_importance.sort_values(by=0, axis=1, ascending=False, inplace=True)
    colors = [colorDict[each] for each in feat_importance]
    labels = ['Social Distancing', 'Other']
    ax.bar(feat_importance.columns, feat_importance.values.flatten(),
           color=colors, label='Social Distance')
    ax.set_title('Feature Importances for {}'.format(state_to_predict))
    ax.bar([-1], [0.003], color='grey', label='Other')
    ax.set_xlim(-0.5)
    ax.set_ylabel('Feature Importance')
    fig.autofmt_xdate(rotation=20)
    fig.tight_layout()
    ax.legend()
    fig.savefig('images/{}features'.format(state_to_predict), dpi=300)

def run_model(state, min_recovery_factor=1.2, pop_density_tolerance=20, SD_delay=7, train_test_split=0.4, percent_max_cases=0.25, test_row_start=35, scatter=False):
    #Specify state to draw predictions for below, and similar state finding parameters
    #Similar_States.master_pop_density_df[Similar_States.master_pop_density_df['Recovery Factor'] < 1.1]
    #Define outbreak start (percent_of_max_cases)

    normalize_days = True

    raw_covid_df = load_and_clean_data(use_internet=True)
    covid_df = convert_to_moving_avg_df(raw_covid_df, SD_delay=SD_delay)
    Similar_States = Comparable_States(covid_df)
    sim_states_df = Similar_States.get_similar_states(
        state_to_predict=state, recovery_factor_min=min_recovery_factor, pop_density_tolerance=pop_density_tolerance)
    similar_states = sim_states_df.index.values

    if len(similar_states) == 0:
        print('No similar states found. Try to expand recovery_factor_min and pop_density_tolerance.')
    else:
        print("The Most similar states to {} that meet the comparable parameters are: {}. These will be used to predict for {}.".format(
            state, similar_states, state))
        State_Compile = Combined_State_Analysis(covid_df, state, similar_states, train_test_split=train_test_split,
                                                min_days=0, print_err=True, normalize_day=normalize_days, percent_of_max_cases=percent_max_cases)

        normalized_df = State_Compile.X_norm.copy()
        normalized_df['New_Cases_per_pop'] = State_Compile.y_norm

        feat_importances = State_Compile.get_feature_importances()
        print(feat_importances)
        Prediction_Insights = Predictions(
            covid_df, state, similar_states, State_Compile)
        state_sv = state.replace(' ', '_')
        Prediction_Insights.plot_similar_states(
            save='images/{}similarplots'.format(state_sv))
        plot_normalized(normalized_df, State_Compile,
                        save='images/{}normalized'.format(state_sv))
        Prediction_Insights.plot_pred_vs_actual(
            row_start=35, save='images/{}validity'.format(state_sv))

        plot_feature_importances(State_Compile)
        plot_part_dep(State_Compile)
        plot_part_dep(State_Compile, use_ice=True)

        SD_Table = Prediction_Insights.get_social_distancing_estimates(
            analysis=True)
        print(round(SD_Table, 1))

        Prediction_Insights.forecast_to_future(
            SD_delay=SD_delay, save='images/{}future'.format(state_sv))

        if scatter == True:
            total = State_Compile.X_norm.iloc[:, -9:].copy()
            total['Daily New Cases'] = State_Compile.y_norm
            for each in total.columns:
                total[each] = pd.to_numeric(total[each])
            scatter_matrix(total, figsize=(40, 40))
    #Plots in notebooks/EDA.ipynb


if __name__ == '__main__':
    #Specify state to draw predictions for below, and similar state finding parameters
    #Similar_States.master_pop_density_df[Similar_States.master_pop_density_df['Recovery Factor'] < 1.1]
    state = 'North Carolina'
    min_recovery_factor = 1.2
    pop_density_tolerance = 20
    SD_delay = 7
    train_test_split = 0.4
    test_row_start = 35
    normalize_days = True
    #Define outbreak start
    percent_max_cases = 0.25


    run_model(state, min_recovery_factor, pop_density_tolerance, SD_delay,
              train_test_split, percent_max_cases, test_row_start, scatter=False)
