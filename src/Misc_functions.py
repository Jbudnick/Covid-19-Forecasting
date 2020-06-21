from src.data_clean_script import replace_with_moving_averages
from src.data_clean_script import load_and_clean_data, convert_to_moving_avg_df

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def series_to_supervised(data, columns, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        columns: Columns of data 
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

def get_future_SD_params(SD_delay, state):
    '''
    If a delay is set on the main DataFrame SD parameters for the moving average, this function will be called to pull current moving average SD parameters from the original dataset to populate future SD parameters.
    '''
    raw_covid_df = load_and_clean_data(use_internet=True)
    raw_covid_df = raw_covid_df[raw_covid_df['state'] == state]
    covid_df = convert_to_moving_avg_df(raw_covid_df, SD_delay= 0)
    SD_params_future = covid_df.iloc[-SD_delay:, 2:-2]
    return SD_params_future

def blank_out_lagged_columns(df, row_start):
    col_end = df.columns.get_loc('New_Cases_per_pop(t-1)')
    df.iloc[row_start:, :col_end + 1] = 0
    return df

def populate_predictions(df, preds, model, start_row, end_row = 'all', n_interval=21):
    '''
    Used to populate time lagged observations - diagonal on supervised matrix for time lagged columns
        Parameters:
            df (Pandas DataFrame): Dataframe with time lags
            preds (Series): Predicted values
            model (Regression Model): Model to use to populate predicted values
            start_row (int): Row to start replacing with predictions
            n_interval(int): Number of days of predictions
        Returns:
            df (Pandas DataFrame): DataFrame with time lags populated
            new_preds (Series): Series of new predictions
    '''
    df.fillna(0, inplace=True)
    if end_row == 'all':
        end_row = df.shape[0]
    else:
        end_row = df.index.get_loc(end_row)
    new_preds = list(preds.values)
    start_row = df.index.get_loc(start_row)
    for row in range(start_row, end_row):
        new_pred = model.predict(df[row:row + 1])[0]
        new_preds.append(new_pred)
        j = 1
        for col in range(n_interval-1, -1, -1):
            try:
                if df.iloc[row + j, col] == 0:
                    df.iloc[row + j, col] = new_pred
                j += 1
            except:
                continue
    new_pred = model.predict(df[-1:-2:-1])[0]
    new_preds.append(new_pred)
    return df, new_preds

def fill_blank_known_ts(pred_df, total_y, row_start, row_end = 'all'):
    if row_end == 'all':
        row_end = pred_df.shape[0]
    else:
        row_end = pred_df.index.get_loc(row_end)
    # pred_df.fillna(0, inplace=True)
    col_start = pred_df.columns.get_loc('New_Cases_per_pop(t-1)')
    try:
        row_start = pred_df.index.get_loc(row_start)
    except:
        breakpoint()
    if type(total_y) is pd.Series:
        pred_df.iloc[row_start, col_start] = total_y.values[-1]
    else:
        pass

    for row in range(row_start, row_end):
        for col in range(col_start - 1, -1, -1):
            pred_df.iloc[row, col] = pred_df.iloc[row - 1, col + 1]
    return pred_df

def generate_prediction_df(level, total_x, total_y, rf, predictions=21, SD_delay = 10):
    '''
    Generates a pandas Dataframe out into the future. Uses predictions with time lags on future predictions.

    INPUT:
        level: 'High', 'Medium', or 'Low' or custom list of social distancing parameters
        total_x: Feature matrix (not including target) with all features and time series lags included
        total_y: Target values from total_x
        rf: Random Forest Model
        SD_delay (int): If the moving average taken for social distancing was other than 10, this should be populated
        Predictions: Time lagged features to predict out toward future

    OUTPUT:
        Dataframe with estimated time lags populated and social distancing levels populated
        Series with estimated target values for each row in dataframe

    '''
    #Part 1: Expands time lagged Daily New Cases columns

    #Columns retail_and_recreation through driving are the only ones that need to be specified - rest are populated automatically
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
    future_index = pred_df.index.max()

    state = pred_df['state(t)'].unique()[0]
    pred_df.drop('state(t)', axis=1, inplace=True)

    for i in range(last_recorded_day + 1, last_recorded_day + predictions + 1):
        pred_df_row = pd.DataFrame([i] + pred_params + [pop_dens]).T
        pred_df_row.columns = columns
        future_index += 1
        pred_df_row.index = [future_index]
        pred_df = pred_df.append(pred_df_row, sort=False)

    if SD_delay != 0:
        SD_future = get_future_SD_params(SD_delay, state=state)
        SD_future.columns += '(t)'
        st_idx = pred_df[pred_df['days_elapsed(t)'] == last_recorded_day].index[0] + 1
        st_col = pred_df.columns.get_loc('retail_and_recreation(t)')
        end_col = len(SD_future.columns) + st_col
        for i in range(len(SD_future.index)):
            pred_df.iloc[st_idx + i, st_col: end_col] = SD_future.iloc[i, :]

    # Part 2: Fills in blank known new cases values
    pred_df.fillna(0, inplace=True)
    row_start = pred_df.shape[0] - pred_df[pred_df['New_Cases_per_pop(t-1)'] == 0].count()[0]
    pred_df = fill_blank_known_ts(pred_df = pred_df, total_y = total_y, row_start = row_start)

    #Part 3: Fills in rest of time lagged values for future t values, predicting based on prior predictions
    fill_diag_and_predictions = populate_predictions(pred_df, total_y, rf.model, start_row= row_start, n_interval=21)
    pred_df = fill_diag_and_predictions[0]
    pred_y = fill_diag_and_predictions[1][-pred_df.shape[0]:]
    return pred_df, pred_y

def find_nearest(array, value):
    idx = (np.abs(array - value)).idxmin()
    return idx

def normalize_days(compiled_state_df, percent_max= 0.25):
    '''
    TBD
    Process covid_df day elapsed column into days elapsed since hitting percent_max of its maximum number of cases/person.
    save_x_starts will return a tuple to translate back into actual date later.
        Parameters:
            states (list)
            covid_df (Pandas df): Dataframe used to normalize
            percent_max (float): Value to use to determine the start of outbreak (0.25 = 25% of maximum new cases is starting point)
            save_x_starts (bool): Whether to save original days elapsed values to convert back later
        Returns:
            state_dfs (Pandas DataFrame): Dataframe with added column to normalize days since outbreak
            x_starts (Series): Original time values before normalization
    '''
    states = compiled_state_df['state(t)'].unique()
    normalized_df = pd.DataFrame()     
    for i, state in enumerate(states):
        specific_df = compiled_state_df[compiled_state_df['state(t)'] == state].copy()
        x = specific_df['days_elapsed(t)']
        y = specific_df['New_Cases_per_pop']
        y_start = max(y) * percent_max
        y = pd.to_numeric(y)
        max_index = y.idxmax()
        y_idx = find_nearest(y.loc[:max_index], y_start)
        x_start = x[y == y.loc[y_idx]].values[0]
        specific_df['days_since_start'] = specific_df['days_elapsed(t)'] - x_start
        normalized_df = normalized_df.append(specific_df)
    return normalized_df.reset_index(drop = True)

def plot_normalized(normalized_df, Compiled_State_obj):
    min_day = Compiled_State_obj.min_days
    train_test_split = Compiled_State_obj.rf.train_test_split
    states = normalized_df['state(t)'].unique()
    fig, ax = plt.subplots(figsize=(12, 6))

    state_predict_norm = Compiled_State_obj.state_to_predict_norm

    ax.plot(state_predict_norm['days_since_start'], state_predict_norm['New_Cases_per_pop'], ls='--',
            label=state_predict_norm['state(t)'].unique()[0])

    for i, state in enumerate(states):
        specific_df = normalized_df[normalized_df['state(t)'] == state].copy()
        x = specific_df['days_elapsed(t)']
        y = specific_df['New_Cases_per_pop']
        ax.plot(specific_df['days_elapsed(t)'], y, label=state)
    ax.axvline(min_day, label = 'Minimum Day for Training Set', ls = '-.', c = 'black', lw = 1)
    ax.axvline(train_test_split, label = 'Train/Test Split', ls = '-.', c = 'grey', lw = 1)
    ax.set_title('Daily New Cases Plot (Normalized)')
    ax.set_xlabel('Days Since {}% of Maximum Cases'.format(Compiled_State_obj.percent_of_max * 100))
    ax.set_ylabel('Daily New Cases/1M Pop')
    ax.legend()
    fig.tight_layout()
    fig.show()
