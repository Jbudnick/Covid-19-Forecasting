import more_itertools as mit
import pandas as pd
import numpy as np
import datetime

def create_spline(x, y, day_delay, t=7):
    '''
    Smooths out data values
        Parameters:
            x (Series): Independent Variable
            y (Series): Dependent Variable, will be smoothed
            day_delay (int): Use smoothed values by this value of days in the past
            t (int): Average values over this many days per each data value
        Returns:
            mov_avgs_x (Series): Takes moving average of x (ignore in most cases)
            mov_avgs_y (Series): Result of smoothed y
    '''
    y_raw = y.values if day_delay == 0 else y.values[:-day_delay]
    weights = np.repeat(1.0, t) / t
    mov_avgs_y = np.convolve(y_raw, weights, 'valid')
    mov_avgs_x = list(
        range(int(x.values[0] + t + day_delay), int(x.values[-1] + 2)))
    return mov_avgs_x, mov_avgs_y[:len(mov_avgs_x) + 1]

def convert_to_date(days_elapsed, original_date=datetime.date(2020, 2, 15)):
    '''
        Parameters:
            days_elapsed (int): Number of days since original_date
            original_date (datetime.date) 
        Returns:
            date_result (datetime.date)
    '''
    date_result = original_date + datetime.timedelta(days_elapsed)
    return date_result

def convert_to_days_elapsed(date, start_date=datetime.date(2020, 2, 15)):
    '''
        Parameters:
            date(datetime.date): Date to convert
            start_date(datetime.date): Days elapsed start date
        Returns:
            days_result.days (int)
    '''
    days_result = date - start_date
    return days_result.days


def replace_initial_values(df, col_change, val_col):
    '''
    When creating new feature columns using difference of existing columns, this function will replace the initial value in val_col of col_change with a 0.
        Parameters:
            df (Pandas DataFrame): Dataframe with difference column added
            col_change (string): Column name that identifies different subset of values (states)
            val_col (string): difference column name in df
        Returns:
            df (Pandas DataFrame): Same Dataframe with initial values for val_col for differing col_change values populated with 0.
    '''
    prev = None
    for i, st in zip(df.index, df[col_change]):
        if st != prev:
            df.loc[i, val_col] = 0
        else:
            continue
        prev = st
    return df


def replace_with_moving_averages(df, cols, day_delay, xcol='days_elapsed'):
    '''
    Replaces applicable rows in columns with weekly average day_delay days ago.

        Parameters:
            df (Pandas DataFrame): DataFrame including columns to replace with moving average values
            cols (list): List of strings of column names of which to replace with moving average values
            day_delay (int): Sets moving average to moving average a certain number of days in the past
            xcol (str): Column name identifying the unchanging independent variable of df
        Returns:
            df_ma (Pandas DataFrame): Same Dataframe with cols specified replace with moving average values
    '''
    df_ma = df.copy()
    for col in cols:
        max_index = max(df_ma.index)
        mv_avgs = create_spline(
            df_ma[xcol], df_ma[col], day_delay=day_delay)[1]
        applicable_row_indices = max_index - len(mv_avgs) + 1
        df_ma.loc[applicable_row_indices:, col] = mv_avgs
    return df_ma


def load_and_clean_data(use_internet=True, new_cases_per_pop=True):
    '''
    Sets up and returns dataframe for analysis
    If new cases per pop is disabled, will use raw number of new cases instead.

        Parameters:
                new_cases_per_pop (bool): True or False. Will use per capita new cases instead of raw new cases if True.
                use_internet (bool): True or False. Will retrieve latest data from the internet if left as True. Otherwise will use local files that have most recently been saved manually. (Excludes Pop Density Info- will always be imported offline)

        Returns:
                covid_df (df): Dataframe with estimated time lags populated and social distancing levels populated
    '''

    #Imports raw data - population density uploaded in cleaning section since based on 2020 and not likely to change
    if use_internet == True:
        covid_raw_df = pd.read_csv(
            'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv', parse_dates = ['date'])
        mobility_raw_df = pd.read_csv(
            'https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv?cachebust=694ae9957380f150', low_memory=False, parse_dates=['date'])
        transp_raw_df = pd.read_csv(
            'https://covid19-static.cdn-apple.com/covid19-mobility-data/2011HotfixDev15/v3/en-us/applemobilitytrends-2020-07-05.csv')
    else:
        covid_raw_df = pd.read_csv(
            'data/covid-19-data/us-states.csv', parse_dates=['date'])
        mobility_raw_df = pd.read_csv(
            'data/Global_Mobility_Report.csv', low_memory=False, parse_dates=['date'])
        transp_raw_df = pd.read_csv('data/applemobilitytrends-2020-06-01.csv')
    '''
    Clean covid data (Cases in 2020)  - From NY Times
    '''
    covid_raw_df.sort_values(['state', 'date'], inplace=True)
    covid_raw_df['New_Cases'] = covid_raw_df['cases'].diff()

    covid_raw_df = replace_initial_values(covid_raw_df, 'state', 'New_Cases')

    '''
    Clean Mobility Data - From Google
    Baseline is the median value, for the corresponding day of the week, during the 5-week period Jan 3–Feb 6, 2020
    https://www.google.com/covid19/mobility/index.html?hl=en
    '''
    US_mobility_raw_df = mobility_raw_df[(mobility_raw_df['country_region'] == 'United States') & (
        mobility_raw_df['sub_region_1'].isnull() == False) & (mobility_raw_df['sub_region_2'].isnull() == True)]
    # mobility_df = clean_data(US_mobility_raw_df, datetime_col='date')
    mobility_df = US_mobility_raw_df.reset_index()
    mobility_df.rename(columns=lambda x: x.replace(
        '_percent_change_from_baseline', ''), inplace=True)
    mobility_df.rename(columns={'sub_region_1': 'state'}, inplace=True)
    num_cols = ['retail_and_recreation', 'grocery_and_pharmacy',
                'parks', 'transit_stations', 'workplaces', 'residential']
    mobility_df[num_cols] = mobility_df[num_cols].apply(pd.to_numeric)
    mobility_df.drop(['index', 'country_region_code',
                      'country_region', 'sub_region_2'], axis=1, inplace=True)

    #Convert to percent of normal
    mobility_df[num_cols] = mobility_df[num_cols].apply(
        lambda x: (x + 100)/100)
    states = list(set(mobility_df['state']))

    '''
    Transp data - From Apple
    The CSV file and charts on this site show a relative volume of directions requests per country/region, sub-region or city compared to a baseline volume on January 13th, 2020. We define our day as midnight-to-midnight, Pacific time.
    https://www.apple.com/covid19/mobility 
    '''
    transp_df = transp_raw_df[(transp_raw_df['geo_type'] == 'sub-region')
                              & (transp_raw_df['region'].isin(states))].copy()
    #Driving is only available transportation type data available for statewide data
    transp_df.drop(['geo_type', 'alternative_name', 'transportation_type', 'sub-region', 'country'],
                   axis=1, inplace=True)
    transp_df.set_index('region', inplace=True)
    transp_df = (transp_df.T) / 100  # Convert to percentage of normal
    transp_df.reset_index(inplace=True)
    transp_df.rename(columns={'index': 'date'}, inplace=True)
    transp_df['date'] = pd.to_datetime(transp_df['date'])
    transp_df = transp_df.melt(id_vars=["date"])
    transp_df.rename(columns={'region': 'state'}, inplace=True)

    mobility_df = mobility_df.merge(
        transp_df, how='inner', on=['date', 'state'])
    covid_df = mobility_df.merge(covid_raw_df, how='inner', on=['date', 'state'])
    covid_df.rename(columns={'value': 'driving'}, inplace=True)
    #Importing state populations and land areas implement state density
    state_pops = pd.read_csv('data/pop_by_state.csv',
                             header=1, usecols=['State', 'Pop'])
    state_area = pd.read_csv('data/state_area.csv',
                             usecols=['State', 'LandArea'])
    state_pops.rename(columns={'State': 'state'}, inplace=True)
    state_area.rename(columns={'State': 'state'}, inplace=True)
    state_pops = state_pops.merge(state_area, on='state')
    state_pops['pop_density'] = state_pops['Pop'] / state_pops['LandArea']
    if new_cases_per_pop == True:
        state_pops['Pop'] = state_pops['Pop'] / 1000000
        covid_df = covid_df.merge(state_pops, on='state')
        covid_df['New_Cases_per_pop'] = covid_df['New_Cases'] / covid_df['Pop']

    #Converts date into days elapsed since Feb 15th
    covid_df['date'] = covid_df['date'].apply(
        lambda x: convert_to_days_elapsed(x.date()))
    covid_df.rename(columns={'date': 'days_elapsed'}, inplace=True)

    #Include only columns of interest
    wanted_columns = ['state', 'days_elapsed', 'retail_and_recreation',
                      'grocery_and_pharmacy', 'parks', 'transit_stations', 'workplaces',
                      'residential', 'driving', 'pop_density', 'New_Cases_per_pop']
    covid_df = covid_df.loc[:, wanted_columns]
    covid_df = fill_na_with_surround(covid_df)
    return covid_df


def fill_na_with_surround(df, cols = 'all'):
    '''
    Used to fill NaN values with the average of the two surrounding values in each series of missing values
    or standalone value. If most recent value for state is NaN, will replicate the most recently known value to all suceeding NaNs.
    Assumes that each state has a numeric value populated as its initial value.

        Parameters:
            df (Pandas DataFrame): Dataframe to modify to fill na values with average of surrounding
            cols (Pandas Series): Columns of df to fill na values
        Returns:
            df (Pandas DataFrame): DataFrame with NA values filled for specified cols
    '''
    if cols == 'all':
        cols = df.columns

    for col in cols:
        indices = df[df[col].isnull()].index.values
        if len(indices > 0):
            consec_list = [list(consecutive)
                           for consecutive in mit.consecutive_groups(indices)]
            for sub_list in consec_list:
                sub_list.insert(0, sub_list[0]-1)
                sub_list.append(sub_list[-1] + 1)
                relevant_cols = ['state', col]
                val_1 = df.loc[sub_list, col].iloc[0]
                in_index = sub_list[-1] <= df.index.max()
                try:
                    if in_index and df.loc[sub_list[0], 'state'] == df.loc[sub_list[-1], 'state']:
                        val_2 = df.loc[sub_list, col].iloc[-1]
                        avg_val = (val_1 + val_2)/2
                        df.loc[sub_list, col] = df.loc[sub_list,
                                                    col].fillna(avg_val)
                    else:
                        sub_list.pop()
                        df.loc[sub_list, col] = df.loc[sub_list, col].fillna(val_1)
                except:
                    breakpoint()
    return df

def convert_to_moving_avg_df(covid_df, states = 'all', SD_delay = 10):
    '''
    Converts dataframe into moving averages instead of raw values. Differs from replace_with_moving average in that this function is intended for multiple states.
        Parameters:
            covid_df (pandas DataFrame)
            states (list or 'all')
            SD_delay = 10 (int): A delayed number of days to use for social distancing parameters (10 will be moving average of 10 days ago set on current day)
        Returns:
            ma_df (pandas DataFrame): Dataframe of specified states with values converted to moving averages.
    '''
    ma_df = pd.DataFrame()
    if states == 'all':
        states = covid_df['state'].unique()
    for state in states:
        mask1 = (covid_df['state'] == state)
        state_df = covid_df[mask1]
        state_df = replace_with_moving_averages(state_df, [state_df.columns[-1]], day_delay = 0)
        state_df = replace_with_moving_averages(state_df, state_df.columns[2:-1], day_delay= SD_delay)
        ma_df = ma_df.append(state_df)
    return ma_df
