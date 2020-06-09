import pandas as pd
import numpy as np
import datetime

def create_spline(x, y, day_delay, t=7):
    '''
    Use moving average of t points at day_delay
    '''
    y_raw = y.values if day_delay == 0 else y.values[:-day_delay]
    weights = np.repeat(1.0, t) / t
    mov_avgs_y = np.convolve(y_raw, weights, 'valid')
    mov_avgs_x = list(
        range(int(x.values[0] + t + day_delay), int(x.values[-1] + 2)))
    return mov_avgs_x, mov_avgs_y[:len(mov_avgs_x) + 1]


def clean_data(df, datetime_col=None):
    clean_df = df.copy()
    if datetime_col != None:
        clean_df[datetime_col] = pd.to_datetime(clean_df[datetime_col])
    return clean_df


def convert_to_date(days_elapsed, original_date=datetime.date(2020, 2, 15)):
    date_result = original_date + datetime.timedelta(days_elapsed)
    return date_result


def replace_initial_values(df, col_change, val_col):
    '''
    When creating new feature columns using difference of existing columns, this function will replace the initial value in val_col of col_change with a 0.
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
    Replaces applicable rows  in columns with weekly average days_past days ago.
    Day delay is an optional parameter if we want to set the moving average to the weekly moving average x number of days ago.

    '''
    df_ma = df.copy()
    for col in cols:
        max_index = max(df_ma.index)
        mv_avgs = create_spline(
            df_ma[xcol], df_ma[col], day_delay=day_delay)[1]
        applicable_row_indices = max_index - len(mv_avgs) + 1
        df_ma.loc[applicable_row_indices:, col] = mv_avgs
    return df_ma


def load_and_clean_data(new_cases_per_pop=True):
    '''
    Arguments: new_cases_per_pop = True
    Sets up and generates dataframe for analysis 
    If new cases per pop is disabled, will use raw number of new cases instead.
    '''

    #Import and clean covid data (Cases in 2020)
    covid_raw_df = pd.read_csv('data/covid-19-data/us-states.csv')
    covid_df = clean_data(covid_raw_df, datetime_col='date')
    covid_df.sort_values(['state', 'date'], inplace=True)
    covid_df['New_Cases'] = covid_df['cases'].diff()

    covid_df = replace_initial_values(covid_df, 'state', 'New_Cases')

    '''
    Mobility Data - From Google
    #The baseline is the median value, for the corresponding day of the week, during the 5-week period Jan 3–Feb 6, 2020
    https://www.google.com/covid19/mobility/index.html?hl=en
    '''

    mobility_raw_df = pd.read_csv(
        'data/Global_Mobility_Report.csv', low_memory=False)
    US_mobility_raw_df = mobility_raw_df[(mobility_raw_df['country_region'] == 'United States') & (
        mobility_raw_df['sub_region_1'].isnull() == False) & (mobility_raw_df['sub_region_2'].isnull() == True)]
    mobility_df = clean_data(US_mobility_raw_df, datetime_col='date')
    mobility_df.reset_index(inplace=True)
    mobility_df.drop(['index', 'country_region_code',
                      'country_region', 'sub_region_2'], axis=1, inplace=True)
    mobility_df.rename(columns=lambda x: x.replace(
        '_percent_change_from_baseline', ''), inplace=True)
    mobility_df.rename(columns={'sub_region_1': 'state'}, inplace=True)
    num_cols = ['retail_and_recreation', 'grocery_and_pharmacy',
                'parks', 'transit_stations', 'workplaces', 'residential']
    mobility_df[num_cols] = mobility_df[num_cols].apply(pd.to_numeric)

    #Convert to percent of normal
    mobility_df[num_cols] = mobility_df[num_cols].apply(
        lambda x: (x + 100)/100)
    states = list(set(mobility_df['state']))
    '''
    Transp data - From Apple
    The CSV file and charts on this site show a relative volume of directions requests per country/region, sub-region or city compared to a baseline volume on January 13th, 2020. We define our day as midnight-to-midnight, Pacific time.
    https://www.apple.com/covid19/mobility 
    '''
    transp_raw_df = pd.read_csv('data/applemobilitytrends-2020-06-01.csv')
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
    covid_df = mobility_df.merge(covid_df, how='inner', on=['date', 'state'])
    covid_df.rename(columns={'value': 'driving'}, inplace=True)
    covid_df.drop(['cases', 'deaths', 'fips'], axis=1, inplace=True)

    #Converts date into days elapsed since outbreak- some functions don't work with datetime objects
    #February 15th is earliest data
    min_date = datetime.datetime(2020, 2, 15)
    covid_df['date'] = covid_df['date'].apply(
        lambda x: (x.to_pydatetime() - min_date).days)
    dates = covid_df['date']
    covid_df.rename(columns={'date': 'days_elapsed'}, inplace=True)

    #Importing state populations and land areas - going to convert cases to new cases per capita for better comparison, implement state density
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
        covid_df.drop(['LandArea', 'Pop'], axis=1, inplace=True)
        covid_df.drop(['New_Cases'], axis=1, inplace=True)

    #2 missing park values; manually fill them in with average of surrounding value
    missing_parks_ind = [507, 514, 661, 668,
                         675, 682, 689, 1017, 1024, 1031, 2940]
    covid_df = fill_na_with_surround(
        covid_df, 'parks', series=missing_parks_ind)
    return covid_df


def fill_na_with_surround(df, col, series=True, ind_loc='iloc'):
    '''
    Can be used to fill NA values with the average of the two surrounding values in a series of missing values,
    or standalone value (specified with series argument).
    Note: Currently only tested if one continuous series of non numeric values exists in the specified col.
    Assumes valid value exists after the series of NaNs.
    '''
    indices = df[df[col].isnull()].index.values
    if len(indices) != 0:
        if series == True:
            if ind_loc == 'iloc':
                val_1 = df[col].iloc[min(indices) - 1]
                val_2 = df[col].iloc[max(indices) + 1]
            else:
                val_1 = df[col].loc[min(indices) - 1]
                val_2 = df[col].loc[max(indices) + 1]
            replace = (val_1 + val_2) / 2
            df[col].fillna(replace, inplace=True)
        else:
            for row in series:
                df.loc[row, col] = (df.loc[row - 1, col] +
                                    df.loc[row + 1, col]) / 2
    return df

def get_moving_avg_df(covid_df, state):
    mask1 = (covid_df['state'] == state)
    state_df = covid_df[mask1]
    y = state_df.pop('New_Cases_per_pop')
    X = state_df.iloc[:, 1: -1]

    #Calculate moving average, use as target variable instead of raw new cases/pop
    smooth_x, smooth_y = create_spline(X['days_elapsed'], y, day_delay=0)
    mov_avg_df = pd.DataFrame([smooth_x, smooth_y]).T
    mov_avg_df.columns = ('days_elapsed', 'Daily_Cases_per_pop')
    state_df = replace_with_moving_averages(
        state_df, state_df.columns[2:-1], day_delay=10)
    #Mask to limit start of moving average dataframe to when the number of daily new cases reaches threshold
    mask_mov_avg = (mov_avg_df['Daily_Cases_per_pop'] >= 450) | (mov_avg_df['days_elapsed'] > 55)
    mov_avg_df = mov_avg_df[mask_mov_avg]

    revised_df = state_df.merge(mov_avg_df, on='days_elapsed').iloc[:, 1:]
    fill_na_with_surround(revised_df, 'driving')
    return revised_df
