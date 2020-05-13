'''
Ideas : 

questions:

A little concerned about the data - seem to show negative trend in scatter matrix - should i trim off data with very little/ no new cases?
Implement ARIMA?
Use data from countries that have covid contained (Have data for south Korea - use that data to help train the data?)

Bring in weather data? Other country data?

Could spikes in the data be caused by a sudden increase in available tests for a state, and should be removed?
Maybe try using deaths instead of cases; lack of test availablility is a problem so it's possible that the number is fluctuating on account of this? But plot of cases versus deaths looks pretty linear.... but deaths seem to be more predictable than cases...
Trim data so that the signal is not misled by earlier observations?

Bring in data from travel from other countries - maybe that would help?
https://travel.trade.gov/view/m-2017-I-001/index.asp

Change days elapsed to unique to each individual state or from earliest date with data?


'''
import pandas as pd
import datetime
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from scipy.interpolate import make_interp_spline

import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.basemap import Basemap
matplotlib.rcParams.update({'font.size': 16})
plt.style.use('fivethirtyeight')
plt.close('all')


class reg_model(object):
    def __init__(self, X, y, log_trans_y=False):
        self.X = X
        if log_trans_y == True:
            elim_invalid = y.copy()
            elim_invalid[elim_invalid < 0] = 0
            self.y = np.log(elim_invalid + 1)
        else:
            self.y = y
        self.log_trans_y = log_trans_y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 0.5) # This doesn't work for forecasting
        self.error_metric = None

    def lin_reg(self):
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)
        self.error_metric = 'rmse'

    def log_reg(self):
        self.model = LogisticRegression()
        self.model.fit(self.X_train, self.y_train)
        self.error_metric = 'rmse'

    def ridge_reg(self, alpha=0.5):
        def optimize_alpha(alpha_list):
            pass
        self.model = Ridge(alpha=alpha)
        self.model.fit(self.X_train, self.y_train)
        self.error_metric = 'rss'

    def rand_forest(self, n_trees=50):
        '''
        Upon inspection of the model over time, the number of new cases shows a period of exponential growth, then linear growth where the new cases levels off. Then a random forest model can be applied. A Y-transform should be applied on the data. 
        '''
        if n_trees == 'optimize':
            max_trees = 100
            n = np.arange(1, max_trees + 1, 1)
            error = []
            for each in n:
                self.model = RandomForestRegressor(n_estimators=each)
                self.model.fit(self.X_train, self.y_train)
                self.error_metric = 'rmse'
                error.append(self.evaluate_model())
            #plt.plot(n, error)
            n_trees = n[error.index(min(error))]
        self.model = RandomForestRegressor(n_estimators=n_trees)
        self.model.fit(self.X_train, self.y_train)
        self.error_metric = 'rmse'

    def evaluate_model(self):
        self.y_hat = self.model.predict(self.X_test)
        self.predicted_vals_df = pd.DataFrame(self.y_test)
        self.predicted_vals_df['days_elapsed'] = self.X_test['days_elapsed']
        self.predicted_vals_df['y_hat'] = self.y_hat
        self.predicted_vals_df.sort_index(inplace=True)
        if self.error_metric == 'rmse':
            rmse = np.sqrt(mean_squared_error(self.y_test, self.y_hat))
            return rmse
        elif self.error_metric == 'rss':
            rss = np.mean((self.y_test - self.y_hat)**2)
            return rss

    def forecast_vals(self, to_forecast_df):
        self.forecasted = self.model.predict(to_forecast_df)
        return self.forecasted

    def plot_model(self, use_smoothed = True):
        fig, ax = plt.subplots(figsize=(10, 6))
        if self.log_trans_y == True:
            self.y_test = np.e ** self.predicted_vals_df['New_Cases_per_pop']
            self.y_hat = np.e ** self.predicted_vals_df['y_hat']
        ax.bar(self.predicted_vals_df.loc[:, 'days_elapsed'], self.y_test, color = 'blue', label="Test Data")
        ax.bar(self.X_train.loc[:, 'days_elapsed'], np.e ** self.y_train, color = 'red', label="Training Data")
        if use_smoothed == True:
            x, y = create_spline(self.predicted_vals_df.loc[:, 'days_elapsed'], self.y_hat)
            ax.plot(x, y, c = 'green', label='Predicted Data')
        else:
            ax.plot(self.predicted_vals_df.loc[:, 'days_elapsed'], self.y_hat, c='green', label='Predicted Data')
        
        ax.legend()
        ax.set_xlabel('Days Since Feb 15')
        ax.set_ylabel('Cases per 1 Million Population')
        fig.show()


def clean_data(df, datetime_col=None):
    clean_df = df.copy()
    if datetime_col != None:
        clean_df[datetime_col] = pd.to_datetime(clean_df[datetime_col])
    return clean_df


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

def create_spline(x, y):
    x_smooth = np.linspace(x.min(), x.max(), 15)
    a_BSpline = make_interp_spline(x, y)
    y_smooth = a_BSpline(x_smooth)
    return x_smooth, y_smooth

def load_and_clean_data():
    '''
    Sets up and generates dataframe for analysis
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
    transp_raw_df = pd.read_csv('data/applemobilitytrends-2020-05-09.csv')
    transp_df = transp_raw_df[(transp_raw_df['geo_type'] == 'sub-region')
                              & (transp_raw_df['region'].isin(states))].copy()
    #Driving is only available transportation type data available for statewide data
    transp_df.drop(['geo_type', 'alternative_name', 'transportation_type'],
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
    state_area = pd.read_csv('data/state_area.csv', usecols=['State', 'TotalArea'])
    state_pops.rename(columns={'State': 'state'}, inplace=True)
    state_area.rename(columns={'State': 'state'}, inplace=True)
    state_pops = state_pops.merge(state_area, on = 'state')
    state_pops['pop_density'] = state_pops['Pop'] / state_pops['TotalArea']
    state_pops['Pop'] = state_pops['Pop'] / 1000000
    covid_df = covid_df.merge(state_pops, on = 'state')
    covid_df['New_Cases_per_pop'] = covid_df['New_Cases'] / covid_df['Pop']
    covid_df.drop(['TotalArea', 'New_Cases', 'Pop'], axis = 1, inplace = True)

    #2 missing park values; manually fill them in with average of surrounding value
    covid_df.loc[507, 'parks'] = (covid_df.loc[506, 'parks'] + covid_df.loc[508, 'parks'])/2
    covid_df.loc[514, 'parks'] = (covid_df.loc[513, 'parks'] + covid_df.loc[515, 'parks'])/2
    
    return covid_df

def state_plot(state, df):
    fig, axes = plt.subplots(8, 1, figsize=(12, 15))
    for i, ax in enumerate(axes, 2):
        query = df[df['state'] == state]['days_elapsed']
        x = query.values
        y = covid_df.loc[query.index].iloc[:, i]
        ax.plot(x, y)
    fig.show()

if __name__ == '__main__':
    covid_df = load_and_clean_data()
    mask1 = (covid_df['state'] == 'New York')
    mask2 = (covid_df['days_elapsed'] >= 0)
    y = covid_df[mask1 & mask2].pop('New_Cases_per_pop')
    X = covid_df[mask1 & mask2].iloc[:, 1: -1]
    rf_model = reg_model(X, y, log_trans_y=True)
    rf_model.rand_forest(n_trees='optimize')
    rf_model.evaluate_model()
    rf_model.plot_model()
