'''
Ideas : 

questions:

A little concerned about the data - seem to show negative trend in scatter matrix - should i trim off data with very little/ no new cases?
Implement ARIMA?
Use data from countries that have covid contained (Have data for south Korea - use that data to help train the data?)

Use just CO data or specialize in specific region of US?

Use this on cases as opposed to new cases?

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

import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.basemap import Basemap
matplotlib.rcParams.update({'font.size': 16})
plt.style.use('fivethirtyeight')
plt.close('all')

class reg_model(object):
    def __init__(self, X, y, log_trans_y = False):
        self.X = X
        self.y = y if log_trans_y == False else np.log(y + 1)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y)
        self.error_metric = None

    def lin_reg(self):
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)
        self.error_metric = 'rmse'

    def log_reg(self):
        self.model = LogisticRegression()
        self.model.fit(self.X_train, self.y_train)
        self.error_metric = 'rmse'

    def ridge_reg(self, alpha= 0.5):
        def optimize_alpha(alpha_list):
            pass
        self.model = Ridge(alpha = alpha)
        self.model.fit(self.X_train, self.y_train)
        self.error_metric = 'rss'

    def rand_forest(self, n_trees = 50):
        '''
        Upon inspection of the model over time, the number of new cases shows a period of exponential growth, then linear growth where the new cases levels off. Then a random forest model can be applied. A Y-transform should be applied on the data. 
        '''
        if n_trees == 'optimize':
            max_trees = 100
            n = np.arange(1, max_trees + 1, 1)
            error = []
            for each in n:
                self.model = RandomForestRegressor(n_estimators = each)
                self.model.fit(self.X_train, self.y_train)
                self.error_metric = 'rmse'
                error.append(self.evaluate_model())
            #plt.plot(n, error)
            n_trees = n[error.index(min(error))]
        self.model = RandomForestRegressor(n_estimators = n_trees)
        self.model.fit(self.X_train, self.y_train)
        self.error_metric = 'rmse'

    def evaluate_model(self):
        self.y_hat = self.model.predict(self.X_test)
        self.predicted_vals_df = pd.DataFrame(self.y_test)
        self.predicted_vals_df['y_hat'] = self.y_hat
        self.predicted_vals_df.sort_index(inplace = True)
        if self.error_metric == 'rmse':
            rmse = np.sqrt(mean_squared_error(self.y_test, self.y_hat))
            return rmse
        elif self.error_metric == 'rss':
            rss = np.mean((self.y_test - self.y_hat)**2)
            return rss
    
    def forecast_vals(self, to_forecast_df):
        self.forecasted = self.model.predict(to_forecast_df)
        return self.forecasted
    
    def plot_model(self):
        fig, ax = plt.subplots(figsize = (10,6))
        if self.X_test.iloc[:, 0].shape != self.y_test.shape:
            self.X_test = self.X_test.loc[:,0]
        ax.scatter(self.X_test, self.y_test, c = 'blue', label = "Test Data")
        ax.scatter(self.X_test, self.y_hat, c = 'green', label = 'Predicted Data')
        ax.legend()
        fig.show()

def clean_data(df, datetime_col = None):
    clean_df = df.copy()
    if datetime_col != None:
        clean_df[datetime_col] = pd.to_datetime(clean_df[datetime_col])
    return clean_df

def load_and_clean_data():
    '''
    Sets up and generates dataframe for analysis
    '''

    #Import and clean covid data (Cases in 2020)
    covid_raw_df = pd.read_csv('data/covid-19-data/us.csv')
    covid_df = clean_data(covid_raw_df, datetime_col='date')
    covid_df['New_Cases'] = covid_df['cases'].diff()

    # To get daily death and ratio:
    # covid_df = calc_row_diff(covid_df, colname='deaths',
    #                          added_colname='daily_death')
    # covid_df['death/case_ratio'] = covid_df['daily_death'] / \
    #     covid_df['New_Cases']
    # covid_df['death/case_ratio'] = covid_df['death/case_ratio'].fillna(0)

    #Begin loading features that may have something to do with the fluctuation of the virus

    '''
    Mobility Data - From Google
    #The baseline is the median value, for the corresponding day of the week, during the 5-week period Jan 3–Feb 6, 2020
    https://www.google.com/covid19/mobility/index.html?hl=en
    '''

    mobility_raw_df = pd.read_csv(
        'data/Global_Mobility_Report.csv', low_memory=False)
    mobility_raw_df.fillna('None', inplace=True)
    US_mobility_raw_df = mobility_raw_df[(mobility_raw_df['country_region'] == 'United States') & (
        mobility_raw_df['sub_region_1'] == 'None')]
    mobility_df = clean_data(US_mobility_raw_df, datetime_col='date')
    mobility_df.reset_index(inplace=True)
    mobility_df.drop(['index', 'country_region_code',
                      'country_region', 'sub_region_1', 'sub_region_2'], axis=1, inplace=True)
    mobility_df.rename(columns=lambda x: x.replace(
        '_percent_change_from_baseline', ''), inplace=True)
    num_cols = ['retail_and_recreation', 'grocery_and_pharmacy',
                'parks', 'transit_stations', 'workplaces', 'residential']
    mobility_df[num_cols] = mobility_df[num_cols].apply(pd.to_numeric)

    #Convert to percent of normal
    mobility_df.iloc[:, 1:] = mobility_df.iloc[:,
                                               1:].apply(lambda x: (x + 100)/100)

    '''
    Transp data - From Apple
    The CSV file and charts on this site show a relative volume of directions requests per country/region, sub-region or city compared to a baseline volume on January 13th, 2020. We define our day as midnight-to-midnight, Pacific time.
    https://www.apple.com/covid19/mobility 
    '''
    transp_raw_df = pd.read_csv('data/applemobilitytrends-2020-05-09.csv')
    transp_df = transp_raw_df[(
        transp_raw_df['region'] == 'United States')].copy()
    transp_df.drop(['geo_type', 'region', 'alternative_name'],
                   axis=1, inplace=True)
    transp_df.set_index('transportation_type', inplace=True)
    transp_df = (transp_df.T) / 100  # Convert to percentage of normal
    transp_df.columns = ['driving', 'transit', 'walking']
    transp_df.reset_index(inplace=True)
    transp_df.rename(columns={'index': 'date'}, inplace=True)
    transp_df['date'] = pd.to_datetime(transp_df['date'])

    mobility_df = mobility_df.merge(transp_df, how='inner', on='date')
    covid_df = mobility_df.merge(covid_df, how='inner', on='date')
    covid_df.drop(['cases', 'deaths'], axis=1, inplace=True)
    
    #Converts date into days elapsed since outbreak- some functions don't work with datetime objects
    dates = covid_df['date']
    covid_df['date'] = covid_df.index
    covid_df.rename(columns={'date': 'days_elapsed'}, inplace=True)
    return covid_df

if __name__ == '__main__':
    covid_df = load_and_clean_data()
    y = covid_df.pop('New_Cases')
    X = covid_df
    # ridge_model = reg_model(X, y)
    # ridge_model.ridge_reg()
    # ridge_model.evaluate_model()
    rf_model = reg_model(X, y, log_trans_y = True)
    rf_model.rand_forest(n_trees = 'optimize')
    rf_model.evaluate_model()