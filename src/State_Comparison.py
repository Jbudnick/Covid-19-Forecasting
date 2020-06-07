from pandas.plotting import register_matplotlib_converters
pop_density = covid_df[['state', 'pop_density']].drop_duplicates()
NY_pop_density = pop_density[(
    pop_density['state'] == 'New York')]['pop_density'].values[0]


class Combined_State_Analysis(reg_model):

    '''
    Provide list of state_dfs to combined into one model to create a dataset for training.
    MUST HAVE AT LEAST 2 STATES IN LIST
    '''

    def __init__(self, state_list, eval=False):
        covid_df = load_and_clean_data()
        self.state_list = state_list
        X_df_list = [state_analysis(covid_df, state=state, print_err=False)[
            1] for state in state_list]
        y_df_list = [state_analysis(covid_df, state=state, print_err=False)[
            2] for state in state_list]
        self.X = X_df_list[0].append(X_df_list[1:])
        self.y = y_df_list[0].append(y_df_list[1:])

        self.rf = reg_model(self.X, self.y)
        self.rf.rand_forest()
        self.rf.evaluate_model(
            print_err_metric=True) if eval == True else self.Compile_rf.evaluate_model()
            
def get_day_of_peak(df, target='New_Cases'):
    top = df.sort_values(target, ascending=False).iloc[0]
    peak_day = top.loc['days_elapsed']
    peak_val = top.loc['New_Cases']
    return peak_day, peak_val


class other_state(object):
    '''
    This class is intended to load data for prediction purposes. Unlike 
    use_new_case_per_capita is False by default for scaling to make units more interpretable when normalizing models
    mvg_avg_df = replace_with_moving_averages(self.NY_df, self.NY_df.columns[2:-1], day_delay = 3)
        mvg_avg_df = replace_with_moving_averages(self.df, self)
        '''

    def __init__(self, state, per_capita=False, use_mvg_avg=True):

        self.state = state
        self.pop_density = pop_density[(
            pop_density['state'] == state)]['pop_density'].values[0]
        if per_capita == True:
            covid_df = load_and_clean_data(new_cases_per_pop=True)
            self.df = covid_df[covid_df['state'] == state]
            self.NY_df = covid_df[covid_df['state'] == 'New York']
            if use_mvg_avg == True:
                self.df = self.apply_moving_avgs(
                    self.df, ['New_Cases_per_pop'])
                self.df = self.apply_moving_avgs(
                    self.df, self.df.columns[2: -1], day_delay=3)
                self.NY_df = self.apply_moving_avgs(self.NY_df, ['New_Cases'])
                self.NY_df = self.apply_moving_avgs(
                    self.NY_df, self.NY_df.columns[2: -1], day_delay=3)
            self.y = self.df['New_Cases_per_pop']
            self.NY_data_y = self.NY_df['New_Cases_per_pop']
        else:
            covid_df = load_and_clean_data(new_cases_per_pop=False)
            self.df = covid_df[covid_df['state'] == state]
            self.NY_df = covid_df[covid_df['state'] == 'New York']
            if use_mvg_avg == True:
                self.df = self.apply_moving_avgs(self.df, ['New_Cases'])
                self.df = self.apply_moving_avgs(
                    self.df, self.df.columns[2: -1], day_delay=3)
                self.NY_df = self.apply_moving_avgs(self.NY_df, ['New_Cases'])
                self.NY_df = self.apply_moving_avgs(
                    self.NY_df, self.NY_df.columns[2: -1], day_delay=3)
            self.y = self.df['New_Cases']
            self.NY_data_y = self.NY_df['New_Cases']

        self.X = self.df['days_elapsed']
        self.NY_data_X = self.NY_df['days_elapsed']

    def pop_dens_scale(self):
        self.pop_scale = pop_density[(pop_density['state'] == 'New York')]['pop_density'].values[0] / \
            pop_density[(pop_density['state'] == self.state)
                        ]['pop_density'].values[0]
        return self.pop_scale

    def apply_moving_avgs(self, df, cols, day_delay=0):
        '''
        replace_with_moving_averages(
        Minnesota_Analysis.NY_df, Minnesota_Analysis.NY_df.columns[2:-1], day_delay=3)
        '''
        mvg_avg_df = replace_with_moving_averages(
            df, cols, xcol='days_elapsed', day_delay=0)
        return mvg_avg_df

    def normalize_to_NY(self, x_mod_adj):
        '''        
        Currently determined by visual inspection of plots:
        x_mod is the number of days the virus infection appears to be behind NY
        y_mod is a number subtracted from the density scale to normalize peaks/shape of data to match NY.
        '''
        self.x_mod = get_day_of_peak(self.df)[0] + x_mod_adj
        self.y_scale = get_day_of_peak(
            self.NY_df)[1] / get_day_of_peak(self.df)[1]
        self.day_diff = get_day_of_peak(
            self.df)[0] - get_day_of_peak(self.NY_df)[0] + x_mod_adj

    def plot_vs_NY(self, x_mod_adj=0, axis='Date', save=False):
        NY_peak_day = 55
        register_matplotlib_converters()

        fig, ax = plt.subplots(figsize=(12, 6))
        if axis == 'Date':
            NY_X = self.NY_data_X.apply(convert_to_date)
            state_X = self.X.apply(convert_to_date)
        elif axis == 'Days Since Peak of Outbreak':
            self.normalize_to_NY(x_mod_adj=x_mod_adj)
            NY_X = self.NY_data_X - NY_peak_day
            state_X = self.X - self.x_mod
            ax.annotate("*X - Shifted Back {0:.{1}f} days\n*y - Scaled {2:.{3}f} times".format(self.day_diff, 0, self.y_scale, 1),
                        xy=(0.1, 0.72), xycoords='figure fraction')
        else:
            NY_X = self.NY_data_X
            state_X = self.X

        ax.plot(NY_X, self.NY_data_y, label='New York', c='red')
#         ax.legend(loc = 2)
        ax.set_title('Covid-19 New Cases Comparison (Weekly Average)')
        ax.set_xlabel(axis)
        ax.set_ylabel('NY Daily New Cases')

        ax2 = ax.twinx()
        ax2.plot(state_X, self.y, label=self.state + '*')
#         ax2.legend(loc = 1)
        ax2.set_ylabel('{} Daily Cases'.format(self.state))
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc=2)
        ax2.grid(None)
        fig.tight_layout()

        if save != False:
            fig.savefig(save)

    def scale_state_to_NY(self):
        '''
        In order to get insight as to how the specified state should aim for social distancing, the data
        will be scaled up to numbers for predictions with NY's random forest, then scaled back down for
        interpretable estimates.
        '''
        fill_na_with_surround(self.df, 'driving', ind_loc='loc')
        self.df['New_Cases'] = (self.df['New_Cases'] *
                                self.y_scale) / (self.pop_density)
        self.df['days_elapsed'] = self.df['days_elapsed'] - self.x_mod
        self.df.drop('state', axis=1, inplace=True)
        self.df.rename(columns={'New_Cases': 'Daily New Cases'}, inplace=True)
        self.ts_df = series_to_supervised(
            self.df.values, self.df.columns, 21, 1)
        self.ts_df = self.ts_df.iloc[:, 8:-5:9].join(self.ts_df.iloc[:, -9:])
        self.ts_df.index.name = self.state

    def scale_back_to_state(self):
        pass
