pop_density = covid_df[['state', 'pop_density']].drop_duplicates()


class other_state(object):
    '''
    This class draws comparisons with the NY data to make insights about the state's data -
    it assumes that covid data is already imported.
    '''

    def __init__(self, state):

        self.state = state
        self.df = covid_df[covid_df['state'] == state]
        self.X = self.df['days_elapsed']
        self.y = self.df['New_Cases_per_pop']

        NY_df = covid_df[covid_df['state'] == 'New York']
        self.NY_data_X = NY_df['days_elapsed']
        self.NY_data_y = NY_df['New_Cases_per_pop']

    def pop_dens_scale(self):
        self.pop_scale = pop_density[(pop_density['state'] == 'New York')]['pop_density'].values[0] / \
            pop_density[(pop_density['state'] == self.state)
                        ]['pop_density'].values[0]
        return self.pop_scale

    def apply_moving_avgs(self):
        self.replace_with_moving_averages(
            df, cols, xcol='days_elapsed', day_delay=0)

    def plot_vs_NY(self, x_mod=0, y_mod=0):
        '''
        Currently determined by visual inspection of plots:
        x_mod is the number of days the virus infection appears to be behind NY
        y_mod is a number subtracted from the density scale to normalize peaks/shape of data to match NY.
        '''
        self.pop_scale = self.pop_dens_scale()
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(self.X - x_mod, self.y *
                (self.pop_scale - y_mod), label=self.state)

        ax.plot(self.NY_data_X, self.NY_data_y, label='New York')
        ax.legend()
        fig.tight_layout()
        ax.set_title('Covid-19 New Cases')
