from src.data_clean_script import load_and_clean_data
import more_itertools as mit


def fill_na_with_surround(df, cols = df.columns):
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
    for col in cols:
        indices = df[df[col].isnull()].index.values
        if len(indices > 0):
            consec_list = [list(consecutive) for consecutive in mit.consecutive_groups(indices)]
            for sub_list in consec_list:
                sub_list.insert(0, sub_list[0]-1)
                sub_list.append(sub_list[-1] + 1)
                relevant_cols = ['state', col]
                val_1 = df.loc[sub_list, col].iloc[0]
                if df.loc[sub_list[0], 'state'] == df.loc[sub_list[-1], 'state']:
                    val_2 = df.loc[sub_list, col].iloc[-1]
                    avg_val = (val_1 + val_2)/2
                    df.loc[sub_list, col] = df.loc[sub_list, col].fillna(avg_val)
                else:
                    df.loc[sub_list, col] = df.loc[sub_list, col].fillna(val_1)
    return df

if __name__ == '__main__':
    covid_df = load_and_clean_data(use_internet = True)
    covid_df = fill_na_with_surround(covid_df, covid_df.columns)
