# Forecasting the Future of COVID-19 with Social Distancing 

## Table of Contents
<!-- vscode-markdown-toc -->
* 1. [Introduction](#Introduction)
* 2. [Data + Cleaning](#DataCleaning)
* 3. [Exploratory Data Analysis](#ExploratoryDataAnalysis)
* 4. [Model Results and Forecasting Per State](#ModelResultsandForecastingPerState)
		* 4.1. [Alabama](#Alabama)
		* 4.2. [Arizona](#Arizona)
		* 4.3. [Arkansas](#Arkansas)
		* 4.4. [California](#California)
		* 4.5. [Mississippi](#Mississippi)
		* 4.6. [North Carolina](#NorthCarolina)
		* 4.7. [Oklahoma](#Oklahoma)
* 5. [Conclusion](#Conclusion)
* 6. [Future Ideas for Improvement](#FutureIdeasforImprovement)
* 7. [Data Sources](#DataSources)

<!-- vscode-markdown-toc-config
	numbering=true
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc --><p>
<img src="images/covid19_economy_jpg.jpeg" width="600">
<p>

##  1. <a name='Introduction'></a>Introduction

The question on everyone's mind these days seems to be the same: when will things go back to normal? The United States has been hit especially hard by the COVID-19 pandemic and we are all hoping that the end is in sight. With the implementation of social distancing playing a massive role in our lives today, what level of social distancing should be maintained so that we see a consistent drop in the number of new cases of the virus? 

In this project, I have created a model that will predict a forecast of daily new cases per capita for a state currently experiencing the maximum number of cases with different levels of social distancing by drawing a subset of states similar in population density that are further along in recovery by normalizing the data to days since outbreak in each state, and training a random forest model on the data and using it to assess the most effective areas of social distancing and make predictions on the future.

##  2. <a name='DataCleaning'></a>Data + Cleaning

I used 4 different datasets for this study and combined them into a single DataFrame for analysis and prediction.

New York Times: Github Repo of cases/deaths daily per state
The New York Times offers dataset on the number of cases and deaths by COVID-19 per each state. I created a new column of daily new cases and divided these numbers by the state's population for each state for a fairer comparison from state to state, resulting in new cases per 1 million residents. To smooth out the many spikes in the number of new cases per day, I used a 7 day moving average and used this as the target variable.
<p>
<img src="images/Covid_Data.png" width="400">
<p>
Apple: Mobility Data
Apple offers a dataset on mobility that breaks categories into walking, driving, and transit. Unfortunately, walking and transit data are only available on a national and/or city level so I was only able to get driving data from this set. These data are reported as compared to a percentage of the baseline value measured on January 13th - this number was converted by 100 to get a multiplier of normal for each day. A 7 day moving average set 7-10 days in the past was then applied on this data.
<p>
<img src="images/AppleData.png" width="1000">
<p>

Google: Global Mobility Data
From Google, I was able to get a massive dataset detailing mobility trends througout the past few months at grocery stores/ pharmacies, parks, transit stations, retail/recreation outlets, residential, and workplaces. This data was reported as the percent change from the baseline, the median value, for the corresponding day of the week, during the 5-week period Jan 3–Feb 6, 2020. To make this measurement consistent with the apple data, I added 100 and divided by 100 to get a multiplier of normal for each day. I also used a 7-10 day delayed 7 day moving average for these features as well, so that weekend trends were avoided and any correlation would be easier to see.
<p>
<img src="images/GoogleData.png" width="1000">
<p>

World Population Review
I obtained state population density from the world population review and implemented it into my dataframe in order to create subsets states similar in population density.

<p>
<img src="images/flowchart.png" width="600">
<p>

Each of the preceeding datasets were combined into one Pandas DataFrame and cleaned. For missing mobility/social distancing entries, the average of the next surrounding values was used. A 21 day time lagged series of the target value (Daily New Cases) was then applied. In total, the dataset consistented of 30 features: 21 time lagged daily new cases, days since outbreak, 7 social distancing features, and state population density.

In order to determine the set of states to create predictions for a specific state, the model will take a subset of states with similar population densities that are further ahead in recovering from the virus. The date values are then normalized so that that daily new cases curves were as closely aligned as possible by defining a start of the outbreak as the time on which the daily number of new cases reached a certain percentage (subjective based on the subset of states selected, and model performance). A random forest is trained on the dataset for this subset of states, and then the random forest model is applied to original state to make predictions and draw insights for the specific state. 

In predicting future values, the high and low public activity levels are determined by the minimum and maximum levels of social distancing, respectively, found in the training set of similar states.

Please note that this model has two assumptions:
1. Medical knowledge/awareness of virus is correlated with time since first infections
2. Population distribution throughout each individual state is uniform (for simplicity - this project could be reworked on a county by county basis that would likely make a uniform population distribution more feasible)

Please see the Model Results and Forecasting per State section for examples of this model on several different states.

##  3. <a name='ExploratoryDataAnalysis'></a>Exploratory Data Analysis
Although there has been news about a shortage of tests being available for the virus in the USA, the data show a very heavy correlation between deaths and new cases, so I decided to focus on new cases instead of deaths, as there is data earlier and a greater amount of data available for cases. 
<p>
<img src="images/DeathVsCases.png" width="800">
<p>

Since COVID-19 hit the state of New York first, and the state is currently showing strong signs of recovery, I used New York as the focus of my preliminary EDA. I sorted the data available by the maximum number of daily cases and plotted them. Shown below are the states that have had the highest number of new cases in the United States before a moving average was applied.

<p>
<img src="images/Top5States.png">
<p>

Shown below is an early random forest model based on the state of New York; since it has shown a strong negative trend, states that show this kind of pattern are helpful to train models to help similar states who are currently at their maximum number of cases.

<p>
<img src="images/NY_New_100.png">
<p>
*Please note that the figure above represents the data available as of early May 2020.

Looking at plots of New Cases versus the amount of public activity before the social distancing moving average delay parameter was set, there seems to be a surprising negative trend between social distancing attributes and daily new cases per population. There seems to be a positive rate of of new cases to presence at home, which is also the opposite of what I was expecting.
<p>
<img src="images/CasesperActivit.png">
<p>

To get a better visual of how each mobility trend may relate to the new number of cases each day, I scaled each feature data point to a fraction of its maximum value in the interval closest to the peak of the plot. It definitely appears as though a decrease in activity to various venues is followed by a decrease in new cases. Something noteable I found is that there are several spikes in outside activity from mid April and a temporary increase in new cases a few days later. It definitely appears that social distancing affects the number of new cases, but with a time lag, which is likely why these trends were not as apparent on the scatter matrices. Based on my research, it takes 2-14 days for symptoms of COVID-19 to develop after infection; this figure seems consistent with this, hence the reasoning for 7-10 day in delay of social distancing parameters from their current day (this parameter was modified in several models to adjust performance).

Something notable in this plot is there seems to be an explosion in activity in the later portion of the data, concurrent with a sharp decrease in the number of new cases. Looking at previous trends, I don't think converging to 0 anytime soon is going to be very likely.

<p>
<img src="images/NY_Social_Distance_days.png">
<p>

After collecting information for New York, I decided to investigate states that are further behind in recovery. I can use information from recovered states, such as New York, to train a model and come up with insights for states currently in the peak of outbreak. This model will aid in coming up with input as to where social distancing efforts should be focused, and how much.

The bar graph below illustrates the extent of recovery for the least recovered states in the country (as of June 14, 2020).
<p>
<img src="images/State_Recovery.png">
<p>

* Recovery factor is defined as the number of maximum new cases divided by the most recent number of cases for that specific state.

The states with a recovery factor of 1 are experiencing more new cases per person than ever before. By training a random forest model of a subset of states that are similar in population density as these states and futher along in recovery, I can extract insights that could help theses states with the most effective features that are likely to reduce the number of daily new cases in the future.

##  4. <a name='ModelResultsandForecastingPerState'></a>Model Results and Forecasting Per State

The criteria for states to predict and draw insights for are as follows:
- Low recovery factor
- There are other states with similar population densities
- These states have higher rates of recovery

The following states that meet this criteria best as of June 14, 2020 are Alabama, Arizona, Arkansas, California, Mississippi, North Carolina, and Oklahoma. These states are analyzed below.

####  4.1. <a name='Alabama'></a>Alabama

The first step of the model is to select a subset of states with similar population densities as Alabama. With a minimum recovery factor of 1.7 (1.7 times as many new cases were recorded at the maximum as the most recent reporting - moving average applied) and a population density within plus or minus 30 people per square mile, the resulting states are plotted below.
<p>
<img src="images/Alabamasimilarplots.png" width="1000">
<p>

The curves shown peak in different places, and resemble a variety of different shapes. In order to assess how these states from the training sets practiced social distancing to decrease the number of new cases, the data needs to be normalized to get the peaks to overlap as much as possible, and then determine the number of days since the outbreak reached a certain percentage of maximum daily cases.

The normalized plot for states similar to Alabama is shown below. To create this model, the following parameters were used for this model were used in order to get the curves aligned to one another as much as possible: a defined starting of outbreak at 65% of the maximum cases, and a train/test split at 35% of the data available after outbreak start

<p>
<img src="images/Alabamanormalized.png" width="1000">
<p>

Once the model has a time series lag implemented on it, the time lagged subset of similar states can be trained on a Random Forest Regressor model. The time lagged DataFrame after a certain day for the Alabama dataset was reset to remove any leakage of known values to similate predicting the future, but the social distancing parameters were untouched (so that they can be used to test the validity for predicting the future number of new cases). The random forest model trained on the subset of similar states was then applied on the Alabama dataset to create predictions on the known values, and then predictions based on these predictions were applied into the dataframe and applied to the time lagged values for the rest of the days in the dataset. 

By tweaking the social distance delay to 7 days and modifying the aforementioned train/test split and outbreak start definition, the simulated predictions and the actual observed values are shown below.

<p>
<img src="images/Alabamavalidity.png" width="1000">
<p>

The model underpredicts the actual observed value, but the model does seem to catch on to a positive trend based on how people in Alabama are social distancing.

Based on the random forest model, the most important features are shown below.
<p>
<img src="images/Alabamafeatures.png" width="1000">
<p>

To determine the individual effect of these features, individual conditional expectation (ICE) plots are generated below. The fractional levels of public activity relative to pre-pandemic levels (January 2020) are plotted against the esimated number of daily new cases with all other factors held constant. The most important feature was deemed to be grocery and pharmacy activity, but the significant downward trend at higher daily cases is likely due to people reducing social distancing in Lousiana after new cases started dropping. Transit stations and workplaces are the next most important features.
<p>
<img src="images/AlabamaICE.png" width="1000">
<p>

Based on the feature importances and the ICE plots, my recommendations for the states of Alabama to reduce the number of new cases would be to limit the capacity of transit stations to 65% of occupancy relative to pre-pandemic levels until the number of daily new cases goes down to 75 per 1 million residents, at which point this may be relaxed slightly. Based on the model, activity at workplaces should be limited to 58% of pre-pandemic occupancy at least until the number of daily cases decreases to 100 per 1 million citizens or less.

Shown below is a table of the maximum and minimum levels of social distancing found in the training set for Alabama as a percentage of pre-pandemic levels (January 2020).
|                     |Retail/ Recreation %|Grocery/ Pharmacy %|Parks %|Transit Stations %|Workplaces %|Residential %|Driving %|
|---------------------|-------------------|------------------|-------|------------------|------------|-------------|---------|
|Low Public Activity  |44.3               |68.1              |75.6   |39                |46.6        |121          |44       |
|High Public Activity |109                |117.1             |312.7  |108               |103.1       |99.1         |152.4    |

Projecting this model out in the future using these values shows that it is likely Alabama will see a small decrease in the number of new cases with maintained maximized social distancing.

<p>
<img src="images/Alabamafuture.png" width="1000">
<p>

####  4.2. <a name='Arizona'></a>Arizona

Implementing the same model for Arizona results in similar states plotted below:
<p>
<img src="images/Arizonasimilarplots.png" width="1000">
<p>

<p>
<img src="images/Arizonanormalized.png" width="1000">
<p>
The model seems to catch on to the positive trend in Arizona, though the fit is not great.
<p>
<img src="images/Arizonavalidity.png" width="1000">
<p>
The top features for Arizona are driving, parks, and grocery/pharmacy.
<p>
<img src="images/Arizonafeatures.png" width="1000">
<p>

Based on ICE plots, it appears that driving is more correlated with high daily numbers of cases per day when more than 0.6 times as many people are out on the road as normal. If possible, based on this model, the citizens of Arizona should be discouraged to drive more than 60% of pre-pandemic levels. and park attendance should be limited to below 1.25 times as much as January 2020 levels. As for grocery stores and pharmacies, capacity should be limited to about 88% as much as normal.

<p>
<img src="images/ArizonaICE.png" width="1000">
<p>
Shown below is a table of the maximum and minimum levels of social distancing found in the training set for Arizona as a percentage of pre-pandemic levels (January 2020).

|                     |Retail/ Recreation %|Grocery/ Pharmacy %|Parks %|Transit Stations %|Workplaces %|Residential %|Driving %|
|---------------------|-------------------|------------------|-------|------------------|------------|-------------|---------|
|Low Public Activity  |44.3               |68.1              | 63.7  |39.0              |46.6        |121.0        |44.0     |
|High Public Activity |101.3              |120.4             |293.3  |102.4             |79.0        |105.9        |154.1    |

Forecasting these levels into the near future, the results for Arizona show a considerable effect in the reduction of new cases after just a few days.

<p>
<img src="images/Arizonafuture.png" width="1000">
<p>

####  4.3. <a name='Arkansas'></a>Arkansas

When the model is applied to the state of Arkansas, the results are shown below.
<p>
<img src="images/Arkansassimilarplots.png" width="1000">
<p>

<p>
<img src="images/Arkansasnormalized.png" width="1000">
<p>

The model performance plot shows a strong ability for the model to capture the trend in the data with a small extent of underpredicting.
<p>
<img src="images/Arkansasvalidity.png" width="1000">
<p>
Based on feature importances, the level of activity in parks is significantly correlated the estimated number of new cases, followed by grocery stores and pharmacies.

<p>
<img src="images/Arkansasfeatures.png" width="1000">
<p>

The ICE plots for estimated fractional levels of public activity relative to pre-pandemic levels (January 2020) are plotted with all other factors held constant are shown below. There appears to be a significant increase the number of new cases once the public activity in parks exceeds 1.7 times as much as pre-pandemic levels, so at this time limiting park attendance to this level for the state will likely prevent the spread of the disease. Based on this model, Grocery stores and pharmacies should be limited to about 0.9 times as much as pre-pandemic levels until the number of new daily cases goes down to about 50 cases per 1 million residents, after which they can be relaxed.

<p>
<img src="images/ArkansasICE.png" width="1000">
<p>
Shown below is a table of the maximum and minimum levels of social distancing found in the training set for Arkansas as a percentage of pre-pandemic levels (January 2020).

|                     |Retail/ Recreation %|Grocery/ Pharmacy %|Parks %|Transit Stations %|Workplaces %|Residential %|Driving %|
|---------------------|-------------------|------------------|-------|------------------|------------|-------------|---------|
|Low Public Activity  |44.3               |68.1              | 63.7  |39.0              |46.6        |121.0        |44.0     |
|High Public Activity |101.3              |116.0             |249.9  |108.0             |79.1        |105.6        |151.1    |

Forecasting these values out into the future for Arkansas, after a few days it looks like increased social distancing would have a considerable effect.

<p>
<img src="images/Arkansasfuture.png" width="1000">
<p>

####  4.4. <a name='California'></a>California

The findings for California are shown below.
<p>
<img src="images/Californiasimilarplots.png" width="1000">
<p>

Since that states similar to California show peaks that are much wider, an increased train/test split is used to capture as much of trends driving the curve up and down as possible.

<p>
<img src="images/Californianormalized.png" width="1000">
<p>

The model does a good job predicting future values.

<p>
<img src="images/Californiavalidity.png" width="1000">
<p>

The public activity in parks also seems to be a very import feature for California, followed by activity in workplaces.
<p>
<img src="images/Californiafeatures.png" width="1000">
<p>

Based on the ICE plots below, keeping park attendence below 0.9 times as much as normal levels will likely reduce the number of new cases in the future. Encouraging employees to work from home if possible and taking precautions in workplaces to keep attendence below about 0.5 times as much as normal until new cases go below 50 per 1 million California residents also seems to be a good approach for California.

<p>
<img src="images/CaliforniaICE.png" width="1000">
<p>
Shown below is a table of the maximum and minimum levels of social distancing found in the training set for California as a percentage of pre-pandemic levels (January 2020).

|                     |Retail/ Recreation %|Grocery/ Pharmacy %|Parks %|Transit Stations %|Workplaces %|Residential %|Driving %|
|---------------------|-------------------|------------------|-------|------------------|------------|-------------|---------|
|Low Public Activity  |45.0               |67.4              |32.4   |28.9              |47.9        |122.1        |29.5     |
|High Public Activity |110.1              |127.3             |266.4  |105.3             |102.7       |98.6         |138.1    |

Increasing social distancing to the maximum level in California appears to have a drastic effect in the number of new cases in the forecasted future, but if things are relaxed right away, no improvement is predicted to occur.
<p>
<img src="images/Californiafuture.png" width="1000">
<p>

####  4.5. <a name='Mississippi'></a>Mississippi
The state of Mississippi is analyzed below.
<p>
<img src="images/Mississippisimilarplots.png" width="1000">
<p>

<p>
<img src="images/Mississippinormalized.png" width="1000">
<p>

The model for Mississippi appears to overpredict the data slightly, but a positive trend is successfully reflected.

<p>
<img src="images/Mississippivalidity.png" width="1000">
<p>

Like previous states analyzed, parks and grocery/pharmacy activity seems to be the top most important features. 
<p>
<img src="images/Mississippifeatures.png" width="1000">
<p>
Looking at the data from the ice curves, there appears to be a significant increase in the number of new cases when the level of activity in parks exceeds about 1.25 times as much as pre-pandemic levels, so park capacity should be limited accordingly. Grocery stores and pharmacies should be limited to 0.9 as much as normal, and it doesn't appear that the social distancing guidelines should be relaxed at any time in the foreseeable future for Mississippi.
<p>
<img src="images/MississippiICE.png" width="1000">
<p>
Shown below is a table of the maximum and minimum levels of social distancing found in the training set for Mississippi as a percentage of pre-pandemic levels (January 2020).

|                     |Retail/ Recreation %|Grocery/ Pharmacy %|Parks %|Transit Stations %|Workplaces %|Residential %|Driving %|
|---------------------|-------------------|------------------|-------|------------------|------------|-------------|---------|
|Low Public Activity  |44.3               |68.1              |63.7   |39.0              |46.6        |121.0        |44.0     |
|High Public Activity |101.3              |120.4             |293.3  |108.0             |78.9        |105.6        |154.1    |

Forecasting into the future, it looks like the number of new cases is only going to increase if social distancing is relaxed. However, the curve can be flattened if it is maintained.
<p>
<img src="images/Mississippifuture.png" width="1000">
<p>

####  4.6. <a name='NorthCarolina'></a>North Carolina

Shown below are the results when the model is applied to North Carolina.
<p>
<img src="images/North_Carolinasimilarplots.png" width="1000">
<p>

<p>
<img src="images/North_Carolinanormalized.png" width="1000">
<p>

The model does an excellent job predicting the data for North Carolina.

<p>
<img src="images/North_Carolinavalidity.png" width="1000">
<p>

Activity in parks, driving, and people staying at home seem to be the most important features for this state (Population density isn't something that can be changed so the focus will be on the next most important features).

<p>
<img src="images/North_Carolinafeatures.png" width="1000">
<p>

Based on the ICE plots, now that North Carolina has exceed 100 new daily cases per 1 million residents, park attendance should be limited to about 0.7 times as much as their typical capacity. Once tumber of new daily cases goes below 100, they may be relaxed slightly. As far as driving, discouraging citizens to drive more than 0.65 times as much as normal could reduce the number of new cases. Finally, encouraging people to stay home 1.18 times as much as normal will likely help as well, but this could be relaxed once the number of cases gets down to about 75 new daily cases per 1 million residents.

<p>
<img src="images/North_CarolinaICE.png" width="1000">
<p>
Shown below is a table of the maximum and minimum levels of social distancing found in the training set for North Carolina as a percentage of pre-pandemic levels (January 2020).

|                     |Retail/ Recreation %|Grocery/ Pharmacy %|Parks %|Transit Stations %|Workplaces %|Residential %|Driving %|
|---------------------|-------------------|------------------|-------|------------------|------------|-------------|---------|
|Low Public Activity  |45.0               |67.4              |32.4   |28.9              |48.7        |122.1        |29.5     |
|High Public Activity |110.1              |127.3             |169.0  |105.3             |102.7       |98.6         |120.9    |

Unfortunately, it looks like the number of new cases in North Carolina is going to keep climbing for the foreseeable future according to this model. However, the increase is smaller with low public activity.
<p>
<img src="images/North_Carolinafuture.png" width="1000">
<p>

####  4.7. <a name='Oklahoma'></a>Oklahoma
Oklahoma is analyzed below.
<p>
<img src="images/Oklahomasimilarplots.png" width="1000">
<p>

<p>
<img src="images/Oklahomanormalized.png" width="1000">
<p>
The model appears to fit the data for Oklahoma very well.
<p>
<img src="images/Oklahomavalidity.png" width="1000">
<p>

Activity in parks and people staying home seem to have the greatest correlation with the number of new cases per day.
<p>
<img src="images/Oklahomafeatures.png" width="1000">
<p>

Drawing insights from the ICE plots for Oklahoma for the most important featues, my recommendations are as follows. Parks should be limited to about 1.3 times as much as January 2020 levels (or 1.6 if that is not possible). Once the number of daily new cases hits 70 per 1 million residents, citizens of Oklahoma should be urged to stay at home 1.13 times as much as January 2020 levels. 
<p>
<img src="images/OklahomaICE.png" width="1000">
<p>
Shown below is a table of the maximum and minimum levels of social distancing found in the training set for Oklahoma as a percentage of pre-pandemic levels (January 2020).

|                     |Retail/ Recreation %|Grocery/ Pharmacy %|Parks %|Transit Stations %|Workplaces %|Residential %|Driving %|
|---------------------|-------------------|------------------|-------|------------------|------------|-------------|---------|
|Low Public Activity  |44.3               |68.1              |63.7   |39.0              |46.6        |121.0        |44.0     |
|High Public Activity |101.3              |120.4             |293.3  |108.0             |78.9        |105.6        |154.1    |

Looking into the future, it appears that COVID-19 is currently on an exponential rise in Oklahoma, and there is a massive surge predicted in the number of new cases. However, the damage can be significantly reduced by limiting public activity during this critical time.
<p>
<img src="images/Oklahomafuture.png" width="1000">
<p>

##  5. <a name='Conclusion'></a>Conclusion
Unfortunately, it does not look like the occurrence of new cases of COVID-19 are going away anytime soon, though some states are much closer than others. Although everyone is eager to finally get out of the house after so long, it looks like social distancing is effective in preventing the spread of the virus with a time delay, and hopefully we will continue to see numbers go down further in the near future in the states currently struggling most with the virus, and a second wave of infections will be avoided. However, even if there is a second wave in infections, there is data that can be used to determine the extent of how social distancing should be maintained and the amount of an effect that it can have on the future number of cases.

##  6. <a name='FutureIdeasforImprovement'></a>Future Ideas for Improvement
- Replace states with counties and run county by county analysis (better results for varying population densities)
- Apply a scalar value to multiply the x axis on specific curves so that the shapes more closely resemble each other to improve results
- Pursue original goal of predicting economic recovery; forecast recovery out further and use recession unemployment data to forecast
- Import more features from other data sources:
    - Look at South Korea/Italy data or other recovered countries - use that to predict into future?
    - Bring in data from travel from other countries - maybe that would help?
        https://travel.trade.gov/view/m-2017-I-001/index.asp
    - Weather data - Rumors that the virus doesn't do well in warm/humid conditions
- Use web-scraping and NLP to analyze the type of words being used in recent economic news (ratio of words such as "lockdown", "quarantine", "extended" to words like "protest", "end", "open") to determine better scaling factors for prediction matrix 

##  7. <a name='DataSources'></a>Data Sources

Header image : https://spectrumlocalnews.com/nc/triangle-sandhills/tying-it-together-with-tim-boyum/2020/04/28/tying-it-together-with-tim-boyum-checking-the-pulse-on-nc-s-economic-health-during-the-coronavirus-crisis
https://www.xifin.com/resources/blog/202004/covid-19-maintaining-momentum-during-volume-recessions 

https://github.com/nytimes/covid-19-data/blob/master/us-states.csv 

https://worldpopulationreview.com/states/

https://www.apple.com/covid19/mobility

https://www.google.com/covid19/mobility/index.html?hl=en 