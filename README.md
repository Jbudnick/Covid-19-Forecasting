<p>
<img src="images/covid19_economy_jpg.jpeg" width="600">
<p>

# Forecasting the Future of COVID-19 with Social Distancing 

## Table of Contents
<!-- vscode-markdown-toc -->
* 1. [Data + Cleaning](#DataCleaning)
* 2. [Exploratory Data Analysis](#ExploratoryDataAnalysis)
* 3. [Model Results and Forecasting per State](#ModelResultsandForecastingperState)
		* 3.1. [Alabama](#Alabama)
		* 3.2. [Arizona](#Arizona)
		* 3.3. [Arkansas](#Arkansas)
		* 3.4. [California](#California)
		* 3.5. [Mississippi](#Mississippi)
		* 3.6. [North Carolina](#NorthCarolina)
		* 3.7. [Oklahoma](#Oklahoma)
* 4. [Conclusion](#Conclusion)
* 5. [Future Ideas for Improvement](#FutureIdeasforImprovement)
* 6. [Data Sources](#DataSources:)

<!-- vscode-markdown-toc-config
	numbering=true
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->

The question on everyone's mind these days seems to be the same: when will things go back to normal? The United States has been hit especially hard by the COVID-19 pandemic and we are all hoping that the end is in sight. With the implementation of social distancing playing a massive role in our lives today, what level of social distancing should be maintained so that we see a consistent drop in the number of new cases of the virus? 

In this project, I have created a model that will predict a forecast of daily new cases per capita for a state currently experiencing the maximum number of cases with different levels of social distancing by drawing a subset of states similar in population density that are further along in recovery by normalizing the data to days since outbreak in each state, and training a random forest model on the data and using it to assess the most effective areas of social distancing and make predictions on the future.

##  1. <a name='DataCleaning'></a>Data + Cleaning

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

##  2. <a name='ExploratoryDataAnalysis'></a>Exploratory Data Analysis
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
*Please note that the figure below represents the data available as of early May 2020.

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

##  3. <a name='ModelResultsandForecastingperState'></a>Model Results and Forecasting per State

The criteria for states to predict and draw insights for are as follows:
- Low recovery factor
- There are other states with similar population densities
- These states have higher rates of recovery

The following states that meet this criteria best as of June 14, 2020 are Alabama, Arizona, Arkansas, California, Mississippi, North Carolina, and Oklahoma. These states are analyzed below.

####  3.1. <a name='Alabama'></a>Alabama

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

The estimated fractional levels of public activity relative to pre-pandemic levels (January 2020) are plotted with all other factors held constant are shown.
<p>
<img src="images/Alabamapart_dep.png" width="1000">
<p>

<p>
<img src="images/AlabamaICE.png" width="1000">
<p>

Based on the feature importances, my recommendations for the states of Alabama would be to:

Shown below is a table of the maximum and minimum levels of social distancing found in the training set for Alabama as a percentage of pre-pandemic levels (January 2020).
|                     |Retail/Recreation %|Grocery/Pharmacy %|Parks %|Transit Stations %|Workplaces %|Residential %|Driving %|
|---------------------|-------------------|------------------|-------|------------------|------------|-------------|---------|
|Low Public Activity  |44.3               |68.1              |75.6   |39                |46.6        |121          |44       |
|High Public Activity |109                |117.1             |312.7  |108               |103.1       |99.1         |152.4    |

Projecting this model out in the future using these values shows that it is likely Alabama will see a small decrease in the number of new cases with maximized social distancing.

<p>
<img src="images/Alabamafuture.png" width="1000">
<p>


####  3.2. <a name='Arizona'></a>Arizona

Implementing the same model for Arizona results in similar states plotted below:
<p>
<img src="images/Arizonasimilarplots.png" width="1000">
<p>

<p>
<img src="images/Arizonanormalized.png" width="1000">
<p>

<p>
<img src="images/Arizonavalidity.png" width="1000">

<p>
<img src="images/Arizonafeatures.png" width="1000">
<p>

<p>
<img src="images/Arizonapart_dep.png" width="1000">
<p>

<p>
<img src="images/ArizonaICE.png" width="1000">
<p>

|                     |Retail/Recreation %|Grocery/Pharmacy %|Parks %|Transit Stations %|Workplaces %|Residential %|Driving %|
|---------------------|-------------------|------------------|-------|------------------|------------|-------------|---------|
|Low Public Activity  |44.3               |68.1              | 63.7  |39.0              |46.6        |121.0        |44.0     |
|High Public Activity |101.3              |120.4             |293.3  |102.4             |79.0        | 105.9       |154.1    |

The results for Arizona are similar - the model slightly underpredicts the later future values but catches on to a positive trend.

<p>
<img src="images/Arizonafuture.png" width="1000">
<p>

####  3.3. <a name='Arkansas'></a>Arkansas
<p>
<img src="images/Arkansassimilarplots.png" width="1000">
<p>

<p>
<img src="images/Arkansasnormalized.png" width="1000">
<p>

<p>
<img src="images/Arkansasvalidity.png" width="1000">
<p>

The model performance plot shows a very strong ability for the model to capture the trend in the data.
<p>
<img src="images/Arkansasfeatures.png" width="1000">
<p>

The estimated fractional levels of public activity relative to pre-pandemic levels (January 2020) are plotted with all other factors held constant are shown.
<p>
<img src="images/Arkansaspart_dep.png" width="1000">
<p>

<p>
<img src="images/ArkansasICE.png" width="1000">
<p>

<p>
<img src="images/Arkansasfuture.png" width="1000">
<p>

Forecasting these values out into the future for Arkansas, after a few days it looks like increased social distancing would have a considerable effect.

####  3.4. <a name='California'></a>California
<p>
<img src="images/Californiasimilarplots.png" width="1000">
<p>

Since that states similar to California show peaks that are much wider, an increased train/test split is used to capture as much of trends driving the curve up and down as possible.

<p>
<img src="images/Californianormalized.png" width="1000">
<p>

The model does a decent job predicting future values.

<p>
<img src="images/Californiavalidity.png" width="1000">
<p>

Increasing social distancing in California appears to have a drastic effect in the number of new cases in the future.
<p>
<img src="images/Californiafeatures.png" width="1000">
<p>

<p>
<img src="images/Californiapart_dep.png" width="1000">
<p>

<p>
<img src="images/CaliforniaICE.png" width="1000">
<p>

<p>
<img src="images/Californiafuture.png" width="1000">
<p>

####  3.5. <a name='Mississippi'></a>Mississippi

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

Forecasting into the future, it looks like the number of new cases is only going to increase if social distancing is relaxed. However, the curve can be flattened if it is maintained.
<p>
<img src="images/Mississippifeatures.png" width="1000">
<p>

<p>
<img src="images/Mississippipart_dep.png" width="1000">
<p>

<p>
<img src="images/MississippiICE.png" width="1000">
<p>

<p>
<img src="images/Mississippifuture.png" width="1000">
<p>

<p>
<img src="images/Mississippifuture.png" width="1000">
<p>

####  3.6. <a name='NorthCarolina'></a>North Carolina
<p>
<img src="images/North_Carolinasimilarplots.png" width="1000">
<p>

<p>
<img src="images/North_Carolinanormalized.png" width="1000">
<p>

The model does a great job predicting the data for North Carolina.

<p>
<img src="images/North_Carolinavalidity.png" width="1000">
<p>

Unfortunately, it looks like the number of new cases in North Carolina is going to keep climbing for the foreseeable future according to this model. However, the increase is much smaller with low public activity.
<p>
<img src="images/North_Carolinafeatures.png" width="1000">
<p>
<p>
<img src="images/North_Carolinapart_dep.png" width="1000">
<p>

<p>
<img src="images/North_CarolinaICE.png" width="1000">
<p>

<p>
<img src="images/North_Carolinafuture.png" width="1000">
<p>

<p>
<img src="images/North_Carolinafuture.png" width="1000">
<p>

####  3.7. <a name='Oklahoma'></a>Oklahoma

<p>
<img src="images/Oklahomasimilarplots.png" width="1000">
<p>

<p>
<img src="images/Oklahomanormalized.png" width="1000">
<p>

<p>
<img src="images/Oklahomavalidity.png" width="1000">
<p>

It appears that COVID-19 is on an exponential rise in Oklahoma, and there is a massive surge predicted in the number of new cases. However, the damage can be reduced by limiting public activity during this time.
<p>
<img src="images/Oklahomafeatures.png" width="1000">
<p>

<p>
<img src="images/Oklahomapart_dep.png" width="1000">
<p>

<p>
<img src="images/OklahomaICE.png" width="1000">
<p>

<p>
<img src="images/Oklahomafuture.png" width="1000">
<p>

##  4. <a name='Conclusion'></a>Conclusion
Unfortunately, it does not look like the occurrence of new cases of COVID-19 are going away anytime soon. Although everyone is eager to finally get out of the house after so long, it looks like social distancing is effective in preventing the spread of the virus, with a time delay, and hopefully we will continue to see numbers go down further in the near future in the states currently struggling most with the virus, and a second wave of infections is avoided.

##  5. <a name='FutureIdeasforImprovement'></a>Future Ideas for Improvement
- Replace states with counties and run county by county analysis (better results for varying population densities)
- Apply a scalar value to multiply the x axis on specific curves so that the shapes more closely resemble each other to improve results
- Pursue original goal of predicting economic recovery; forecast recovery out further and use recession unemployment data to forecast
- Forecast out futher; hopefully will be able to pinpoint a recovery date range
- Consider trying other regression models
- Import more features from other data sources:
    - Look at South Korea data - use that to predict into future?
    - Bring in data from travel from other countries - maybe that would help?
        https://travel.trade.gov/view/m-2017-I-001/index.asp
    - Weather data - Rumors that the virus doesn't do well in warm/humid conditions
- Use web-scraping and NLP to analyze the type of words being used in recent economic news (ratio of words such as "lockdown", "quarantine", "extended" to words like "protest", "end", "open") to determine better scaling factors for my prediction matrix. 

##  6. <a name='DataSources:'></a>Data Sources

Header image : https://spectrumlocalnews.com/nc/triangle-sandhills/tying-it-together-with-tim-boyum/2020/04/28/tying-it-together-with-tim-boyum-checking-the-pulse-on-nc-s-economic-health-during-the-coronavirus-crisis
https://www.xifin.com/resources/blog/202004/covid-19-maintaining-momentum-during-volume-recessions 

https://github.com/nytimes/covid-19-data/blob/master/us-states.csv 

https://worldpopulationreview.com/states/

https://www.apple.com/covid19/mobility

https://www.google.com/covid19/mobility/index.html?hl=en 