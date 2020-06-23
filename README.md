<p>
<img src="images/covid19_economy_jpg.jpeg" width="600">
<p>

# Forecasting the Future of COVID-19 with Social Distancing 
The question on everyone's mind these days seems to be when will things go back to normal? The United States, as well as the rest of the world, has been hit hard by the COVID-19 pandemic and we are all hoping that the end is in sight. With the implementation of social distancing playing a massive role in our lives today, what level of social distancing should be maintained so that we see a consistent drop in the number of new cases of the virus? 

In this project, I have created a model that will predict a forecast of daily new cases per capita for a state currently experiencing the maximum number of cases with different levels of social distancing by drawing a subset of states similar in population density that are further along in recovery by normalizing the data to days since outbreak in each state, and training a random forest model on their data and using it to make predictions on the future.

## Data + Cleaning

I used 4 different datasets for this study and combined them into a single DataFrame for analysis and prediction.

New York Times: Github Repo of cases/deaths daily per state
The New York Times offers dataset on the number of cases and deaths by COVID-19 per each state. I created a new column of daily new cases and divided these numbers by the state's population for each state for a fairer comparison from state to state, resulting in new cases per 1 million residents. To smooth out the many spikes in the number of new cases per day, I used a 7 day moving average and used this as the target variable.
<p>
<img src="images/Covid_Data.png" width="400">
<p>
Apple: Mobility Data
Apple offers a dataset on mobility that breaks categories into walking, driving, and transit. Unfortunately, walking and transit data are only available on a national and/or city level so I was only able to get driving data from this set. These data are reported as compared to a percentage of the baseline value measured on January 13th - this number was converted by 100 to get a multiplier of normal for each day.
<p>
<img src="images/AppleData.png" width="1000">
<p>

Google: Global Mobility Data
From Google, I was able to get a massive dataset detailing mobility trends througout the past few months at grocery stores/ pharmacies, parks, transit stations, retail/recreation outlets, residential, and workplaces. This data was reported as the percent change from the baseline, the median value, for the corresponding day of the week, during the 5-week period Jan 3–Feb 6, 2020. To make this measurement consistent with the apple data, I added 100 and divided by 100 to get a multiplier of normal for each day. I also used a 7 day moving average for these features as well, so that weekend trends were avoided and any correlation would be easier to see.
<p>
<img src="images/GoogleData.png" width="1000">
<p>

World Population Review
I obtained state population density from the world population review and implemented it into my dataframe in order to create subsets states similar in population density.

<p>
<img src="images/flowchart.png" width="600">
<p>

## Exploratory Data Analysis
Although there has been news about a shortage of tests being available for the virus in the USA, the data show a very heavy correlation between deaths and new cases, so I decided to focus on new cases instead of deaths, as there is data earlier and a greater amount of data available for cases. 
<p>
<img src="images/DeathVsCases.png" width="800">
<p>

Since COVID-19 hit the state of New York first, and the state is currently showing strong signs of recovery, I used New York as the focus of my preliminary EDA. I sorted the data available by the maximum number of daily cases and plotted them. Shown below are the states tha have had the highest number of new cases in the United States.

<p>
<img src="images/Top5States.png">
<p>

When I plotted the data, I noticed that there are numerous spikes in the data whereas the predicted trend should be based on the moving average. Considering this dataset covers a considerable amount of time and data before the outbreak, I needed to made sure not to use any training data below a certain threshold of new daily cases per 1 million population for each subset of states. This is because this data features no new cases and no changes in social distancing. Using the data below the threshold could mislead the model. 

*Please note that the figure below represents the data available and model as of early May 2020.
<p>
<img src="images/NY_New_100.png">
<p>

Looking at plots of New Cases versus the amount of public activity, there seems to be a surprising negative trend between social distancing attributes and daily new cases per population. There seems to be a positive rate of of new cases to presence at home, which is also the opposite of what I was expecting.
<p>
<img src="images/CasesperActivit.png">
<p>

To get a better visual of how each mobility trend may relate to the new number of cases each day, I scaled each feature data point to a fraction of its maximum value in the interval closest to the peak of the plot. It definitely appears as though a decrease in activity to various venues is followed by a decrease in new cases. Something noteable I found is that there are several spikes in outside activity from mid April and a temporary increase in new cases a few days later. It definitely appears that social distancing effects the number of new cases, but with a time lag, which is likely why these trends were not as apparent on the scatter matrices.

Something notable is that there seems to be an explosion in activity in the later portion of the data, concurrent with a sharp decrease in the number of new cases. Looking at previous trends, I don't think converging to 0 anytime soon is going to be very likely.

<p>
<img src="images/NY_Social_Distance_days.png">
<p>

After collecting information for New York, I decided to investigate states that are further behind in recovery. I can use information from recovered states, such as New York, to train a model and come up with insights for states currently in the peak of outbreak. This model will aid in coming up with input as to where social distancing efforts should be focused, and how much.

The bar graph below illustrates the extent of recovery for the least recovered states in the country.
<p>
<img src="images/State_Recovery.png">
<p>

*Recovery factor is defined as the number of maximum new cases divided by the most recent number of cases for that specific state.

The states with a recovery factor of 1 are experiencing more new cases per person than ever before. By training a random forest model of a subset of states that are similar in population density as these states and futher along in recovery, I can extract insights that could help theses states with the most effective features that are likely to reduce the number of daily new cases in the future.

## Model Results and Forecasting per State

#### North Carolina
<p>
<img src="images/North_Carolinasimilarplots.png" width="1000">
<p>

<p>
<img src="images/North_Carolinanormalized.png" width="1000">
<p>

<p>
<img src="images/North_Carolinavalidity.png" width="1000">
<p>

<p>
<img src="images/North_Carolinafuture.png" width="1000">
<p>

#### North Carolina

## Forecasting
In order to set up a regression model and predict future values, I converted my original dataframe into a time series matrix and I decided to use a prediction of 21 days using the moving average data points as this seemed like a large enough interval to cover any cause-effects between the features and the target that may be lagging behind. The time series ended up with about 188 features in all, so I decided to use a random forest on my model as it can support high dimensionality with high accuracy.

Feature importances were determined below by summing all previous time series individual feature importances for the 20 day time lag into each specific category. 
<p>
<img src="images/feature_importances.png">
<p>

I split the data into a training and testing group. Training data was used from the threshold to day 70, and Test Data was used from day 70 to the last known point.

Looking at the performance of the model on the testing set, it doesn't appear to be a great fit. Of course, it is very difficult to model this, and given the large shift in activity from the later part of the dataset, it makes sense that the model would predict a surge in the number of daily new cases.

<p>
<img src="images/RF_Performance.png">
<p>

I used my model to predict values out 20 days from the last known data point, this brings us to around the end of May. Using a prediction matrix with different set levels of activity, it does look like a high degree of social distancing and staying home will reduce the number of cases, but we still a long way from 0. 

<p>
<img src="images/Predictions.png">
<p>

## Conclusion
Unfortunately, it does not look like the occurrence of new cases of COVID-19 are going away anytime soon. Although everyone is eager to finally get out of the house after so long, it looks like social distancing is effective in preventing the spread of the virus, with a time delay, and hopefully we will continue to see numbers go down further in the near future.

## Future Plans/Ideas for improvement
- Replace states with counties and run county by county analysis (better results for varying population densities)
- Provide sample of X and y matrices
- Pursue original goal of predicting economic recovery; forecast recovery out further and use recession unemployment data to forecast
- Forecast out futher; hopefully will be able to pinpoint a recovery date range
- Consider trying other regression models
- Import more features from other data sources:
    - Look at South Korea data - use that to predict into future?
    - Bring in data from travel from other countries - maybe that would help?
        https://travel.trade.gov/view/m-2017-I-001/index.asp
    - Weather data - Rumors that the virus doesn't do well in warm/humid conditions
- Use web-scraping and NLP to analyze the type of words being used in recent economic news (ratio of words such as "lockdown", "quarantine", "extended" to words like "protest", "end", "open") to determine better scaling factors for my prediction matrix. 

## Data Sources:

Header image : https://spectrumlocalnews.com/nc/triangle-sandhills/tying-it-together-with-tim-boyum/2020/04/28/tying-it-together-with-tim-boyum-checking-the-pulse-on-nc-s-economic-health-during-the-coronavirus-crisis
https://www.xifin.com/resources/blog/202004/covid-19-maintaining-momentum-during-volume-recessions 

https://github.com/nytimes/covid-19-data/blob/master/us-states.csv 

https://worldpopulationreview.com/states/

https://www.apple.com/covid19/mobility

https://www.google.com/covid19/mobility/index.html?hl=en 