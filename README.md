<p>
<img src="images/covid19_economy_jpg.jpeg" width="500">
<p>

#Forecasting the Future of Covid 19 with Social Distancing 


Original goal - estimate Covid 19 under control, things go back to normal
Question: Based on previous recessions in US history and/or containment time from affected countries, when will the United States recover from the economic hardships brought on by the pandemic? 



<p>
<img src="images/Covid-19_New.png" width="500">
<p>

<p>
<img src="images/DeathVsCases.png" width="500">
<p>

<p>
<img src="images/NY_New_100.png" width="500">
<p>

<p>
<img src="images/Scatter100.png"width="700">
<p>

<p>
<img src="images/NY_New_300.png">
<p>

<p>
<img src="images/NY_Social_Distance_days.png">
<p>

<p>
<img src="images/Predictions.png">
<p>

<p>
<img src="images/RF_Performance.png">
<p>

MVP: 
Part 1. Predict when outbreak is contained to the point of full lift of lockdown/quarantine. SInce there is a lot of instability in this estimate due to protests and noncompliance with stay at home orders, I plan to take a range of predictions and analyze a best and worst case scenario.
I plan on using the stats from NY Times on Github for the data - the level of new cases seems to have levelled off, I will try a few regression models to hopefully see a decrease. https://github.com/nytimes/covid-19-data/blob/master/us-states.csv

MVP+:
2. I then plan to split the unemployment data into two sections: Economic expansions and recessions to form two models. I will use the recession model along with the data since the outbreak to assume an ongoing recession and increase in unemployment until the estimated containment date from part 1. Then, I will assume the unemployment rate will proceed to go back down per the expansion/recovery model. Then I will forecast this model into the future until a viable unemployment rate (3.5-4.5 % based on research).

MVP ++:​ Use web-scraping and NLP to analyze the type of words being used in recent economic news (ratio of words such as "lockdown", "quarantine", "extended" to words like "protest", "end", "open") to determine a scaling factor between best and worst cases for part 1 model

#As many features/ data as possible, limit to specific to state, change every state to specific timescale, day 0 for new york
#day for 0 for Arizona, use stacking, different states of US, first data point, death per people, y axis new cases per capita

Do more than random forest, Can't predict future with train_test_split, use only past values, go through time-series lecture
Instead of doing train_test_split, hold out a small percentange of recent values, train with the rest? Still use random forest method?

Look at South Korea data - use that to predict into future?
Bring in data from travel from other countries - maybe that would help?
https://travel.trade.gov/view/m-2017-I-001/index.asp

# Forecasting




Wednesday:

Thursday:
Finalize


Sources:

Header image : https://spectrumlocalnews.com/nc/triangle-sandhills/tying-it-together-with-tim-boyum/2020/04/28/tying-it-together-with-tim-boyum-checking-the-pulse-on-nc-s-economic-health-during-the-coronavirus-crisis

https://github.com/nytimes/covid-19-data/blob/master/us-states.csv 

https://worldpopulationreview.com/states/

https://worldpopulationreview.com/states/states-by-area/