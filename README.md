<p>
<img src="images/covid19_economy_jpg.jpeg" width="500">
<p>

#Capstone-2
End goal - estimate Covid 19 under control, things go back to normal
Question: Based on previous recessions in US history and/or containment time from affected countries, when will the United States recover from the economic hardships brought on by the pandemic? I will be using the unemployment rate as an indicator of economic health.

MVP: 
Part 1. Predict when outbreak is contained to the point of full lift of lockdown/quarantine. SInce there is a lot of instability in this estimate due to protests and noncompliance with stay at home orders, I plan to take a range of predictions and analyze a best and worst case scenario.
I plan on using the stats from NY Times on Github for the data - the level of new cases seems to have levelled off, I will try a few regression models to hopefully see a decrease. https://github.com/nytimes/covid-19-data/blob/master/us-states.csv

MVP+:
2. I then plan to split the unemployment data into two sections: Economic expansions and recessions to form two models. I will use the recession model along with the data since the outbreak to assume an ongoing recession and increase in unemployment until the estimated containment date from part 1. Then, I will assume the unemployment rate will proceed to go back down per the expansion/recovery model. Then I will forecast this model into the future until a viable unemployment rate (3.5-4.5 % based on research).

MVP ++:​ Use web-scraping and NLP to analyze the type of words being used in recent economic news (ratio of words such as "lockdown", "quarantine", "extended" to words like "protest", "end", "open") to determine a scaling factor between best and worst cases for part 1 model


Wednesday:

Thursday:
Finalize


Sources:

Header image : https://spectrumlocalnews.com/nc/triangle-sandhills/tying-it-together-with-tim-boyum/2020/04/28/tying-it-together-with-tim-boyum-checking-the-pulse-on-nc-s-economic-health-during-the-coronavirus-crisis

https://github.com/nytimes/covid-19-data/blob/master/us-states.csv 

https://worldpopulationreview.com/states/

https://worldpopulationreview.com/states/states-by-area/