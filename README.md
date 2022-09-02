# soccer_season_prediction
The main goal of this project is to predict a soccer final ranking based on what happened until some leg.

# Introduction

Measures implemented to contain Covid-19 outbreaks impacted all aspect of life. In particular soccer championship had been forced to pause their season. In France, it meant a final stop after 27 full legs and positions were frozen. The decision was met with outrage by clubs that were not in positions they might have been had the season run to its normal end because it had huge consequences for them; from a sportive point of view as well as a financial one.

As a soccer fan and a data scientist, I saw a great opportunity to combine both field in one new original (at least for me!) project : predicting the final ranking given the course of the championship is known up to some leg.

The project is decomposed in 5 steps :
- getting the data
- exploring it and get some insights
- build different type of models and pick the most interesting
- moving the workflow from notebooks to scripts 
- writing an app with Streamlit.

A more detailed description from this project can be found here.

# Getting the data

The data used in this project had been scrapped from the website of "L'Équipe", a French newspaper focused on sports.
In particular, one can find all soccer games results since season 2004-2005 for the main championships
(England, Germany, Italy, Spain and France).

In a second step, data from the 2019-2020 season was gathered and set apart since it is only meant as 'new data'.

# Exploratory Data Analysis

In order to get a sense if the data, this project contains an EDA part where the user can explore the 
evolution of some soccer kpis depending on the final rank.

Initial exploratory analysis was performed in a Jupyter Notebook in order to get a sense of the data and find out the 
best charts.

Now, a user can simply navigate in the app.

PICT

# EDA for 2019-2020 season

Here user can see how teams were performing until the championships were paused due to the first Covid-19 
lockdowns. Hence, the user could guess how teams would have performed under "normal circumstances".

# Models
There are several ways one could try to build a model to predict the final ranking. In this work, I tested my ideas
with 1+3 different approaches : naive, regression on points, classification on points evolutions and ranking design 
algorithms (XGBoost Ranker).

SPECIFY THE MAIN IDEAS

For more details on my choices, readers are encouraged to have a look here.

# Notebooks to scripts
Scrapping, first analysis and model building were made in Jupyter Notebooks. Once I deemed the project mature enough, I 
moved to Python files and finally to a Streamlit app 

# The app
In order to illustrate the work done in this project and allow an user-friendly interface for someone else to explore,
I made a Streamlit app. I designed it as a multipage app so that EDA and ML parts are independents but easy to use.

To start the app, one will have to run the following command : streamlit run soccer_front_app.py


## NB : 
https://stackoverflow.com/questions/72032032/importerror-cannot-import-name-iterable-from-collections-in-python