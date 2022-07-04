# soccer_season_prediction
The goal of this project is to predict a soccer final ranking based on what happened until some leg

# Introduction

Measures implemented to contain Covid-19 outbreaks impacted all aspect of life. In particular soccer championship had been forced to pause their season. In France, it meant a final stop after 27 full legs and positions were frozen. The decision was met with outrage by clubs that were not in positions they might have been had the season run to its normal end because it had huge consequences for them; from a sportive point of view as well as a financial one.

As a soccer fan and a data scientist, I saw a great opportunity to combine both field in one new original (at least for me!) project : predicting the final ranking given the course of the championship is known up to some leg.

The project is decomposed in 5 steps :
- getting the data
- exploring it and get some insights
- build different type of models and pick the most interesting
- moving the workflow from notebooks to scripts 
- writing an app with Streamlit.


# Getting the data

(data was scrapped from L'Équipe, a french newspaper focused on sports --> full soccer data available from season 2004-2005)
TBD 

# Exploratory Data Analysis

Evolution of several kpis based upon final rank
TBD 
List the questions

# EDA for 2019-2020 season

TODO

# Models
give a small description of the ideas

### The naive one
TBD 

### Regression type
TBD 

### Classification type
TBD 

### Ranking type
TBD 

### What metrics ?
TBD 

### Results
TBD

# Notebooks to scripts
Scrapping, first analysis and model building were made in Jupyter Notebooks. Once I deemed the project mature enough, I 
moved to Python files and finally to a Streamlit app 

# The app
In order to illustrate the work done in this project, I made a Streamlit app. I designed it as multipage app
so that EDA and ML parts are independents but easy to use.

To start the app, one will have to run the following command : streamlit run soccer_front_app.py
# Miscellaneous
As part of this project, a document where all details from the project can be found has been written.

# Side note
I consider my notebooks as rough drafts before turning to .py script with content I deem sufficient; so notebooks are 
not very user-friendly.

Best practises requires providing docstrings and variable typing. As it is personal and been written over the course of several months (when I had free time and motivation to go on)
and not meant to be used widely by the public, it didn't seem to me to be as important.



## NB : 
https://stackoverflow.com/questions/72032032/importerror-cannot-import-name-iterable-from-collections-in-python