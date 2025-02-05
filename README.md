# House-Price-Prediction-Flask
The following repository contains a house price prediction model trained using sklearn and deployed with the help of flask API. The dataset used for this project is the california housing dataset provided in the sklearn.datasets library.It contains the following parts:

Setup your environment
Build your house price prediction model
Create flask API

## Dataset
The dataset was taken from the sklearn.datasets library. Following is the description of the dataset as per sklearn documentation:

:Number of Instances: 20640

:Number of Attributes: 8 numeric, predictive attributes and the target

:Attribute Information:
    - MedInc        median income in block group
    - HouseAge      median house age in block group
    - AveRooms      average number of rooms per household
    - AveBedrms     average number of bedrooms per household
    - Population    block group population
    - AveOccup      average number of household members
    - Latitude      block group latitude
    - Longitude     block group longitude

:Missing Attribute Values: None

This dataset was obtained from the StatLib repository.
https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html

The target variable is the median house value for California districts,
expressed in hundreds of thousands of dollars ($100,000).

This dataset was derived from the 1990 U.S. census, using one row per census
block group. A block group is the smallest geographical unit for which the U.S.
Census Bureau publishes sample data (a block group typically has a population
of 600 to 3,000 people).

A household is a group of people residing within a home. Since the average
number of rooms and bedrooms in this dataset are provided per household, these
columns may take surprisingly large values for block groups with few households
and many empty houses, such as vacation resorts.

It can be downloaded/loaded using the
:func:`sklearn.datasets.fetch_california_housing` function.

.. topic:: References

    - Pace, R. Kelley and Ronald Barry, Sparse Spatial Autoregressions,
      Statistics and Probability Letters, 33 (1997) 291-297


## Building Model
The model used on this dataset was the RandomForestRegressor. The following are the metrics along with the best model found using GridSearchCV: 
RandomForestRegressor(max_depth=20, max_features='sqrt', n_estimators=200)
R square :  0.8138637284893164
Mean squared error :  0.2439146413638855
Mean absolute error :  0.32823864287321625

## Create Flask API

Flask is a web framework for python. It is used for managing HTTP request and render templates. Now, start to create basic flask API (for example app.py). 

Pickle load function is used to load the model data in flask API. App route('/') map to home page('page.html') and App route(/predict) map to predict function, it is also call in home page. In predict function get the values from the form and then used them for model prediction.

