# Bike Availability Prediction

A machine learning pipeline that predicts the number of available city bikes at a station one hour ahead, built with Python and scikit-learn. This was a course project for INF161 at the University of Bergen, using data from the Bergen city bike system.

## Overview

The pipeline combines three data sources into one model-ready dataset:

* **stations**: the number of free bikes per station over time (the prediction target, `free_bikes`)
* **weather**: temperature, precipitation and wind speed
* **trips**: started and ended bike trips, used to build a traffic volume feature

The project covers data cleaning, exploratory analysis, model selection with time-series-aware validation, and evaluation of how well the model generalises.

## Data preparation

* The data is split chronologically (60/20/20 train, validation, test, no shuffling) before any analysis, to avoid data leakage.
* Data before September 2024 was removed from all sources because of a large gap at the start of the stations file.
* Duplicate observations were removed (some stations showed more than 100 % coverage).
* Missing weather values are filled with forward fill, then backward fill for any gaps at the start of the series.
* Gaps in the station data longer than 4 hours are dropped instead of filled with last observation carried forward, to avoid introducing artificial trends.
* The cleaned dataset has 30,969 observations.

## Features

| Feature | Description |
| --- | --- |
| `free_bikes` | Number of free bikes at the station (current value, also the target one hour ahead) |
| `hour` | Hour of the day |
| `is_weekend` | 1 for weekend, 0 for weekday |
| `flow` | Total trips started and ended in Bergen per time step |
| `temperature`, `precipitation`, `wind_speed` | Weather variables |
| `station_*` | One-hot encoded station |

Removed features: `longitude`, `latitude` and `skipped_updates` (no meaningful contribution), `free_spots` (linearly dependent on `free_bikes`), and `month` (the training set only covers about five months, so the model cannot learn seasonal patterns from it).

## Exploratory findings

* Bike usage follows the working week: rush hour peaks on weekdays and lower activity on weekends.
* Temperature has a weak positive correlation with `free_bikes` (0.342) and a stronger one with trips started (0.437). Precipitation and wind reduce the number of trips (−0.151 and −0.159).

## Modelling

All models were wrapped in scikit-learn pipelines and tuned with `GridSearchCV` using `TimeSeriesSplit` (5 splits) and negative MSE, so validation always respects the order of time.

| Model | Role |
| --- | --- |
| `DummyRegressor` | First baseline (about 8 RMSE) |
| Persistence baseline | Predict the current `free_bikes` value for the next hour |
| `HistGradientBoostingRegressor` | Non-linear ensemble model (best model) |
| `Ridge` | Linear model with L2 regularisation |
| `KNeighborsRegressor` | Local, non-parametric model (with `StandardScaler`) |

## Results

| Evaluation | RMSE |
| --- | --- |
| Persistence baseline, test set | 1.824 |
| Best model (HistGradientBoosting), 5-fold time series cross-validation | 1.59 |
| Best model, validation set | 1.11 |
| Best model, test set | 1.81 |

The standard deviation of `free_bikes` is about 7, so an error of about 1.8 bikes is small in absolute terms. However, the improvement over the persistence baseline is marginal. The report attributes this to the limited training data (about five months), which prevents the model from learning seasonal patterns.

Permutation importance shows that `free_bikes` is by far the most important feature (ΔMSE about 14.9), followed by `hour` (0.40) and `flow` (0.17). Weather features contribute very little to the predictions. A test with only the three most important features changed RMSE by less than 0.1.

## Limitations and next steps

* The training data covers only about five months, so seasonality is not captured and the test error rose compared to validation.
* More data across a full year would likely improve generalisation.

## References

* Blasern, B. (2023). Data Science lecture notes. https://blasern.github.io/data-science-forelesningsnotater/intro.html
* Pedregosa, F. et al. Scikit-learn: Machine Learning in Python. https://scikit-learn.org/stable/api/index.html
