# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
import importlib
import My_Pipeline
importlib.reload(My_Pipeline)
from My_Pipeline import my_pipeline

randomstate = 42

# %%
model_ready_df = my_pipeline('raw_data')
model_ready_df.head()

# %%
X = model_ready_df.iloc[:, :-1]
y = model_ready_df.iloc[:, -1]
X = X.drop(columns=['timestamp','station'])


# %%
X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, random_state=randomstate, train_size=.6, shuffle=False)
X_val, X_test, y_val, y_test = train_test_split(X_test_val, y_test_val, train_size=.5, shuffle=False)

# %%
models = {}
hgb_pipe = Pipeline([
    ("gb", HistGradientBoostingRegressor(
        max_iter=500,
        early_stopping=False,
        random_state=randomstate,
    )),
])

tscv = TimeSeriesSplit(n_splits=5)

param_grid = {
    "gb__learning_rate": [0.03, 0.05, 0.1],
    "gb__max_depth": [None, 6, 12],
    "gb__min_samples_leaf": [20, 50],
    "gb__l2_regularization": [0.0, 0.1],
}

gs_hgb = GridSearchCV(
    estimator=hgb_pipe,
    param_grid=param_grid,
    cv=tscv,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    verbose=0,
)

gs_hgb.fit(X_train, y_train)

best_hgb = gs_hgb.best_estimator_
cv_rmse_hgb = (-gs_hgb.best_score_) ** 0.5
models[best_hgb] = cv_rmse_hgb


# %%
ridge_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("reg", Ridge(alpha=1.0, random_state=randomstate))
])

param_grid = {
    "reg__alpha": np.logspace(-2, 2, 9)
}

gs_ridge = GridSearchCV(
    estimator=ridge_pipe,
    param_grid=param_grid,
    cv=tscv,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    verbose=0,
)

gs_ridge.fit(X_train, y_train)

best_ridge = gs_ridge.best_estimator_
cv_rmse_ridge = (-gs_ridge.best_score_) ** 0.5

models[best_ridge] = cv_rmse_ridge

# %%
knn_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsRegressor())
])

param_grid = {
    "knn__n_neighbors": [3, 5, 10, 20],
    "knn__weights": ["uniform", "distance"],
}

gs_knn = GridSearchCV(
    estimator=knn_pipe,
    param_grid=param_grid,
    cv=tscv,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    verbose=0,
)

gs_knn.fit(X_train, y_train)

best_knn = gs_knn.best_estimator_
cv_rmse_knn = (-gs_knn.best_score_) ** 0.5
models[best_knn] = cv_rmse_knn

# %%
best_model = min(models, key=models.get)

y_val_pred = best_model.predict(X_val)
best_model_val_rmse = mean_squared_error(y_val, y_val_pred)**.5


# %%
y_test_pred = best_model.predict(X_test)
best_model_test_rmse = round(mean_squared_error(y_test, y_test_pred)**.5, 4)