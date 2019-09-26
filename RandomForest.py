from sklearn.ensemble import RandomForestRegressor
from data_loader import *
from sklearn.model_selection import RandomizedSearchCV

# get data
X, y = load_train_data()

z = load_test_data()

# Defining parameter range for grid search
gridSearch = True

# Random gridsearch code below from:
# https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

if gridSearch:
    rf = RandomForestRegressor()

    grid = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, n_jobs = -1)

    # Run grid search
    grid.fit(X, y)

    # Make predictions
    discrete_predictions = grid.predict(z)

    print("Best parameters: " + str(grid.best_params_))

    # Save output
    save_predictions(discrete_predictions, name="grid_discrete_RF.csv")
else:
    # discrete predictions
    model = RandomForestRegressor(n_estimators=100)

    model.fit(X, y)

    discrete_predictions = model.predict(z)

    # save output
    save_predictions(discrete_predictions, name="discrete_RF.csv")