from sklearn.svm import SVC
from data_loader import *
from sklearn.model_selection import GridSearchCV

# get data
X, y = load_train_data()

z = load_test_data()

# Defining parameter range for grid search
gridSearch = False

param_grid = {'C': [1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['linear', 'rbf']}

if gridSearch:
    grid = GridSearchCV(SVC(probability=True), param_grid, refit=True, verbose=3, n_jobs=-1)

    # Run grid search
    grid.fit(X, y)

    # Make predictions
    discrete_predictions = grid.predict(z)
    prob_predictions = grid.predict_proba(z)[:, 1]

    print("Best parameters: " + str(grid.best_params_))

    # Save output
    save_predictions(discrete_predictions, name="grid_discrete_SVM.csv")
else:
    # discrete predictions
    model = SVC(C=1000, gamma=1, kernel='linear')

    model.fit(X, y)

    discrete_predictions = model.predict(z)
    prob_predictions = model.predict_proba(z)[:, 1]

    # save output
    save_predictions(discrete_predictions, name="discrete_SVM.csv")