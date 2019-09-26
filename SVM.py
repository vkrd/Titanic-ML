from sklearn.svm import SVC
from data_loader import *
from sklearn.model_selection import GridSearchCV

# get data
X, y = load_train_data()

z = load_test_data()

# defining parameter range for grid search
gridSearch = True

param_grid = {'C': [0.1, 1, 10, 100, 1000, 10000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['linear', 'rbf']}

if gridSearch:
    grid = GridSearchCV(SVC(probability=True), param_grid, refit=True, verbose=3)

    # Run grid search
    grid.fit(X, y)

    # Make predictions
    discrete_predictions = grid.predict(z)
    prob_predictions = grid.predict_proba(z)[:, 1]

    print("Best parameters: " + str(grid.get_params()))

    # Save outputs
    save_predictions(discrete_predictions, name="grid_discrete_SVM.csv")
    save_predictions(prob_predictions, name="grid_probability_SVM.csv", fmt="%.4f")
else:
    # discrete predictions
    model = SVC(C=1000, gamma='auto', kernel='linear', probability=True)

    model.fit(X, y)

    discrete_predictions = model.predict(z)
    prob_predictions = model.predict_proba(z)[:, 1]

    # save submissions
    save_predictions(discrete_predictions, name="discrete_SVM.csv")
    save_predictions(prob_predictions, name="probability_SVM.csv", fmt="%.4f")