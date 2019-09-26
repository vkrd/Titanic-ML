from sklearn.linear_model import LogisticRegression
from data_loader import *
from sklearn.model_selection import GridSearchCV

# Get data
X, y = load_train_data()

z = load_test_data()

# Defining parameter range for grid search
gridSearch = True

# Create regularization penalty space
penalty = ['none', 'l2']

# Create regularization hyperparameter space
C = np.logspace(0, 10)

# Create hyperparameter options
hyperparameters = dict(C=C, penalty=penalty)

if gridSearch:
    lr = LogisticRegression(solver='lbfgs')
    grid = GridSearchCV(lr, hyperparameters, cv=5, verbose=3)

    # Run grid search
    grid.fit(X, y)

    # Make predictions
    discrete_predictions = grid.predict(z)

    # Print best parameters
    print("Best parameters: " + str(grid.get_params()))

    # Save output
    save_predictions(discrete_predictions, name="grid_discrete_LR.csv")

else:
    # Create model
    model = LogisticRegression(solver='lbfgs')

    model.fit(X, y)

    discrete_predictions = model.predict(z)

    # Save output
    save_predictions(discrete_predictions, name="discrete_LR.csv")
