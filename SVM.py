from sklearn import svm
from data_loader import *

X, y = load_train_data()

z = load_test_data()

# discrete_predictions
clf = svm.SVC(gamma='auto')
clf.fit(X, y)

discrete_predictions = clf.predict(z)


# probability predictions
clf = svm.SVC(gamma='auto', probability=True)
clf.fit(X, y)

prob_predictions = clf.predict_proba(z)[:, 1]

# save submissions
save_predictions(discrete_predictions, name="discrete_SVM.csv")
save_predictions(prob_predictions, name="probability_SVM.csv")