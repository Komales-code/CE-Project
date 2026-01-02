import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

def evaluate_svr(params, X_train, X_test, y_train, y_test):
    C, epsilon, gamma = params

    model = SVR(C=C, epsilon=epsilon, gamma=gamma)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    return rmse
