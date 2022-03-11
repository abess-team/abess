import numpy as np
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


def new_data_check(self, X, y=None, weights=None):
    """
    Check new data for predicting, scoring or else.
    """
    # Check1 : whether fit had been called
    check_is_fitted(self)

    # Check2 : X validation
    X = check_array(X, accept_sparse=True)
    if X.shape[1] != self.n_features_in_:
        raise ValueError("X.shape[1] should be " +
                         str(self.n_features_in_))

    # Check3 : X, y validation
    if (y is not None) and (weights is None):
        X, y = check_X_y(X,
                         y,
                         accept_sparse=True,
                         multi_output=True,
                         y_numeric=True)
        return X, y

    # Check4: X, y, weights validation
    if weights is not None:
        X, y = check_X_y(X,
                         y,
                         accept_sparse=True,
                         multi_output=True,
                         y_numeric=True)
        weights = np.array(weights, dtype=float)

        if len(weights.shape) != 1:
            raise ValueError("weights should be 1-dimension.")
        if weights.shape[0] != X.shape[0]:
            raise ValueError("weights should have a length of X.shape[0].")
        return X, y, weights

    return X


def categorical_to_dummy(x, classes=None):
    """
    Transfer categorical variable into dummy variable.

    Parameters
    ----------
    x: array-like, shape(n,)
        Data of the categorical variable.
    classes: array-like, shape(M,), optional, default=numpy.unique(x)
        All possible classes in x.
        If not given, it would be set as numpy.unique(x).

    Returns
    -------
    dummy_x: array-like, shape(n, M)
        The transfered dummy data.
    """
    if not classes:
        classes = np.unique(x)
    print("classes: {}".format(classes))
    n = len(x)
    M = len(classes)
    index = dict(zip(classes, np.arange(M)))
    dummy_x = np.zeros((n, M), dtype=float)
    for i, x_i in enumerate(x):
        if x_i in classes:
            dummy_x[i, index[x_i]] = 1
        # else:
        #     print(
        #         "Data {} (index {}) is not in classes.".format(
        #             x_i,
        #             i))
    return dummy_x
