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
    classes: array-like, shape(M,), optional, default=np.unique(x)
        All possible classes in x.
        If not given, it would be set as np.unique(x).

    Returns
    -------
    dummy_x: array-like, shape(n, M)
        The transfered dummy data.
    """
    if not classes:
        classes = np.unique(x)
    n = len(x)
    M = len(classes)
    index = dict(zip(classes, np.arange(M)))
    dummy_x = np.zeros((n, M), dtype=float)
    for i, x_i in enumerate(x):
        if x_i in classes:
            dummy_x[i, index[x_i]] = 1
        else:
            print(
                "Data {} (index {}) is not in classes.".format(
                    x_i,
                    i))
    return dummy_x



def check_y_survival(y_or_event, *args, allow_all_censored=False):
    """Check that array correctly represents an outcome for survival analysis.

    Parameters
    ----------
    y_or_event : structured array with two fields, or boolean array
        If a structured array, it must contain the binary event indicator
        as first field, and time of event or time of censoring as
        second field. Otherwise, it is assumed that a boolean array
        representing the event indicator is passed.

    *args : list of array-likes
        Any number of array-like objects representing time information.
        Elements that are `None` are passed along in the return value.

    allow_all_censored : bool, optional, default: False
        Whether to allow all events to be censored.

    Returns
    -------
    event : array, shape=[n_samples,], dtype=bool
        Binary event indicator.

    time : array, shape=[n_samples,], dtype=float
        Time of event or censoring.
    """
    if len(args) == 0:
        y = y_or_event

        if not isinstance(y, np.ndarray) or y.dtype.fields is None or len(y.dtype.fields) != 2:
            raise ValueError('y must be a structured array with the first field'
                             ' being a binary class event indicator and the second field'
                             ' the time of the event/censoring')

        event_field, time_field = y.dtype.names
        y_event = y[event_field]
        time_args = (y[time_field],)
    else:
        y_event = np.asanyarray(y_or_event)
        time_args = args

    event = check_array(y_event, ensure_2d=False)
    if not np.issubdtype(event.dtype, np.bool_):
        raise ValueError('elements of event indicator must be boolean, but found {0}'.format(event.dtype))

    if not (allow_all_censored or np.any(event)):
        raise ValueError('all samples are censored')

    return_val = [event]
    for i, yt in enumerate(time_args):
        if yt is None:
            return_val.append(yt)
            continue

        yt = check_array(yt, ensure_2d=False)
        if not np.issubdtype(yt.dtype, np.number):
            raise ValueError('time must be numeric, but found {} for argument {}'.format(yt.dtype, i + 2))

        return_val.append(yt)

    return tuple(return_val)

def _compute_counts(event, time, order=None):
    """Count right censored and uncensored samples at each unique time point.

    Parameters
    ----------
    event : array
        Boolean event indicator.

    time : array
        Survival time or time of censoring.

    order : array or None
        Indices to order time in ascending order.
        If None, order will be computed.

    Returns
    -------
    times : array
        Unique time points.

    n_events : array
        Number of events at each time point.

    n_at_risk : array
        Number of samples that have not been censored or have not had an event at each time point.

    n_censored : array
        Number of censored samples at each time point.
    """
    n_samples = event.shape[0]

    if order is None:
        order = np.argsort(time, kind="mergesort")

    uniq_times = np.empty(n_samples, dtype=time.dtype)
    uniq_events = np.empty(n_samples, dtype=int)
    uniq_counts = np.empty(n_samples, dtype=int)

    i = 0
    prev_val = time[order[0]]
    j = 0
    while True:
        count_event = 0
        count = 0
        while i < n_samples and prev_val == time[order[i]]:
            if event[order[i]]:
                count_event += 1

            count += 1
            i += 1

        uniq_times[j] = prev_val
        uniq_events[j] = count_event
        uniq_counts[j] = count
        j += 1

        if i == n_samples:
            break

        prev_val = time[order[i]]

    times = np.resize(uniq_times, j)
    n_events = np.resize(uniq_events, j)
    total_count = np.resize(uniq_counts, j)
    n_censored = total_count - n_events

    # offset cumulative sum by one
    total_count = np.r_[0, total_count]
    n_at_risk = n_samples - np.cumsum(total_count)

    return times, n_events, n_at_risk[:-1], n_censored