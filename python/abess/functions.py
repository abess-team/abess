# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import warnings
import numbers
import operator
import numpy
import numpy as np
from sklearn.utils import (
    check_consistent_length,
    column_or_1d,
    check_array,
    check_consistent_length)
from sklearn.metrics._regression import (
    _num_samples,
    _check_reg_targets)
from sklearn.exceptions import UndefinedMetricWarning
from scipy.special import xlogy


class StepFunction:
    """Callable step function.

    .. math::

        f(z) = a * y_i + b,
        x_i \\leq z < x_{i + 1}

    Parameters
    ----------
    x : ndarray, shape = (n_points,)
        Values on the x axis in ascending order.

    y : ndarray, shape = (n_points,)
        Corresponding values on the y axis.

    a : float, optional, default: 1.0
        Constant to multiply by.

    b : float, optional, default: 0.0
        Constant offset term.
    """

    def __init__(self, x, y, a=1., b=0.):
        check_consistent_length(x, y)
        self.x = x
        self.y = y
        self.a = a
        self.b = b

    def __call__(self, x):
        """Evaluate step function.

        Parameters
        ----------
        x : float|array-like, shape=(n_values,)
            Values to evaluate step function at.

        Returns
        -------
        y : float|array-like, shape=(n_values,)
            Values of step function at `x`.
        """
        x = np.atleast_1d(x)
        if not np.isfinite(x).all():
            raise ValueError("x must be finite")
        if np.min(x) < self.x[0] or np.max(x) > self.x[-1]:
            raise ValueError(
                "x must be within [%f; %f]" % (self.x[0], self.x[-1]))
        i = np.searchsorted(self.x, x, side='left')
        not_exact = self.x[i] != x
        i[not_exact] -= 1
        value = self.a * self.y[i] + self.b
        if value.shape[0] == 1:
            return value[0]
        return value

    # def __repr__(self):
    #     return "StepFunction(x=%r, y=%r, a=%r, b=%r)" % (
    #         self.x, self.y, self.a, self.b)


class BreslowEstimator:
    r"""Breslow's estimator of the cumulative hazard function.
    Attributes
    ----------
    cum_baseline_hazard_ : :class:`sksurv.functions.StepFunction`
        Cumulative baseline hazard function.
    baseline_survival_ : :class:`sksurv.functions.StepFunction`
        Baseline survival function.
    """

    def fit(self, linear_predictor, event, time):
        r"""Compute baseline cumulative hazard function.
        Parameters
        ----------
        linear_predictor : array-like, shape = (n_samples,)
            Linear predictor of risk: `X @ coef`.
        event : array-like, shape = (n_samples,)
            Contains binary event indicators.
        time : array-like, shape = (n_samples,)
            Contains event/censoring times.
        Returns
        -------
        self
        """
        risk_score = np.exp(linear_predictor)
        order = np.argsort(time, kind="mergesort")
        risk_score = risk_score[order]
        uniq_times, n_events, n_at_risk, _ = self._compute_counts(
            event, time, order)

        divisor = np.empty(n_at_risk.shape, dtype=float)
        value = np.sum(risk_score)
        divisor[0] = value
        k = 0
        for i in range(1, len(n_at_risk)):
            d = n_at_risk[i - 1] - n_at_risk[i]
            value -= risk_score[k:(k + d)].sum()
            k += d
            divisor[i] = value

        assert k == n_at_risk[0] - n_at_risk[-1]

        y = np.cumsum(n_events / divisor)
        self.cum_baseline_hazard_ = StepFunction(uniq_times, y)
        self.baseline_survival_ = StepFunction(
            self.cum_baseline_hazard_.x,
            np.exp(- self.cum_baseline_hazard_.y))
        return self

    # def get_cumulative_hazard_function(self, linear_predictor):
    #     r"""Predict cumulative hazard function.
    #     Parameters
    #     ----------
    #     linear_predictor : array-like, shape = (n_samples,)
    #         Linear predictor of risk: `X @ coef`.
    #     Returns
    #     -------
    #     cum_hazard : ndarray, shape = (n_samples,)
    #         Predicted cumulative hazard functions.
    #     """
    #     risk_score = np.exp(linear_predictor)
    #     n_samples = risk_score.shape[0]
    #     funcs = np.empty(n_samples, dtype=object)
    #     for i in range(n_samples):
    #         funcs[i] = StepFunction(x=self.cum_baseline_hazard_.x,
    #                                 y=self.cum_baseline_hazard_.y,
    #                                 a=risk_score[i])
    #     return funcs

    def get_survival_function(self, linear_predictor):
        r"""Predict survival function.
        Parameters
        ----------
        linear_predictor : array-like, shape = (n_samples,)
            Linear predictor of risk: `X @ coef`.
        Returns
        -------
        survival : ndarray, shape = (n_samples,)
            Predicted survival functions.
        """
        risk_score = np.exp(linear_predictor)
        n_samples = risk_score.shape[0]
        funcs = np.empty(n_samples, dtype=object)
        for i in range(n_samples):
            funcs[i] = StepFunction(
                x=self.baseline_survival_.x,
                y=np.power(self.baseline_survival_.y, risk_score[i]))
        return funcs

    @staticmethod
    def _compute_counts(event, time, order=None):
        """
        Count right censored and uncensored samples at each unique time point.

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
            Number of samples that have not been censored or
            have not had an event at each time point.

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


def check_scalar(
    x,
    name,
    target_type,
    *,
    min_val=None,
    max_val=None,
    include_boundaries="both",
):
    """Validate scalar parameters type and value.

    Parameters
    ----------
    x : object
        The scalar parameter to validate.

    name : str
        The name of the parameter to be printed in error messages.

    target_type : type or tuple
        Acceptable data types for the parameter.

    min_val : float or int, default=None
        The minimum valid value the parameter can take. If None (default) it
        is implied that the parameter does not have a lower bound.

    max_val : float or int, default=None
        The maximum valid value the parameter can take. If None (default) it
        is implied that the parameter does not have an upper bound.

    include_boundaries : {"left", "right", "both", "neither"}, default="both"
        Whether the interval defined by `min_val` and `max_val` should include
        the boundaries. Possible choices are:

        - `"left"`: only `min_val` is included in the valid interval.
          It is equivalent to the interval `[ min_val, max_val )`.
        - `"right"`: only `max_val` is included in the valid interval.
          It is equivalent to the interval `( min_val, max_val ]`.
        - `"both"`: `min_val` and `max_val` are included in the valid interval.
          It is equivalent to the interval `[ min_val, max_val ]`.
        - `"neither"`: neither `min_val` nor `max_val` are included in the
          valid interval. It is equivalent to the interval `( min_val, max_val )`.

    Returns
    -------
    x : numbers.Number
        The validated number.

    Raises
    ------
    TypeError
        If the parameter's type does not match the desired type.

    ValueError
        If the parameter's value violates the given bounds.
        If `min_val`, `max_val` and `include_boundaries` are inconsistent.
    """

    def type_name(t):
        """Convert type into humman readable string."""
        module = t.__module__
        qualname = t.__qualname__
        if module == "builtins":
            return qualname
        elif t == numbers.Real:
            return "float"
        elif t == numbers.Integral:
            return "int"
        return f"{module}.{qualname}"

    if not isinstance(x, target_type):
        if isinstance(target_type, tuple):
            types_str = ", ".join(type_name(t) for t in target_type)
            target_type_str = f"{{{types_str}}}"
        else:
            target_type_str = type_name(target_type)

        raise TypeError(
            f"{name} must be an instance of {target_type_str}, not"
            f" {type(x).__qualname__}."
        )

    expected_include_boundaries = ("left", "right", "both", "neither")
    if include_boundaries not in expected_include_boundaries:
        raise ValueError(
            f"Unknown value for `include_boundaries`: {repr(include_boundaries)}. "
            f"Possible values are: {expected_include_boundaries}."
        )

    if max_val is None and include_boundaries == "right":
        raise ValueError(
            "`include_boundaries`='right' without specifying explicitly `max_val` "
            "is inconsistent."
        )

    if min_val is None and include_boundaries == "left":
        raise ValueError(
            "`include_boundaries`='left' without"
            " specifying explicitly `min_val` "
            "is inconsistent."
        )

    comparison_operator = (
        operator.lt if include_boundaries in ("left", "both") else operator.le
    )
    if min_val is not None and comparison_operator(x, min_val):
        raise ValueError(
            f"{name} == {x}, must be"
            f" {'>=' if include_boundaries in ('left', 'both') else '>'} "
            f"{min_val}."
        )

    comparison_operator = (
        operator.gt if include_boundaries in ("right", "both") else operator.ge
    )
    if max_val is not None and comparison_operator(x, max_val):
        raise ValueError(
            f"{name} == {x}, must be"
            f" {'<=' if include_boundaries in ('right', 'both') else '<'} "
            f"{max_val}."
        )

    return x


def _mean_tweedie_deviance(y_true, y_pred, sample_weight, power):
    """Mean Tweedie deviance regression loss."""
    p = power
    if p < 0:
        # 'Extreme stable', y any real number, y_pred > 0
        dev = 2 * (
            np.power(np.maximum(y_true, 0), 2 - p) / ((1 - p) * (2 - p))
            - y_true * np.power(y_pred, 1 - p) / (1 - p)
            + np.power(y_pred, 2 - p) / (2 - p)
        )
    elif p == 0:
        # Normal distribution, y and y_pred any real number
        dev = (y_true - y_pred) ** 2
    elif p == 1:
        # Poisson distribution
        dev = 2 * (xlogy(y_true, y_true / y_pred) - y_true + y_pred)
    elif p == 2:
        # Gamma distribution
        dev = 2 * (np.log(y_pred / y_true) + y_true / y_pred - 1)
    else:
        dev = 2 * (
            np.power(y_true, 2 - p) / ((1 - p) * (2 - p))
            - y_true * np.power(y_pred, 1 - p) / (1 - p)
            + np.power(y_pred, 2 - p) / (2 - p)
        )

    return np.average(dev, weights=sample_weight)


def mean_tweedie_deviance(y_true, y_pred, *, sample_weight=None, power=0):
    """Mean Tweedie deviance regression loss.
    Read more in the :ref:`User Guide <mean_tweedie_deviance>`.
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    power : float, default=0
        Tweedie power parameter. Either power <= 0 or power >= 1.
        The higher `p` the less weight is given to extreme
        deviations between true and predicted targets.
        - power < 0: Extreme stable distribution. Requires: y_pred > 0.
        - power = 0 : Normal distribution, output corresponds to
          mean_squared_error. y_true and y_pred can be any real numbers.
        - power = 1 : Poisson distribution. Requires: y_true >= 0 and
          y_pred > 0.
        - 1 < p < 2 : Compound Poisson distribution. Requires: y_true >= 0
          and y_pred > 0.
        - power = 2 : Gamma distribution. Requires: y_true > 0 and y_pred > 0.
        - power = 3 : Inverse Gaussian distribution. Requires: y_true > 0
          and y_pred > 0.
        - otherwise : Positive stable distribution. Requires: y_true > 0
          and y_pred > 0.
    Returns
    -------
    loss : float
        A non-negative floating point value (the best value is 0.0).
    Examples
    --------
    >>> from sklearn.metrics import mean_tweedie_deviance
    >>> y_true = [2, 0, 1, 4]
    >>> y_pred = [0.5, 0.5, 2., 2.]
    >>> mean_tweedie_deviance(y_true, y_pred, power=1)
    1.4260...
    """
    y_type, y_true, y_pred, _ = _check_reg_targets(
        y_true, y_pred, None, dtype=[np.float64, np.float32]
    )
    if y_type == "continuous-multioutput":
        raise ValueError("Multioutput not supported in mean_tweedie_deviance")
    check_consistent_length(y_true, y_pred, sample_weight)

    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)
        sample_weight = sample_weight[:, np.newaxis]

    p = check_scalar(
        power,
        name="power",
        target_type=numbers.Real,
    )

    message = f"Mean Tweedie deviance error with power={p} can only be used on "
    if p < 0:
        # 'Extreme stable', y any real number, y_pred > 0
        if (y_pred <= 0).any():
            raise ValueError(message + "strictly positive y_pred.")
    elif p == 0:
        # Normal, y and y_pred can be any real number
        pass
    elif 0 < p < 1:
        raise ValueError(
            "Tweedie deviance is only defined for power<=0 and power>=1.")
    elif 1 <= p < 2:
        # Poisson and compound Poisson distribution, y >= 0, y_pred > 0
        if (y_true < 0).any() or (y_pred <= 0).any():
            raise ValueError(
                message + "non-negative y and strictly positive y_pred.")
    elif p >= 2:
        # Gamma and Extreme stable distribution, y and y_pred > 0
        if (y_true <= 0).any() or (y_pred <= 0).any():
            raise ValueError(message + "strictly positive y and y_pred.")
    else:  # pragma: nocover
        # Unreachable statement
        raise ValueError

    return _mean_tweedie_deviance(
        y_true, y_pred, sample_weight=sample_weight, power=power
    )


def d2_tweedie_score(y_true, y_pred, *, sample_weight=None, power=0):
    """D^2 regression score function, fraction of Tweedie deviance explained.
    Best possible score is 1.0 and it can be negative (because the model can be
    arbitrarily worse). A model that always uses the empirical mean of `y_true` as
    constant prediction, disregarding the input features, gets a D^2 score of 0.0.
    Read more in the :ref:`User Guide <d2_score>`.
    .. versionadded:: 1.0
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.
    sample_weight : array-like of shape (n_samples,), optional
        Sample weights.
    power : float, default=0
        Tweedie power parameter. Either power <= 0 or power >= 1.
        The higher `p` the less weight is given to extreme
        deviations between true and predicted targets.
        - power < 0: Extreme stable distribution. Requires: y_pred > 0.
        - power = 0 : Normal distribution, output corresponds to r2_score.
          y_true and y_pred can be any real numbers.
        - power = 1 : Poisson distribution. Requires: y_true >= 0 and
          y_pred > 0.
        - 1 < p < 2 : Compound Poisson distribution. Requires: y_true >= 0
          and y_pred > 0.
        - power = 2 : Gamma distribution. Requires: y_true > 0 and y_pred > 0.
        - power = 3 : Inverse Gaussian distribution. Requires: y_true > 0
          and y_pred > 0.
        - otherwise : Positive stable distribution. Requires: y_true > 0
          and y_pred > 0.
    Returns
    -------
    z : float or ndarray of floats
        The D^2 score.
    Notes
    -----
    This is not a symmetric function.
    Like R^2, D^2 score may be negative (it need not actually be the square of
    a quantity D).
    This metric is not well-defined for single samples and will return a NaN
    value if n_samples is less than two.
    References
    ----------
    .. [1] Eq. (3.11) of Hastie, Trevor J., Robert Tibshirani and Martin J.
           Wainwright. "Statistical Learning with Sparsity: The Lasso and
           Generalizations." (2015). https://trevorhastie.github.io
    Examples
    --------
    >>> from sklearn.metrics import d2_tweedie_score
    >>> y_true = [0.5, 1, 2.5, 7]
    >>> y_pred = [1, 1, 5, 3.5]
    >>> d2_tweedie_score(y_true, y_pred)
    0.285...
    >>> d2_tweedie_score(y_true, y_pred, power=1)
    0.487...
    >>> d2_tweedie_score(y_true, y_pred, power=2)
    0.630...
    >>> d2_tweedie_score(y_true, y_true, power=2)
    1.0
    """
    y_type, y_true, y_pred, _ = _check_reg_targets(
        y_true, y_pred, None, dtype=[np.float64, np.float32]
    )
    if y_type == "continuous-multioutput":
        raise ValueError("Multioutput not supported in d2_tweedie_score")

    if _num_samples(y_pred) < 2:
        msg = "D^2 score is not well-defined with less than two samples."
        warnings.warn(msg, UndefinedMetricWarning)
        return float("nan")

    y_true, y_pred = np.squeeze(y_true), np.squeeze(y_pred)
    numerator = mean_tweedie_deviance(
        y_true, y_pred, sample_weight=sample_weight, power=power
    )

    y_avg = np.average(y_true, weights=sample_weight)
    denominator = _mean_tweedie_deviance(
        y_true, y_avg, sample_weight=sample_weight, power=power
    )

    return 1 - numerator / denominator


def _check_estimate_1d(estimate, test_time):
    estimate = check_array(estimate, ensure_2d=False)
    if estimate.ndim != 1:
        raise ValueError(
            'Expected 1D array, got {}D array instead:\narray={}.\n'.format(
                estimate.ndim, estimate))
    check_consistent_length(test_time, estimate)
    return estimate


def _check_inputs(event_indicator, event_time, estimate):
    check_consistent_length(event_indicator, event_time, estimate)
    event_indicator = check_array(event_indicator, ensure_2d=False)
    event_time = check_array(event_time, ensure_2d=False)
    estimate = _check_estimate_1d(estimate, event_time)

    if not numpy.issubdtype(event_indicator.dtype, numpy.bool_):
        raise ValueError(
            'only boolean arrays are supported as class labels '
            'for survival analysis, got {}'.format(event_indicator.dtype))

    if len(event_time) < 2:
        raise ValueError("Need a minimum of two samples")

    if not event_indicator.any():
        raise ValueError("All samples are censored")

    return event_indicator, event_time, estimate


def _get_comparable(event_indicator, event_time, order):
    n_samples = len(event_time)
    tied_time = 0
    comparable = {}
    i = 0
    while i < n_samples - 1:
        time_i = event_time[order[i]]
        start = i + 1
        end = start
        while end < n_samples and event_time[order[end]] == time_i:
            end += 1

        # check for tied event times
        event_at_same_time = event_indicator[order[i:end]]
        censored_at_same_time = ~event_at_same_time
        for j in range(i, end):
            if event_indicator[order[j]]:
                mask = numpy.zeros(n_samples, dtype=bool)
                mask[end:] = True
                # an event is comparable to censored samples at same time point
                mask[i:end] = censored_at_same_time
                comparable[j] = mask
                tied_time += censored_at_same_time.sum()
        i = end

    return comparable, tied_time


def _estimate_concordance_index(
        event_indicator, event_time, estimate, weights, tied_tol=1e-8):
    order = numpy.argsort(event_time)

    comparable, tied_time = _get_comparable(event_indicator, event_time, order)

    # if len(comparable) == 0:
    #     raise NoComparablePairException(
    #         "Data has no comparable pairs, "
    #         "cannot estimate concordance index.")

    concordant = 0
    discordant = 0
    tied_risk = 0
    numerator = 0.0
    denominator = 0.0
    for ind, mask in comparable.items():
        est_i = estimate[order[ind]]
        event_i = event_indicator[order[ind]]
        w_i = weights[order[ind]]

        est = estimate[order[mask]]

        msg = 'got censored sample at index {}'.format(
            order[ind]) + ', but expected uncensored'
        assert event_i, msg

        ties = numpy.absolute(est - est_i) <= tied_tol
        n_ties = ties.sum()
        # an event should have a higher score
        con = est < est_i
        n_con = con[~ties].sum()

        numerator += w_i * n_con + 0.5 * w_i * n_ties
        denominator += w_i * mask.sum()

        tied_risk += n_ties
        concordant += n_con
        discordant += est.size - n_con - n_ties

    cindex = numerator / denominator
    return cindex, concordant, discordant, tied_risk, tied_time


def concordance_index_censored(event_indicator, event_time,
                               estimate, sample_weight=None, tied_tol=1e-8):
    """Concordance index for right-censored data

    Reference from scikit-survival:
    `sksurv.metrics.concordance_index_censored`.

    The concordance index is defined as the proportion of
    all comparable pairs
    in which the predictions and outcomes are concordant.
    Two samples are comparable if (i) both of them experienced an event
    (at different times),
    or (ii) the one with a shorter observed survival time experienced an event,
    in which case
    the event-free subject "outlived" the other.
    A pair is not comparable if they experienced
    events at the same time.
    Concordance intuitively means that two samples were
    ordered correctly by the model.
    More specifically, two samples are concordant,
    if the one with a higher estimated
    risk score has a shorter actual survival time.
    When predicted risks are identical for a pair,
    0.5 rather than 1 is added to the count
    of concordant pairs.
    See the :ref:`User Guide </user_guide/evaluating-survival-models.ipynb>`
    and [1]_ for further description.
    Parameters
    ----------
    event_indicator : array-like, shape = (n_samples,)
        Boolean array denotes whether an event occurred
    event_time : array-like, shape = (n_samples,)
        Array containing the time of an event or time of censoring
    estimate : array-like, shape = (n_samples,)
        Estimated risk of experiencing an event
    tied_tol : float, optional, default: 1e-8
        The tolerance value for considering ties.
        If the absolute difference between risk scores is smaller
        or equal than `tied_tol`, risk scores are considered tied.
    Returns
    -------
    cindex : float
        Concordance index
    concordant : int
        Number of concordant pairs
    discordant : int
        Number of discordant pairs
    tied_risk : int
        Number of pairs having tied estimated risks
    tied_time : int
        Number of comparable pairs sharing the same time
    See also
    --------
    concordance_index_ipcw
        Alternative estimator of the concordance index with less bias.
    References
    ----------
    .. [1] Harrell, F.E., Califf, R.M., Pryor, D.B., Lee, K.L., Rosati, R.A,
           "Multivariable prognostic models: issues in developing models,
           evaluating assumptions and adequacy,
           and measuring and reducing errors",
           Statistics in Medicine, 15(4), 361-87, 1996.
    """
    event_indicator, event_time, estimate = _check_inputs(
        event_indicator, event_time, estimate)

    if sample_weight is None:
        sample_weight = numpy.ones_like(estimate)

    return _estimate_concordance_index(
        event_indicator, event_time, estimate, sample_weight, tied_tol)
