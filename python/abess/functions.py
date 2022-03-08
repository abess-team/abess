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

import numpy as np
from sklearn.utils import check_consistent_length
from .utilities import _compute_counts

__all__ = ['StepFunction']


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
        x = numpy.atleast_1d(x)
        if not numpy.isfinite(x).all():
            raise ValueError("x must be finite")
        if numpy.min(x) < self.x[0] or numpy.max(x) > self.x[-1]:
            raise ValueError(
                "x must be within [%f; %f]" % (self.x[0], self.x[-1]))
        i = numpy.searchsorted(self.x, x, side='left')
        not_exact = self.x[i] != x
        i[not_exact] -= 1
        value = self.a * self.y[i] + self.b
        if value.shape[0] == 1:
            return value[0]
        return value

    def __repr__(self):
        return "StepFunction(x=%r, y=%r, a=%r, b=%r)" % (
            self.x, self.y, self.a, self.b)


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
        uniq_times, n_events, n_at_risk, _ = _compute_counts(
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
        self.baseline_survival_ = StepFunction(self.cum_baseline_hazard_.x,
                                               np.exp(- self.cum_baseline_hazard_.y))
        return self

    def get_cumulative_hazard_function(self, linear_predictor):
        r"""Predict cumulative hazard function.
        Parameters
        ----------
        linear_predictor : array-like, shape = (n_samples,)
            Linear predictor of risk: `X @ coef`.
        Returns
        -------
        cum_hazard : ndarray, shape = (n_samples,)
            Predicted cumulative hazard functions.
        """
        risk_score = np.exp(linear_predictor)
        n_samples = risk_score.shape[0]
        funcs = np.empty(n_samples, dtype=object)
        for i in range(n_samples):
            funcs[i] = StepFunction(x=self.cum_baseline_hazard_.x,
                                    y=self.cum_baseline_hazard_.y,
                                    a=risk_score[i])
        return funcs

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
            funcs[i] = StepFunction(x=self.baseline_survival_.x,
                                    y=np.power(self.baseline_survival_.y, risk_score[i]))
        return funcs
