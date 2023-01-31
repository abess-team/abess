import warnings
from .decomposition import SparsePCA, RobustPCA

# This is the old API for `abess.decomposition`
# and will be removed in version 0.6.0.


class abessPCA(SparsePCA):
    warning_msg = ("Class ``abess.pca.abessPCA`` has been renamed to "
                   "``abess.decomposition.SparsePCA``. "
                   "The former will be deprecated in version 0.6.0.")
    __doc__ = warning_msg + '\n' + SparsePCA.__doc__

    def __init__(self, max_iter=20, exchange_num=5,
                 is_warm_start=True, support_size=None,
                 ic_type="loss", ic_coef=1.0, cv=1, cv_score="test_loss",
                 screening_size=-1,
                 always_select=None,
                 thread=1,
                 A_init=None,
                 group=None,
                 splicing_type=1
                 ):
        warnings.warn(self.warning_msg, FutureWarning)
        super().__init__(
            max_iter=max_iter, exchange_num=exchange_num,
            is_warm_start=is_warm_start,
            support_size=support_size,
            ic_type=ic_type, ic_coef=ic_coef,
            cv=cv, cv_score=cv_score,
            screening_size=screening_size,
            always_select=always_select,
            thread=thread,
            A_init=A_init, group=group,
            splicing_type=splicing_type
        )


class abessRPCA(RobustPCA):
    warning_msg = ("Class ``abess.pca.abessRPCA`` has been renamed to "
                   "``abess.decomposition.RobustPCA``. "
                   "The former will be deprecated in version 0.6.0.")
    __doc__ = warning_msg + '\n' + RobustPCA.__doc__

    def __init__(self, max_iter=20, exchange_num=5,
                 is_warm_start=True, support_size=None,
                 ic_type="gic", ic_coef=1.0,
                 always_select=None,
                 thread=1,
                 A_init=None,
                 splicing_type=1
                 ):
        warnings.warn(self.warning_msg, FutureWarning)
        super().__init__(
            max_iter=max_iter, exchange_num=exchange_num,
            is_warm_start=is_warm_start, support_size=support_size,
            ic_type=ic_type, ic_coef=ic_coef,
            always_select=always_select,
            thread=thread,
            A_init=A_init,
            splicing_type=splicing_type
        )
