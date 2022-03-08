import warnings
import abess
import numpy as np

warnings.filterwarnings("ignore")


class TestCheck:
    """
    Test for argument error, which should be recognized before the algorithm.
    """

    @staticmethod
    def test_base():
        # path
        try:
            model = abess.LinearRegression(path_type='other')
            model.fit([[1]], [1])
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model = abess.LinearRegression(support_size=[3])
            model.fit([[1]], [1])
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model = abess.LinearRegression(path_type='gs', s_min=1, s_max=0)
            model.fit([[1]], [1])
        except ValueError as e:
            print(e)
        else:
            assert False

        # ic
        try:
            model = abess.LinearRegression(ic_type='other')
            model.fit([[1]], [1])
        except ValueError as e:
            print(e)
        else:
            assert False

        # exchange_num
        try:
            model = abess.LinearRegression(exchange_num=-1)
            model.fit([[1]], [1])
        except ValueError as e:
            print(e)
        else:
            assert False

        # screening_size
        try:
            model = abess.LinearRegression(screening_size=3)
            model.fit([[1]], [1])
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model = abess.LinearRegression(support_size=[2],
                                           screening_size=1)
            model.fit([[1, 2, 3]], [1])
        except ValueError as e:
            print(e)
        else:
            assert False

        # primary_fit_xxx
        try:
            model = abess.LogisticRegression(primary_model_fit_max_iter=0.5)
            model.fit([[1]], [1])
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model = abess.LogisticRegression(primary_model_fit_epsilon=-1)
            model.fit([[1]], [1])
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model = abess.LogisticRegression(primary_model_fit_epsilon=-1)
            model.fit([[1]], [1])
        except ValueError as e:
            print(e)
        else:
            assert False

        # thread
        try:
            model = abess.LinearRegression(thread=-1)
            model.fit([[1]], [1])
        except ValueError as e:
            print(e)
        else:
            assert False

        # splicing_type
        try:
            model = abess.LinearRegression(splicing_type=-1)
            model.fit([[1]], [1])
        except ValueError as e:
            print(e)
        else:
            assert False

        # cv & cv_fold_id
        try:
            model = abess.LinearRegression(cv=2)
            model.fit([[1]], [1])
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model = abess.LinearRegression(cv=2)
            cv_fold_id = [[1], [2]]
            model.fit([[1], [2]], [1, 2], cv_fold_id=cv_fold_id)
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model = abess.LinearRegression(cv=2)
            cv_fold_id = [1, 1]
            model.fit([[1], [1]], [1, 1], cv_fold_id=cv_fold_id)
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model = abess.LinearRegression(cv=2)
            cv_fold_id = [1, 2, 1]
            model.fit([[1], [1]], [1, 1], cv_fold_id=cv_fold_id)
        except ValueError as e:
            print(e)
        else:
            assert False

        model = abess.LinearRegression()
        # datatype error
        try:
            model.fit([['c', 1, 1]], [1])
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model.fit([[1, 1, 1]], [1], weight=['c'])
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model1 = abess.LinearRegression(cv='c')
            model1.fit([[1]], [1])
        except ValueError as e:
            print(e)
        else:
            assert False

        # A_init
        try:
            model.fit([[1]], [1], A_init=[[0]])
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model.fit([[1]], [1], A_init=[2])
        except ValueError as e:
            print(e)
        else:
            assert False

        # imp search
        try:
            model = abess.LinearRegression(important_search=-1)
            model.fit([[1]], [1])
        except ValueError as e:
            print(e)
        else:
            assert False

        # incompatible shape
        try:
            model.fit([1, 1, 1], [1])
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model.fit([[1, 1, 1]], [1, 2])
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model.fit([[1, 1, 1]], [1])
            model.predict([[1, 1]])
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model.fit([[1, 1, 1]], [1], weight=[1, 2])
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model.fit([[1, 1, 1]], [1], weight=[[1, 2, 3]])
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model.fit([[1, 1, 1]], [1], group=[1])
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model.fit([[1, 1, 1]], [1], group=[[1, 2, 3]])
        except ValueError as e:
            print(e)
        else:
            assert False

        # new data
        try:
            data = abess.make_glm_data(n=100, p=10, k=3, family='gamma')
            model = abess.GammaRegression()
            model.fit(data.x, data.y)
            model.score(data.x, data.y, [[1]])
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            data = abess.make_glm_data(n=100, p=10, k=3, family='gamma')
            model = abess.GammaRegression()
            model.fit(data.x, data.y)
            model.score(data.x, data.y, [1])
        except ValueError as e:
            print(e)
        else:
            assert False

        # lack of necessary parameter
        try:
            model.fit(X=[[1]])
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model.fit(y=[1])
        except ValueError as e:
            print(e)
        else:
            assert False

    @staticmethod
    def test_pca():
        """
        For `abess.decomposition.SparsePCA.fit`.
        """
        model = abess.SparsePCA()
        data = np.random.randn(100, 10)
        # datatype error
        try:
            model.fit([['c']])
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model.fit(Sigma=[['c']])
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model.fit(Sigma=[[np.nan]])
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model1 = abess.SparsePCA(cv='c')
            model1.fit([[1]])
        except ValueError as e:
            print(e)
        else:
            assert False

        # incompatible shape
        try:
            model.fit([1])
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model.fit(Sigma=[1])
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model.fit([[1]], group=[1, 2])
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model.fit([[1]], group=[[1]])
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model1 = abess.SparsePCA(support_size=np.array([1, 2]))
            model1.fit([[1]])
        except ValueError as e:
            print(e)
        else:
            assert False

        # screening_size
        model = abess.SparsePCA(screening_size=0)
        model.fit(data)

        try:
            model = abess.SparsePCA(screening_size=100)
            model.fit(data)
        except ValueError as e:
            print(e)
        else:
            assert False

        # lack of necessary parameter
        try:
            model.fit()
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model1 = abess.SparsePCA(cv=5)
            model1.fit(Sigma=[[1]])
        except ValueError as e:
            print(e)
        else:
            assert False

        # number
        try:
            model.fit([[1]], number=-1)
        except ValueError as e:
            print(e)
        else:
            assert False

        # A_init
        try:
            model.fit([[1]], A_init=[[0]])
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model.fit([[1]], A_init=[2])
        except ValueError as e:
            print(e)
        else:
            assert False

        # invalid sigma
        try:
            model.fit(Sigma=[[1, 0], [1, 0]])
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model.fit(Sigma=[[-1, 0], [0, -1]])
        except ValueError as e:
            print(e)
        else:
            assert False

        # invalid arg
        try:
            model1 = abess.SparsePCA(ic_type='other')
            model1.fit([[1]])
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model1 = abess.SparsePCA(cv=5)
            model1.fit([[1]])
        except ValueError as e:
            print(e)
        else:
            assert False

    @staticmethod
    def test_rpca():
        model = abess.RobustPCA()
        # datatype error
        try:
            model.fit([['c']], r=1)
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model.fit([[1]], r='c')
        except ValueError as e:
            print(e)
        else:
            assert False

        # A_init
        try:
            model.fit([[1]], r=1, A_init=[[0]])
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model.fit([[1]], r=1, A_init=[2])
        except ValueError as e:
            print(e)
        else:
            assert False

        # incompatible shape
        try:
            model.fit([1], r=1)
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model.fit(1, r=1)
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model.fit([[1]], r=1, group=[1, 2])
        except ValueError as e:
            print(e)
        else:
            assert False

        # invalid arg
        try:
            model1 = abess.RobustPCA(ic_type='other')
            model1.fit([[1]], r=1)
        except ValueError as e:
            print(e)
        else:
            assert False
