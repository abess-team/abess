from abess import *
from utilities import *
import numpy as np

class TestCheck:
    """
    Test for argument error, which should be recognized before the algorithm.
    """
    def test_init(self):
        # path
        try:
            model = abessLm(path_type='other')
            model.fit([[1]], [1])
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model = abessLm(support_size=[3])
            model.fit([[1]], [1])
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model = abessLm(path_type='gs', s_min=1, s_max=0)
            model.fit([[1]], [1])
        except ValueError as e:
            print(e)
        else:
            assert False

        # ic
        try:
            model = abessLm(ic_type='other')
            model.fit([[1]], [1])
        except ValueError as e:
            print(e)
        else:
            assert False

        # exchange_num
        try:
            model = abessLm(exchange_num=-1)
            model.fit([[1]], [1])
        except ValueError as e:
            print(e)
        else:
            assert False

        # screening_size
        try:
            model = abessLm(screening_size=3)
            model.fit([[1]], [1])
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model = abessLm(support_size=[2], 
                            screening_size=1)
            model.fit([[1, 2, 3]], [1])
        except ValueError as e:
            print(e)
        else:
            assert False

        # primary_fit_xxx
        try:
            model = abessLogistic(primary_model_fit_max_iter=0.5)
            model.fit([[1]], [1])
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model = abessLogistic(primary_model_fit_epsilon=-1)
            model.fit([[1]], [1])
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model = abessLogistic(primary_model_fit_epsilon=-1)
            model.fit([[1]], [1])
        except ValueError as e:
            print(e)
        else:
            assert False

        # thread
        try:
            model = abessLm(thread=-1)
            model.fit([[1]], [1])
        except ValueError as e:
            print(e)
        else:
            assert False

        # splicing_type
        try:
            model = abessLm(splicing_type=-1)
            model.fit([[1]], [1])
        except ValueError as e:
            print(e)
        else:
            assert False
        
        # cv
        try:
            model = abessLm(cv=2)
            model.fit([[1]], [1])
        except ValueError as e:
            print(e)
        else:
            assert False
        
        try:
            model = abessLm(cv=2)
            cv_fold_id = [1, 1]
            model.fit([[1],[1]], [1, 1], cv_fold_id = cv_fold_id)
        except ValueError as e:
            print(e)
        else:
            assert False

        try:
            model = abessLm(cv=2)
            cv_fold_id = [1, 2, 1]
            model.fit([[1],[1]], [1, 1], cv_fold_id = cv_fold_id)
        except ValueError as e:
            print(e)
        else:
            assert False

    def test_fit(self):
        model = abessLm()
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
            model1 = abessLm(cv='c')
            model1.fit([[1]], [1])
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
            model.fit([[1,1,1]], [1], weight = [1, 2])
        except ValueError as e:
            print(e)
        else:
            assert False
        
        try:
            model.fit([[1,1,1]], [1], group = [1])
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

    def test_pca(self):
        """
        For `abess.pca.abessPCA`.
        """
        model = abessPCA()
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
            model1 = abessPCA(cv='c')
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
            model.fit([[1]], group = [1, 2])
        except ValueError as e:
            print(e)
        else:
            assert False
        
        try:
            model1 = abessPCA(support_size=np.array([1,2]))
            model1.fit([[1]])
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
            model1 = abessPCA(cv=5)
            model1.fit(Sigma=[[1]])
        except ValueError as e:
            print(e)
        else:
            assert False

        # number
        try:
            model.fit([[1]], [1], number=-1)
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
            model1 = abessPCA(ic_type='other')
            model1.fit([[1]])
        except ValueError as e:
            print(e)
        else:
            assert False
        
        try:
            model1 = abessPCA(cv=5)
            model1.fit([[1]])
        except ValueError as e:
            print(e)
        else:
            assert False

    def test_rpca(self):
        model = abessRPCA()
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
            model.fit([[1]], r=1, group = [1, 2])
        except ValueError as e:
            print(e)
        else:
            assert False
        
        # invalid arg
        try:
            model1 = abessRPCA(ic_type='other')
            model1.fit([[1]], r=1)
        except ValueError as e:
            print(e)
        else:
            assert False