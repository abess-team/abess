import abess
import pytest


@pytest.mark.filterwarnings("error::FutureWarning")
class TestDeprecated:
    """
    Test for (future) deprecated modules in abess package.
    """

    @staticmethod
    def test_linear():
        try:
            abess.abessLm()
        except FutureWarning as e:
            print(e)
        else:
            assert False

        try:
            abess.abessLogistic()
        except FutureWarning as e:
            print(e)
        else:
            assert False

        try:
            abess.abessPoisson()
        except FutureWarning as e:
            print(e)
        else:
            assert False

        try:
            abess.abessCox()
        except FutureWarning as e:
            print(e)
        else:
            assert False

        try:
            abess.abessGamma()
        except FutureWarning as e:
            print(e)
        else:
            assert False

        try:
            abess.abessMultigaussian()
        except FutureWarning as e:
            print(e)
        else:
            assert False

        try:
            abess.abessMultinomial()
        except FutureWarning as e:
            print(e)
        else:
            assert False

    @staticmethod
    def test_pca():
        try:
            abess.abessPCA()
        except FutureWarning as e:
            print(e)
        else:
            assert False

        try:
            abess.abessRPCA()
        except FutureWarning as e:
            print(e)
        else:
            assert False
