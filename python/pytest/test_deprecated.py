import abess
import pytest


@pytest.mark.filterwarnings("ignore")
class TestDeprecated:
    """
    Test for (future) deprecated modules in abess package.
    """

    @staticmethod
    def test_linear():
        abess.abessLm()
        abess.abessLogistic()
        abess.abessPoisson()
        abess.abessCox()
        abess.abessGamma()
        abess.abessMultigaussian()
        abess.abessMultinomial()

    @staticmethod
    def test_pca():
        abess.abessPCA()
        abess.abessRPCA()

@pytest.mark.filterwarnings("error::FutureWarning")
class TestDeprecatedWarning:
    """
    Test for (future) deprecated modules' warnings in abess package.
    """
    @staticmethod
    def test_warning():
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
