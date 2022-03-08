import abess


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
