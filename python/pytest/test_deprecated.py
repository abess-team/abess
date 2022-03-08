import abess


class TestDeprecated:
    """
    Test for (future) deprecated modules in abess package.
    """

    @staticmethod
    def test_linear():
        model = abess.abessLm()
        model = abess.abessLogistic()
        model = abess.abessPoisson()
        model = abess.abessCox()
        model = abess.abessGamma()
        model = abess.abessMultigaussian()
        model = abess.abessMultinomial()

    @staticmethod
    def test_pca():
        model = abess.abessPCA()
        model = abess.abessRPCA()
