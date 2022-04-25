
from abess.pybind_cabess import foo
import pytest

@pytest.mark.filterwarnings("ignore")
class TestNlopt:
    """
    Test for nlopt.
    """

    @staticmethod
    def test_nlopt():
        assert foo() == "LD_LBFGS"
