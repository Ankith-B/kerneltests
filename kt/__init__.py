"""kerneltests: unified, tested implementations of modern kernel hypothesis tests."""
from .api import TestResult, two_sample, goodness_of_fit, independence

__version__ = "0.1.0.dev0"
__all__ = ["TestResult", "two_sample", "goodness_of_fit", "independence", "__version__"]
