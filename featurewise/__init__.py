# FEATURE_WISE_PROJECT/featurewise/__init__.py


# Import the main classes from each module
from .date_time_features import DateTimeExtractor
from .encoding import FeatureEncoding
from .imputation import MissingValueImputation
from .scaling import DataNormalize
from .feature_creation import PolynomialFeaturesTransformer,BinningTransformer,AggregationTransformer

# Optional: Define __all__ for controlled imports
__all__ = [
    'DateTimeExtractor',
    'FeatureEncoding',
    'MissingValueImputation',
    'DataNormalize',
    'PolynomialFeaturesTransformer',
    'BinningTransformer',
    'AggregationTransformer'
]

