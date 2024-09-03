# PROJECT/featurewise/__init__.py

# Import the main classes from each module
from .date_time_features import DateTimeExtractor
from .encoding import FeatureEncoding
from .imputation import MissingValueImputation
from .scaling import DataNormalize
from .create_features import PolynomialFeaturesTransformer

# Optional: Define __all__ for controlled imports
__all__ = [
    'DateTimeExtractor',
    'FeatureEncoding',
    'MissingValueImputation',
    'DataNormalize',
    'PolynomialFeaturesTransformer'
]

# Metadata
__version__ = '1.0.0'
__author__ = 'Ambily Biju'
__email__ = 'ambilybiju2408@gmail.com'
__description__ = 'A no-code solution for performing data transformations like imputation, encoding, scaling, and feature creation, with an intuitive interface for interactive DataFrame manipulation and easy CSV export.'



