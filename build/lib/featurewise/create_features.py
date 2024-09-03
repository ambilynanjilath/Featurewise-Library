import logging
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd

# Set up logging configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class PolynomialFeaturesTransformer:
    """
    PolynomialFeaturesTransformer generates polynomial features from a DataFrame's numeric columns.

    Attributes:
        degree (int): Degree of the polynomial features to create.
        poly (PolynomialFeatures): Instance of PolynomialFeatures from sklearn.
    """

    def __init__(self, degree):
        """
        Initialize the PolynomialFeaturesTransformer with a specified degree.

        Parameters:
            degree (int): Degree of the polynomial features to create.

        Raises:
            ValueError: If the degree is not a positive integer.
        """
        if not isinstance(degree, int) or degree < 1:
            logging.error("Degree must be a positive integer.")
            raise ValueError("Degree must be a positive integer.")
        self.degree = degree
        self.poly = PolynomialFeatures(degree, include_bias=False)
        logging.info(f"Initialized PolynomialFeaturesTransformer with degree {degree}.")

    def fit_transform(self, df, degree=None):
        """
        Fit to data and transform it into polynomial features. Optionally update the polynomial degree.

        Parameters:
            df (pd.DataFrame): Input DataFrame to transform.
            degree (int, optional): New degree for polynomial features. If not provided, uses the initial degree.

        Returns:
            pd.DataFrame: Transformed DataFrame with polynomial features.

        Raises:
            ValueError: If degree is not a positive integer, if df is not a DataFrame, or if it contains non-numeric or categorical columns.
        """
        if degree is not None:
            if not isinstance(degree, int) or degree < 1:
                logging.error("Degree must be a positive integer.")
                raise ValueError("Degree must be a positive integer.")
            self.degree = degree
            self.poly = PolynomialFeatures(degree, include_bias=False)
            logging.info(f"Polynomial degree updated to {degree}.")

        if not isinstance(df, pd.DataFrame):
            logging.error("Input must be a pandas DataFrame.")
            raise ValueError("Input must be a pandas DataFrame.")

        # Select numeric columns only
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if len(numeric_cols) == 0:
            logging.error("No numeric columns found in the DataFrame.")
            raise ValueError("No numeric columns found in the DataFrame.")

        # Check for categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            logging.error(f"Categorical columns found: {', '.join(categorical_cols)}. PolynomialFeaturesTransformer can only process numerical data.")
            raise ValueError(f"Categorical columns found: {', '.join(categorical_cols)}. PolynomialFeaturesTransformer can only process numerical data.")

        # Filter DataFrame to include only numeric columns
        df_numeric = df[numeric_cols]

        try:
            transformed_data = self.poly.fit_transform(df_numeric)
        except Exception as e:
            logging.exception("Failed to transform data.")
            raise ValueError(f"Failed to transform data: {str(e)}")

        if transformed_data.shape[1] == 0:
            logging.error("PolynomialFeaturesTransformer produced no features. Check input data.")
            raise ValueError("PolynomialFeaturesTransformer produced no features. Check input data.")

        logging.info("Polynomial features transformation successful.")
        return pd.DataFrame(transformed_data, columns=self.poly.get_feature_names_out(df_numeric.columns))

