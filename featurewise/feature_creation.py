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



class AggregationTransformer:
    """
    AggregationTransformer is a class for performing aggregation operations on a DataFrame.

    Attributes:
        groupby_cols (list): Columns to group by for aggregation.
        agg_funcs (dict): Dictionary mapping columns to aggregation functions.
    """

    def __init__(self, groupby_cols, agg_funcs):
        """
        Initialize the AggregationTransformer.

        Parameters:
        groupby_cols (list): Columns to group by for aggregation.
        agg_funcs (dict): Dictionary mapping columns to aggregation functions.
                          Example: {'column_name': 'mean', 'another_column': 'sum'}

        Raises:
        ValueError: If groupby_cols is not a list or agg_funcs is not a dictionary.
        """
        if not isinstance(groupby_cols, list):
            logging.error("groupby_cols must be a list.")
            raise ValueError("groupby_cols must be a list.")
        
        if not isinstance(agg_funcs, dict):
            logging.error("agg_funcs must be a dictionary.")
            raise ValueError("agg_funcs must be a dictionary.")
        
        self.groupby_cols = groupby_cols
        self.agg_funcs = agg_funcs
        logging.info("Initialized AggregationTransformer.")

    def fit_transform(self, df):
        """
        Group by columns and aggregate according to specified functions.

        Parameters:
        df (pd.DataFrame): Input data to transform.

        Returns:
        pd.DataFrame: Transformed DataFrame with aggregated results.

        Raises:
        ValueError: If groupby_cols or agg_funcs contain columns not present in the DataFrame.
        """
        # Check if input is a valid DataFrame
        if not isinstance(df, pd.DataFrame):
            logging.error("Input data must be a pandas DataFrame.")
            raise ValueError("Input data must be a pandas DataFrame.")

        if not all(col in df.columns for col in self.groupby_cols):
            logging.error("Some groupby columns are not present in the DataFrame.")
            raise ValueError("Some groupby columns are not present in the DataFrame.")

        if not all(col in df.columns for col in self.agg_funcs.keys()):
            logging.error("Some columns in agg_funcs are not present in the DataFrame.")
            raise ValueError("Some columns in agg_funcs are not present in the DataFrame.")

        try:
            # Perform the aggregation
            aggregated_df = df.groupby(self.groupby_cols).agg(self.agg_funcs).reset_index()
        except Exception as e:
            logging.exception("Failed to perform aggregation.")
            raise ValueError(f"Failed to perform aggregation: {str(e)}")

        logging.info("Aggregation transformation successful.")
        return aggregated_df


class BinningTransformer:
    """
    BinningTransformer is a class for binning numerical columns in a DataFrame.

    Attributes:
        binning_cols (list): Columns to bin.
        strategy (str): Binning strategy - 'equal_width', 'equal_frequency', or 'custom'.
        bins (int): Number of bins for 'equal_width' or 'equal_frequency'.
        custom_bins (list): Custom bin edges for 'custom' strategy.
    """

    def __init__(self, binning_cols, strategy='equal_width', bins=10, custom_bins=None):
        """
        Initialize the BinningTransformer.

        Parameters:
        binning_cols (list): Columns to bin.
        strategy (str): Binning strategy - 'equal_width', 'equal_frequency', or 'custom'.
        bins (int): Number of bins for 'equal_width' or 'equal_frequency'.
        custom_bins (list): Custom bin edges for 'custom' strategy.

        Raises:
        ValueError: If strategy is invalid or if custom bins are not provided correctly.
        """
        if not isinstance(binning_cols, list):
            logging.error("binning_cols must be a list.")
            raise ValueError("binning_cols must be a list.")

        if strategy not in ['equal_width', 'equal_frequency', 'custom']:
            logging.error("Invalid strategy. Choose 'equal_width', 'equal_frequency', or 'custom'.")
            raise ValueError("Invalid strategy. Choose 'equal_width', 'equal_frequency', or 'custom'.")
        
        self.binning_cols = binning_cols
        self.strategy = strategy
        self.bins = bins
        self.custom_bins = custom_bins
        logging.info(f"Initialized BinningTransformer with strategy '{strategy}'.")

    def fit_transform(self, df):
        """
        Bin columns according to the specified strategy.

        Parameters:
        df (pd.DataFrame): Input data to transform.

        Returns:
        pd.DataFrame: Transformed DataFrame with binned columns.

        Raises:
        ValueError: If an invalid strategy is provided or if custom bins are not provided correctly.
        """
        if not isinstance(df, pd.DataFrame):
            logging.error("Input data must be a pandas DataFrame.")
            raise ValueError("Input data must be a pandas DataFrame.")

        for col in self.binning_cols:
            if col not in df.columns:
                logging.error(f"Column '{col}' not found in the DataFrame.")
                raise ValueError(f"Column '{col}' not found in the DataFrame.")

            try:
                if self.strategy == 'equal_width':
                    # Calculate bin edges for equal-width binning
                    df[col + '_binned'] = pd.cut(df[col], bins=self.bins, right=True, include_lowest=True)
                elif self.strategy == 'equal_frequency':
                    # Apply equal-frequency binning
                    df[col + '_binned'] = pd.qcut(df[col], q=self.bins, duplicates='drop')
                elif self.strategy == 'custom':
                    # Apply custom binning if custom bins are provided
                    if self.custom_bins is not None and isinstance(self.custom_bins, list) and len(self.custom_bins) > 1:
                        df[col + '_binned'] = pd.cut(df[col], bins=self.custom_bins, right=True, include_lowest=True)
                    else:
                        logging.error("Custom bins must be a list with at least two elements.")
                        raise ValueError("Custom bins must be a list with at least two elements.")
                logging.info(f"Successfully binned column '{col}' with strategy '{self.strategy}'.")
            except Exception as e:
                logging.exception(f"Failed to bin column '{col}'.")
                raise ValueError(f"Failed to bin column '{col}': {str(e)}")

        return df
