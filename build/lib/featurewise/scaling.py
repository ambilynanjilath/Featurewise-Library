import pandas as pd
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler,
                                   MaxAbsScaler, Normalizer, QuantileTransformer,
                                   PowerTransformer)
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataNormalize:
    """
    DataNormalize class provides methods for scaling DataFrame features using various scalers from scikit-learn.
    
    Methods:
    - scale: General method to fit and transform data using a specified scaler.
    - scale_columns: General method to fit and transform specific columns using a specified scaler.

    Suitable for:
    - 'standard': StandardScaler - scales data to have mean=0 and variance=1.
    - 'minmax': MinMaxScaler - scales data to be within a specified range (default 0 to 1).
    - 'robust': RobustScaler - scales data using statistics that are robust to outliers.
    - 'maxabs': MaxAbsScaler - scales data to the [-1, 1] range based on the maximum absolute value.
    - 'l2': Normalizer - scales samples individually to have unit norm (L2 norm).
    - 'quantile': QuantileTransformer - transforms features to follow a uniform or normal distribution.
    - 'power': PowerTransformer - applies a power transformation to make data more Gaussian-like.
    """

    def __init__(self):
        """
        Initialize the DataNormalize object with various scaler objects from scikit-learn.
        """
        self.scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler(),
            'maxabs': MaxAbsScaler(),
            'l2': Normalizer(norm='l2'),
            'quantile': QuantileTransformer(output_distribution='normal'),
            'power': PowerTransformer(method='yeo-johnson')
        }
        self.logger = logging.getLogger(__name__)

    def _check_dataframe(self, df: pd.DataFrame):
        """
        Check if the input is a pandas DataFrame.

        Parameters:
        df (pd.DataFrame): The DataFrame to check.

        Raises:
        ValueError: If the input is not a pandas DataFrame.

        """
        if not isinstance(df, pd.DataFrame):
            self.logger.error("Input must be a pandas DataFrame.")
            raise ValueError("Input must be a pandas DataFrame.")
    
    def _check_columns(self, df: pd.DataFrame, columns: list):
        """
        Check if all specified columns are present in the DataFrame.

        Parameters:
        df (pd.DataFrame): The DataFrame containing the columns.
        columns (list): The list of columns to check.

        Raises:
        ValueError: If any of the specified columns are not present in the DataFrame.

       
        """
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            self.logger.error(f"Columns not present in the DataFrame: {missing_cols}")
            raise ValueError(f"Some columns are not present in the DataFrame: {missing_cols}")
    
    def scale(self, df: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """
        Scales the entire DataFrame using the specified method.
        
        Parameters:
        df (pd.DataFrame): The DataFrame to scale.
        method (str): The scaling method to use. Options: 'standard', 'minmax', 'robust', 'maxabs', 'l2', 'quantile', 'power'.
        
        Returns:
        pd.DataFrame: The scaled DataFrame.

        Raises:
        ValueError: If the input is not a pandas DataFrame or if an invalid scaling method is provided.

        """
        self._check_dataframe(df)
        if method not in self.scalers:
            self.logger.error(f"Invalid method '{method}'. Choose from {list(self.scalers.keys())}.")
            raise ValueError(f"Invalid method. Choose from {list(self.scalers.keys())}.")
        
        self.logger.info(f"Scaling entire DataFrame using '{method}' method.")
        scaler = self.scalers[method]
        numeric_df = df.select_dtypes(include=['number'])
        scaled_data = scaler.fit_transform(numeric_df)
        scaled_df = pd.DataFrame(scaled_data, columns=numeric_df.columns, index=df.index)
        result_df = pd.concat([scaled_df, df.select_dtypes(exclude=['number'])], axis=1)
        self.logger.info("Scaling completed.")
        return result_df

    def scale_columns(self, df: pd.DataFrame, columns: list, method: str = 'standard') -> pd.DataFrame:
        """
        Scales specific columns of the DataFrame using the specified method.
        
        Parameters:
        df (pd.DataFrame): The DataFrame containing the columns to scale.
        columns (list): The list of columns to scale.
        method (str): The scaling method to use. Options: 'standard', 'minmax', 'robust', 'maxabs', 'l2', 'quantile', 'power'.
        
        Returns:
        pd.DataFrame: The DataFrame with specified columns scaled.

        Raises:
        ValueError: If the input is not a pandas DataFrame, if any of the specified columns are not present in the DataFrame, or if an invalid scaling method is provided.

        """
        self._check_dataframe(df)
        self._check_columns(df, columns)
        if method not in self.scalers:
            self.logger.error(f"Invalid method '{method}'. Choose from {list(self.scalers.keys())}.")
            raise ValueError(f"Invalid method. Choose from {list(self.scalers.keys())}.")
        
        self.logger.info(f"Scaling columns {columns} using '{method}' method.")
        df_copy = df.copy()
        scaler = self.scalers[method]
        df_copy[columns] = scaler.fit_transform(df[columns])
        self.logger.info("Column scaling completed.")
        return df_copy
