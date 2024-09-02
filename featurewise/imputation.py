import numpy as np
import pandas as pd
from scipy.stats import mode
from typing import Dict, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MissingValueImputation:
    def __init__(self, strategies: Dict[str, Union[str, int, float]] = None):
        """
        Initialize the MissingValueImputation class with a dictionary of strategies.
        
        Parameters:
        strategies (dict): A dictionary where keys are column names and values are 
                           the imputation strategies ('mean', 'median', 'mode', or a custom number or value).
        
        Steps:
        1. Set the strategies attribute to the provided dictionary or an empty dictionary if none is provided.
        2. Initialize an empty dictionary for storing computed fill values.
        3. Create a logger for the class.
        """
        self.strategies = strategies or {}  # If no strategies are provided, initialize an empty dictionary
        self.fill_values = {}  # Initialize an empty dictionary to store fill values
        self.logger = logging.getLogger(__name__)  # Create a logger for this class

    def _compute_fill_value(self, df: pd.DataFrame, column: str, strategy: Union[str, int, float]) -> Union[float, str]:
        """
        Compute the fill value based on the imputation strategy for a given column.
        
        Parameters:
        df (pd.DataFrame): The dataframe to compute fill values from.
        column (str): The column name for which to compute the fill value.
        strategy (Union[str, int, float]): The imputation strategy to use ('mean', 'median', 'mode', or a custom number or value).
        
        Returns:
        Union[float, str]: The computed fill value based on the strategy.
        
        Raises:
        ValueError: If the strategy is not applicable or unsupported.
        
        Steps:
        1. Check if the strategy is a numeric value (int or float) and return it directly.
        2. If the strategy is 'mean', calculate and return the mean value if the column is numeric.
        3. If the strategy is 'median', calculate and return the median value if the column is numeric.
        4. If the strategy is 'mode', return the mode of the column.
        5. Raise an error if the strategy is unsupported or not applicable to the column type.
        """
        self.logger.debug(f"Computing fill value for column '{column}' with strategy '{strategy}'")
        if isinstance(strategy, (int, float)):
            return strategy
        elif strategy == 'mean':
            if df[column].dtype.kind in 'bifc':
                return df[column].mean()
            else:
                self.logger.error(f"Mean strategy is not applicable for non-numeric column '{column}'.")
                raise ValueError(f"Mean strategy is not applicable for non-numeric column {column}.")
        elif strategy == 'median':
            if df[column].dtype.kind in 'bifc':
                return df[column].median()
            else:
                self.logger.error(f"Median strategy is not applicable for non-numeric column '{column}'.")
                raise ValueError(f"Median strategy is not applicable for non-numeric column {column}.")
        elif strategy == 'mode':
            return df[column].mode()[0]
        else:
            self.logger.error(f"Strategy '{strategy}' not supported.")
            raise ValueError(f"Strategy {strategy} not supported. Please use 'mean', 'median', 'mode', or a custom number or value.")

    def fit(self, df: pd.DataFrame) -> 'MissingValueImputation':
        """
        Compute the fill values for missing data based on the provided strategies.
        
        Parameters:
        df (pd.DataFrame): The dataframe to compute fill values from.
        
        Returns:
        self: The instance of MissingValueImputation with computed fill values.
        
        Raises:
        ValueError: If a strategy is not applicable for a column or if a column is not present in the DataFrame.
        
        Steps:
        1. Log the fitting process.
        2. Iterate through the columns specified in the strategies.
        3. Check if each column exists in the DataFrame.
        4. Compute the fill value for each column using the specified strategy.
        5. Store the computed fill values in the fill_values attribute.
        """
        self.logger.info("Fitting MissingValueImputation with the provided DataFrame.")
        for column, strategy in self.strategies.items():
            if column not in df.columns:
                self.logger.error(f"Column '{column}' is not present in the DataFrame.")
                raise ValueError(f"Column {column} is not present in the DataFrame.")
            self.fill_values[column] = self._compute_fill_value(df, column, strategy)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the imputation to the dataframe using the computed fill values.
        
        Parameters:
        df (pd.DataFrame): The dataframe to apply imputation to.
        
        Returns:
        pd.DataFrame: The dataframe with missing values filled.
        
        Raises:
        ValueError: If a column in the strategies is not present in the DataFrame.
        
        Steps:
        1. Log the transformation process.
        2. Iterate through the columns in the fill_values attribute.
        3. Check if each column exists in the DataFrame.
        4. Fill missing values in each column using the corresponding fill value.
        5. Log the fill values used for each column.
        """
        self.logger.info("Transforming DataFrame using the computed fill values.")
        for column, fill_value in self.fill_values.items():
            if column not in df.columns:
                self.logger.error(f"Column '{column}' is not present in the DataFrame.")
                raise ValueError(f"Column {column} is not present in the DataFrame.")
            df[column] = df[column].fillna(fill_value)
            self.logger.debug(f"Filled missing values in column '{column}' with '{fill_value}'.")
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the fill values and apply the imputation to the dataframe in one step.
        
        Parameters:
        df (pd.DataFrame): The dataframe to fit and transform.
        
        Returns:
        pd.DataFrame: The dataframe with missing values filled.
        
        Raises:
        ValueError: If a strategy is not applicable for a column or if a column is not present in the DataFrame.
        
        Steps:
        1. Log the fit and transform process.
        2. Call the fit method to compute the fill values.
        3. Call the transform method to apply the computed fill values to the DataFrame.
        """
        self.logger.info("Fitting and transforming DataFrame in one step.")
        self.fit(df)  # Compute the fill values
        return self.transform(df)  # Apply the imputation
