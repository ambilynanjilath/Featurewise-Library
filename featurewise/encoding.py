import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class FeatureEncoding:
    """
    FeatureEncoding class provides methods for encoding categorical features in a DataFrame using 
    Label Encoding and One-Hot Encoding.

    Methods:
    - label_encode: Applies Label Encoding to specified columns.
    - one_hot_encode: Applies One-Hot Encoding to specified columns and concatenates the results with the original DataFrame.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the FeatureEncoding class with a pandas DataFrame.
        
        Parameters:
        df (pd.DataFrame): The DataFrame to be processed.
        
        Raises:
        ValueError: If the input is not a pandas DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            logging.error("Initialization Error: Input is not a pandas DataFrame.")
            raise ValueError("Input must be a pandas DataFrame.")
        self.df = df
        logging.info("FeatureEncoding class initialized with a DataFrame.")

    def label_encode(self, columns: list) -> pd.DataFrame:
        """
        Apply Label Encoding to the specified columns.

        Label Encoding converts categorical text data into numerical data by 
        assigning a unique integer to each category. This is useful for algorithms 
        that prefer or require numerical input.

        Parameters:
        columns (list): List of column names to apply label encoding to.
        
        Returns:
        pd.DataFrame: The DataFrame with label encoded columns.
        
        Raises:
        ValueError: If any column in the list is not present in the DataFrame or is not of object type.
        """
        # Initialize the LabelEncoder
        encoder = LabelEncoder()
        for column in columns:
            if column not in self.df.columns:
                logging.error(f"Label Encoding Error: Column '{column}' is not present in the DataFrame.")
                raise ValueError(f"Column '{column}' is not present in the DataFrame.")
            if self.df[column].dtype != 'object':
                logging.error(f"Label Encoding Error: Column '{column}' is not of object type.")
                raise ValueError(f"Column '{column}' is not of object type.")
            
            # Apply Label Encoding
            try:
                self.df[column] = encoder.fit_transform(self.df[column])
                logging.info(f"Label Encoding applied to column '{column}'.")
            except Exception as e:
                logging.error(f"Label Encoding Error for column '{column}': {e}")
                raise e
        return self.df

    def one_hot_encode(self, columns: list) -> pd.DataFrame:
        """
        Apply One-Hot Encoding to the specified columns, concatenate the encoded columns 
        with the original DataFrame, and drop the original columns.

        One-Hot Encoding converts categorical data into a binary matrix, creating a new 
        column for each unique category. Each column contains a binary value indicating 
        the presence of the category.

        Parameters:
        columns (list): List of column names to apply one-hot encoding to.
        
        Returns:
        pd.DataFrame: The DataFrame with one-hot encoded columns.
        
        Raises:
        ValueError: If any column in the list is not present in the DataFrame or is not of object type.
        """
        # Initialize the OneHotEncoder with sparse_output set to False and drop set to 'first'
        encoder = OneHotEncoder(sparse_output=False, drop='first')
        
        for column in columns:
            if column not in self.df.columns:
                logging.error(f"One-Hot Encoding Error: Column '{column}' is not present in the DataFrame.")
                raise ValueError(f"Column '{column}' is not present in the DataFrame.")
            if self.df[column].dtype != 'object':
                logging.error(f"One-Hot Encoding Error: Column '{column}' is not of object type.")
                raise ValueError(f"Column '{column}' is not of object type.")
        
        try:
            # Fit and transform the specified columns to one-hot encoded format
            encoded_cols = encoder.fit_transform(self.df[columns])
            
            # Convert the encoded columns to integers
            encoded_cols = encoded_cols.astype(int)
            
            # Create a new DataFrame with the encoded columns, naming the columns appropriately
            encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(columns))
            
            # Concatenate the original DataFrame (excluding the specified columns) with the new encoded DataFrame
            self.df = pd.concat([self.df.drop(columns, axis=1), encoded_df], axis=1)
            
            logging.info(f"One-Hot Encoding applied to columns: {', '.join(columns)}.")
        except Exception as e:
            logging.error(f"One-Hot Encoding Error: {e}")
            raise e
        
        return self.df
