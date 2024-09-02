import pandas as pd
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

class DateTimeExtractor:
    """
    A class to extract and parse datetime components from a specified column in a pandas DataFrame.

    Attributes:
        df (pandas.DataFrame): The input DataFrame containing the datetime data.
        datetime_col (str): The name of the column containing datetime data.
        date_formats (list of str): List of possible datetime formats to attempt parsing.
    """
    
    def __init__(self, df, datetime_col, date_formats=None):
        """
        Initialize the DateTimeExtractor with a pandas DataFrame and the name of the datetime column.
        
        Parameters:
            df (pandas.DataFrame): The input DataFrame containing the datetime data.
            datetime_col (str): The name of the column containing datetime data.
            date_formats (list of str, optional): List of possible datetime formats to attempt parsing.
        
        Raises:
            ValueError: If the specified datetime column does not exist in the DataFrame.
        """
        if datetime_col not in df.columns:
            logging.error(f"Column '{datetime_col}' does not exist in the DataFrame.")
            raise ValueError(f"Column '{datetime_col}' does not exist in the DataFrame.")
        
        self.df = df.copy()  # Create a copy of the DataFrame to avoid modifying the original data
        self.datetime_col = datetime_col
        
        # Default date formats to try
        self.date_formats = date_formats or [
            "%m/%d/%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d",
            "%m/%d/%y %H:%M", "%Y-%m-%d %I:%M %p", "%m/%d/%Y", "%Y-%m-%d %H:%M",
            "%Y/%m/%d %H:%M:%S", "%d-%b-%y", "%d-%b-%Y", "%d-%B-%Y", "%d/%m/%Y",
            "%d-%m-%Y", "%d/%m/%y", "%d-%m-%y", "%d/%m/%y %H:%M:%S", "%d-%m-%Y %H:%M:%S",
            "%d/%m/%Y %H:%M", "%Y/%m/%d", "%d-%b-%y %I:%M %p", "%d-%b-%Y %I:%M %p",
            "%d-%B-%Y %I:%M %p", "%d/%m/%Y %I:%M %p", "%d-%m-%Y %I:%M %p", "%d/%m/%y %I:%M %p",
            "%d-%m-%y %I:%M %p", "%d/%m/%y %I:%M:%S %p", "%d-%m-%Y %I:%M:%S %p",
            "%d/%m/%Y %I:%M:%S", "%d/%m/%y %I:%M", "%d-%m-%Y %I:%M", "%d/%m/%Y"
        ]
        
        # Apply datetime parsing and handle errors
        self.df[datetime_col] = self.df[datetime_col].apply(self._parse_date)
        unparsable_dates = self.df[self.df[datetime_col].isna()][datetime_col]
        if not unparsable_dates.empty:
            logging.warning("Unparsable dates found:")
            logging.warning(unparsable_dates)
            raise ValueError("Some dates could not be parsed. Please check the date formats provided.")
    
    def _parse_date(self, date_str):
        """
        Try to parse a date string using multiple formats.
        
        Parameters:
            date_str (str): The date string to parse.
        
        Returns:
            datetime or pd.NaT: Parsed datetime object or NaT if parsing fails.
        """
        for fmt in self.date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        logging.debug(f"Failed to parse date: {date_str}")
        return pd.NaT
    
    def extract_year(self):
        """
        Extract the year from the datetime column and add it as a new column named 'year'.
        
        Returns:
            pandas.DataFrame: DataFrame with an added 'year' column.
        
        Raises:
            ValueError: If there is an error extracting the year.
        """
        try:
            self.df['year'] = self.df[self.datetime_col].dt.year
            logging.info("Extracted 'year' column successfully.")
        except Exception as e:
            logging.error(f"Error extracting year: {e}")
            raise ValueError(f"Error extracting year: {e}")
        return self.df
    
    def extract_month(self):
        """
        Extract the month from the datetime column and add it as a new column named 'month'.
        
        Returns:
            pandas.DataFrame: DataFrame with an added 'month' column.
        
        Raises:
            ValueError: If there is an error extracting the month.
        """
        try:
            self.df['month'] = self.df[self.datetime_col].dt.month
            logging.info("Extracted 'month' column successfully.")
        except Exception as e:
            logging.error(f"Error extracting month: {e}")
            raise ValueError(f"Error extracting month: {e}")
        return self.df
    
    def extract_day(self):
        """
        Extract the day from the datetime column and add it as a new column named 'day'.
        
        Returns:
            pandas.DataFrame: DataFrame with an added 'day' column.
        
        Raises:
            ValueError: If there is an error extracting the day.
        """
        try:
            self.df['day'] = self.df[self.datetime_col].dt.day
            logging.info("Extracted 'day' column successfully.")
        except Exception as e:
            logging.error(f"Error extracting day: {e}")
            raise ValueError(f"Error extracting day: {e}")
        return self.df
    
    def extract_day_of_week(self):
        """
        Extract the day of the week from the datetime column and add it as a new column named 'day_of_week'.
        
        Returns:
            pandas.DataFrame: DataFrame with an added 'day_of_week' column.
        
        Raises:
            ValueError: If there is an error extracting the day of the week.
        """
        try:
            self.df['day_of_week'] = self.df[self.datetime_col].dt.day_name()
            logging.info("Extracted 'day_of_week' column successfully.")
        except Exception as e:
            logging.error(f"Error extracting day of week: {e}")
            raise ValueError(f"Error extracting day of week: {e}")
        return self.df
    
    def extract_all(self):
        """
        Extract year, month, day, and day of the week from the datetime column and add them as new columns.
        
        Returns:
            pandas.DataFrame: DataFrame with added 'year', 'month', 'day', and 'day_of_week' columns.
        
        Raises:
            ValueError: If there is an error extracting any of the datetime components.
        """
        try:
            self.extract_year()
            self.extract_month()
            self.extract_day()
            self.extract_day_of_week()
            logging.info("Extracted all datetime components successfully.")
        except Exception as e:
            logging.error(f"Error extracting all datetime components: {e}")
            raise ValueError(f"Error extracting all datetime components: {e}")
        return self.df
