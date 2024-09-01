import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class FeatureSelection:
    """
    FeatureSelection class provides methods for selecting important features in a DataFrame using 
    various techniques like SelectKBest, Recursive Feature Elimination (RFE), and feature importance.

    Methods:
    - select_k_best: Selects the top k features based on statistical tests.
    - recursive_feature_elimination: Performs feature selection by recursively considering smaller sets of features.
    - feature_importance: Uses a RandomForest classifier to rank the importance of features.
    """

    def __init__(self, df: pd.DataFrame, target: str):
        """
        Initialize the FeatureSelection class with a pandas DataFrame and the target variable.
        
        Parameters:
        df (pd.DataFrame): The DataFrame to be processed.
        target (str): The target column name in the DataFrame.
        
        Raises:
        ValueError: If the input is not a pandas DataFrame or target is not in DataFrame columns.
        """
        if not isinstance(df, pd.DataFrame):
            logging.error("Initialization Error: Input is not a pandas DataFrame.")
            raise ValueError("Input must be a pandas DataFrame.")
        if target not in df.columns:
            logging.error(f"Initialization Error: Target '{target}' is not present in the DataFrame.")
            raise ValueError(f"Target '{target}' is not present in the DataFrame.")
        self.df = df
        self.target = target
        self.X = df.drop(target, axis=1)
        self.y = df[target]
        logging.info("FeatureSelection class initialized with a DataFrame and target variable.")

    def select_k_best(self, score_func=f_classif, k=10) -> pd.DataFrame:
        """
        Select the top k features based on a scoring function.

        Parameters:
        score_func (function): Scoring function to use, e.g., chi2, f_classif, or mutual_info_classif.
        k (int): The number of top features to select.

        Returns:
        pd.DataFrame: The DataFrame with only the top k selected features.
        """
        try:
            selector = SelectKBest(score_func=score_func, k=k)
            X_new = selector.fit_transform(self.X, self.y)
            selected_columns = self.X.columns[selector.get_support()]
            logging.info(f"SelectKBest selected the top {k} features: {', '.join(selected_columns)}.")
            return self.df[selected_columns]
        except Exception as e:
            logging.error(f"SelectKBest Error: {e}")
            raise e

    def recursive_feature_elimination(self, estimator=RandomForestClassifier(), n_features_to_select=10) -> pd.DataFrame:
        """
        Perform Recursive Feature Elimination (RFE) to select the top n features.

        Parameters:
        estimator (object): The base model to use for RFE. Default is RandomForestClassifier.
        n_features_to_select (int): The number of features to select.

        Returns:
        pd.DataFrame: The DataFrame with only the selected features.
        """
        try:
            rfe = RFE(estimator=estimator, n_features_to_select=n_features_to_select)
            rfe.fit(self.X, self.y)
            selected_columns = self.X.columns[rfe.support_]
            logging.info(f"RFE selected the top {n_features_to_select} features: {', '.join(selected_columns)}.")
            return self.df[selected_columns]
        except Exception as e:
            logging.error(f"RFE Error: {e}")
            raise e

    def feature_importance(self, model=RandomForestClassifier()) -> pd.DataFrame:
        """
        Select features based on their importance as determined by a model, usually a tree-based model.

        Parameters:
        model (object): The model to use for determining feature importance. Default is RandomForestClassifier.

        Returns:
        pd.DataFrame: The DataFrame with features ranked by their importance.
        """
        try:
            model.fit(self.X, self.y)
            importance = pd.Series(model.feature_importances_, index=self.X.columns)
            importance = importance.sort_values(ascending=False)
            logging.info(f"Features ranked by importance: {', '.join(importance.index)}.")
            return self.df[importance.index]
        except Exception as e:
            logging.error(f"Feature Importance Error: {e}")
            raise e
