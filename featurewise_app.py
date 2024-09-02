import streamlit as st
import pandas as pd
from featurewise.imputation import MissingValueImputation
from featurewise.encoding import FeatureEncoding
from featurewise.scaling import DataNormalize
from featurewise.date_time_features import DateTimeExtractor
from featurewise.create_features import PolynomialFeaturesTransformer
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

def display_dataframe(df):
    """
    Displays a DataFrame in an interactive grid using AgGrid.

    This function uses AgGrid to present the DataFrame in a user-friendly grid format.
    It allows users to interact with the data, including sorting and pagination.
    The function returns the updated DataFrame after any interactions in the grid.

    AgGrid
    ----------
    AgGrid is an advanced data grid library for displaying and interacting with data in web applications. 
    It provides features like sorting, filtering, pagination, and editing in a highly customizable and 
    performant grid. It is often used in applications where users need to interact with large datasets in 
    a user-friendly manner.

    Streamlit
    -----------
    Streamlit is an open-source framework for creating interactive web applications with Python. 
    It simplifies the process of building data-driven web apps, allowing developers to create interactive 
    interfaces and visualizations with minimal code. Streamlit is particularly popular for creating dashboards
    and tools for data exploration, machine learning model deployment, and data analysis.

    Args:
        df (pd.DataFrame): The DataFrame to be displayed.

    Returns:
        pd.DataFrame: The updated DataFrame after interactions in the grid.
    """
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=10)  # Pagination settings
    gb.configure_side_bar()  # Enable sidebar for column selection
    grid_options = gb.build()

    response = AgGrid(
        df,
        gridOptions=grid_options,
        enable_enterprise_modules=True,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        theme="streamlit"
    )
    
    return response['data']

def main():
    """
    The main function to run the Streamlit app for data transformation.

    This function provides a user interface for uploading a CSV file, applying various data transformations,
    and downloading the transformed data. It includes:
    - Displaying a logo and file uploader.
    - Handling file uploads and displaying the DataFrame.
    - Providing options for column deletion, imputation, encoding, scaling, datetime feature extraction, and polynomial feature creation.
    - Allowing users to download the transformed DataFrame as a CSV file.

    It also includes error handling for file reading and transformation steps.
    """
    st.markdown("""
        <style>
            .file-uploader {
                background-color: transparent;
                padding: 0;
                margin-bottom: 20px;
            }
        </style>
    """, unsafe_allow_html=True)

    st.image('/home/user/Documents/Datahut_Internship/project/package_logo.jpeg', width=200, use_column_width=False)  # Display logo

    uploaded_file = st.file_uploader("", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read CSV file into DataFrame
            if 'df' not in st.session_state:
                st.session_state.df = pd.read_csv(uploaded_file)

            st.write("### Uploaded DataFrame")
            st.session_state.df = display_dataframe(st.session_state.df)

        except pd.errors.EmptyDataError:
            st.error("The uploaded file is empty. Please upload a valid CSV file.")
            return
        except pd.errors.ParserError:
            st.error("There was an issue parsing the CSV file. Please ensure it is formatted correctly.")
            return
        except Exception as e:
            st.error(f"Error reading the CSV file: {e}")
            return

        # Choose transformations
        st.sidebar.title("Transformation Toolbox")
        transformations = st.sidebar.multiselect(
            "Select tools to apply:",
            [
                "Delete Columns", 
                "Imputation", 
                "Encoding", 
                "Scaling", 
                "Datetime Features", 
                "Feature Creation"
            ]
        )

        # Delete Columns
        if "Delete Columns" in transformations:
            st.sidebar.header("Delete Columns")
            columns_to_delete = st.sidebar.multiselect("Select columns to delete", st.session_state.df.columns)
            if st.sidebar.button("Apply Column Deletion"):
                st.session_state.df.drop(columns=columns_to_delete, inplace=True)
                st.write("### DataFrame After Column Deletion")
                st.session_state.df = display_dataframe(st.session_state.df)

        # Imputation
        if "Imputation" in transformations:
            st.sidebar.header("Imputation Settings")
            strategies = {}
            try:
                for column in st.session_state.df.columns:
                    if st.session_state.df[column].isnull().any():
                        strategy = st.sidebar.selectbox(
                            f"Select strategy for {column}",
                            ["mean", "median", "mode", "custom"],
                            key=column
                        )
                        if strategy == "custom":
                            custom_value = st.sidebar.number_input(f"Enter custom value for {column}", key=f"{column}_custom")
                            strategies[column] = custom_value
                        else:
                            strategies[column] = strategy
                if st.sidebar.button("Apply Imputation"):
                    imputer = MissingValueImputation(strategies=strategies)
                    st.session_state.df = imputer.fit_transform(st.session_state.df)
                    st.write("### DataFrame After Imputation")
                    st.session_state.df = display_dataframe(st.session_state.df)
            except KeyError as e:
                st.error(f"Column not found: {e}")
            except Exception as e:
                st.error(f"An error occurred during imputation: {e}")

        # Encoding
        if "Encoding" in transformations:
            st.sidebar.header("Encoding Settings")
            try:
                encoding_type = st.sidebar.selectbox("Select encoding type", ["Label Encoding", "One-Hot Encoding"])
                columns = st.sidebar.multiselect("Select columns to encode", st.session_state.df.columns)
                if st.sidebar.button("Apply Encoding"):
                    encoder = FeatureEncoding(st.session_state.df)
                    if encoding_type == "Label Encoding":
                        st.session_state.df = encoder.label_encode(columns)
                    elif encoding_type == "One-Hot Encoding":
                        st.session_state.df = encoder.one_hot_encode(columns)
                    st.write("### DataFrame After Encoding")
                    st.session_state.df = display_dataframe(st.session_state.df)
            except KeyError as e:
                st.error(f"Column not found: {e}")
            except TypeError as e:
                st.error(f"Type error during encoding: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred during encoding: {e}")

        # Scaling
        if "Scaling" in transformations:
            st.sidebar.header("Scaling Settings")
            try:
                normalizer = DataNormalize()
                method = st.sidebar.selectbox("Select scaling method", list(normalizer.scalers.keys()), index=0)
                scale_option = st.sidebar.radio("Scale the entire DataFrame or specific columns?", ("Entire DataFrame", "Specific Columns"))
                if scale_option == "Specific Columns":
                    columns = st.sidebar.multiselect("Select columns to scale", st.session_state.df.columns)
                    if st.sidebar.button("Apply Scaling"):
                        if not all(st.session_state.df[col].dtype in ['int64', 'float64'] for col in columns):
                            raise TypeError("Selected columns must be numeric for scaling.")
                        st.session_state.df = normalizer.scale_columns(st.session_state.df, columns, method)
                        st.write("### DataFrame After Scaling")
                        st.session_state.df = display_dataframe(st.session_state.df)
                else:
                    if st.sidebar.button("Apply Scaling"):
                        st.session_state.df = normalizer.scale(st.session_state.df, method)
                        st.write("### DataFrame After Scaling")
                        st.session_state.df = display_dataframe(st.session_state.df)
            except KeyError as e:
                st.error(f"Column not found: {e}")
            except TypeError as e:
                st.error(f"Type error during scaling: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred during scaling: {e}")

        # Datetime Features
        if "Datetime Features" in transformations:
            st.sidebar.header("Datetime Features Settings")
            try:
                datetime_col = st.sidebar.selectbox("Select the datetime column", st.session_state.df.columns)
                extract_options = st.sidebar.multiselect("Select extraction(s)", ["Year", "Month", "Day", "Day of Week", "All"])
                if st.sidebar.button("Apply Datetime Transformations"):
                    datetime_extractor = DateTimeExtractor(st.session_state.df, datetime_col)
                    if "All" in extract_options:
                        st.session_state.df = datetime_extractor.extract_all()
                    else:
                        st.session_state.df = datetime_extractor.extract_features(extract_options)
                    st.write("### DataFrame After Datetime Feature Extraction")
                    st.session_state.df = display_dataframe(st.session_state.df)
            except KeyError as e:
                st.error(f"Column not found: {e}")
            except Exception as e:
                st.error(f"An error occurred during datetime feature extraction: {e}")

        # Polynomial Features
        if "Feature Creation" in transformations:
            st.sidebar.header("Polynomial Features Settings")
            try:
                poly_columns = st.sidebar.multiselect("Select columns for polynomial feature creation", st.session_state.df.columns)
                degree = st.sidebar.slider("Select the degree of polynomial features", 2, 5, 2)
                if st.sidebar.button("Apply Feature Creation"):
                    if poly_columns:
                        poly_transformer = PolynomialFeaturesTransformer(degree=degree)
                        df_poly = poly_transformer.fit_transform(st.session_state.df[poly_columns])
                        st.session_state.df = pd.concat([st.session_state.df, df_poly], axis=1)
                        st.write("### DataFrame After Polynomial Feature Creation")
                        st.session_state.df = display_dataframe(st.session_state.df)
                    else:
                        st.warning("No columns selected for polynomial feature creation.")
            except KeyError as e:
                st.error(f"Column not found: {e}")
            except Exception as e:
                st.error(f"An error occurred during feature creation: {e}")

        # Download Button
        st.sidebar.download_button(
            "Download Transformed CSV",
            data=st.session_state.df.to_csv(index=False),
            file_name="transformed_data.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
