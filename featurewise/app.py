import streamlit as st
import pandas as pd
from imputation import MissingValueImputation
from encoding import FeatureEncoding
from scaling import DataNormalize
from date_time_features import DateTimeExtractor
from feature_creation import PolynomialFeaturesTransformer

def main():
    st.title("Data Transformation App")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Uploaded DataFrame")
        st.dataframe(df)

        # Choose transformations
        st.sidebar.title("Transformation Options")
        transformations = st.sidebar.multiselect(
            "Select transformations to apply:",
            ["Imputation", "Encoding", "Scaling", "Datetime Features", "Feature Creation"]
        )

        # Imputation
        if "Imputation" in transformations:
            st.sidebar.header("Imputation Settings")
            strategies = {}
            for column in df.columns:
                if df[column].isnull().any():
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
                df = imputer.fit_transform(df)
                st.write("### DataFrame After Imputation")
                st.dataframe(df)

        # Encoding
        if "Encoding" in transformations:
            st.sidebar.header("Encoding Settings")
            encoding_type = st.sidebar.selectbox("Select encoding type", ["Label Encoding", "One-Hot Encoding"])
            columns = st.sidebar.multiselect("Select columns to encode", df.columns)
            if st.sidebar.button("Apply Encoding"):
                encoder = FeatureEncoding(df)
                if encoding_type == "Label Encoding":
                    df = encoder.label_encode(columns)
                elif encoding_type == "One-Hot Encoding":
                    df = encoder.one_hot_encode(columns)
                st.write("### DataFrame After Encoding")
                st.dataframe(df)

        # Scaling
        if "Scaling" in transformations:
            st.sidebar.header("Scaling Settings")
            normalizer = DataNormalize()
            method = st.sidebar.selectbox("Select scaling method", list(normalizer.scalers.keys()), index=0)
            scale_option = st.sidebar.radio("Scale the entire DataFrame or specific columns?", ("Entire DataFrame", "Specific Columns"))
            if scale_option == "Specific Columns":
                columns = st.sidebar.multiselect("Select columns to scale", df.columns)
                if st.sidebar.button("Apply Scaling"):
                    df = normalizer.scale_columns(df, columns, method)
                    st.write("### DataFrame After Scaling")
                    st.dataframe(df)
            else:
                if st.sidebar.button("Apply Scaling"):
                    df = normalizer.scale(df, method)
                    st.write("### DataFrame After Scaling")
                    st.dataframe(df)

        # Datetime Features
        if "Datetime Features" in transformations:
            st.sidebar.header("Datetime Features Settings")
            datetime_col = st.sidebar.selectbox("Select the datetime column", df.columns)
            extract_options = st.sidebar.multiselect("Select extraction(s)", ["Year", "Month", "Day", "Day of Week", "All"])
            if st.sidebar.button("Apply Datetime Transformations"):
                datetime_extractor = DateTimeExtractor(df, datetime_col)
                if "All" in extract_options:
                    df = datetime_extractor.extract_all()
                else:
                    if "Year" in extract_options:
                        df = datetime_extractor.extract_year()
                    if "Month" in extract_options:
                        df = datetime_extractor.extract_month()
                    if "Day" in extract_options:
                        df = datetime_extractor.extract_day()
                    if "Day of Week" in extract_options:
                        df = datetime_extractor.extract_day_of_week()
                st.write("### DataFrame After Datetime Features Extraction")
                st.dataframe(df)

        # Feature Creation
        if "Feature Creation" in transformations:
            st.sidebar.header("Feature Creation Settings")
            poly_degree = st.sidebar.number_input("Degree of polynomial features", min_value=1, value=2)
            poly_columns = st.sidebar.multiselect("Select columns for polynomial features", df.columns)
            if st.sidebar.button("Apply Polynomial Features"):
                if poly_columns:
                    poly_transformer = PolynomialFeaturesTransformer(degree=poly_degree)
                    df_poly = poly_transformer.fit_transform(df[poly_columns], degree=poly_degree)
                    df = pd.concat([df.drop(columns=poly_columns), df_poly], axis=1)
                    st.write("### DataFrame After Polynomial Features")
                    st.dataframe(df)
                else:
                    st.error("Please select at least one column for polynomial features.")

        # Download transformed DataFrame
        st.sidebar.markdown("### Download Transformed Data")
        csv = df.to_csv(index=False)
        st.sidebar.download_button("Download Transformed CSV", csv, "transformed_data.csv", "text/csv")

if __name__ == "__main__":
    main()
