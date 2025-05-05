import pandas as pd

# Load and clean data

# @filepath = path to the CSV file containing the data.
# @features = list of features to be extracted from the data.
# @return = cleaned DataFrame ready for further processing.
def load_and_clean_data(filepath, features):
    df = pd.read_csv(filepath) # DataFrame containing the loaded data.
    data_clean = df[features].dropna().reset_index(drop=True) # DataFrame containing the cleaned data with selected features and no missing values.
    return data_clean
