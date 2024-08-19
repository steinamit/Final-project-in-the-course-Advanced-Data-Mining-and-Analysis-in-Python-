import pandas as pd
from sklearn.impute import SimpleImputer

def prepare_data(raw_df, target='Price'):
    df = raw_df.copy()

    def clean_numeric(x):
        """Cleans numeric values in a DataFrame column."""
        if isinstance(x, str):
            return pd.to_numeric(x.replace(',', ''), errors='coerce')
        return x

    def clean_and_impute_numeric_columns(df, numeric_columns):
        """Cleans and imputes numeric columns."""
        for col in numeric_columns:
            df[col] = df[col].apply(clean_numeric)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())
        return df
    
    def remove_manufactor_from_model(df):
        """Removes the manufactor name from the model column."""
        df['model'] = df.apply(lambda x: x['model'].replace(x['manufactor'], '').strip() if pd.notna(x['model']) and pd.notna(x['manufactor']) else x['model'], axis=1)
        return df
    
    def impute_categorical_columns(df):
        """Imputes categorical columns with a constant value."""
        categorical_columns = df.select_dtypes(include=['object']).columns
        categorical_imputer = SimpleImputer(strategy='constant', fill_value='Unknown')
        df[categorical_columns] = categorical_imputer.fit_transform(df[categorical_columns])
        return df

    def convert_date_columns(df, date_columns):
        """Converts specified columns to datetime."""
        for col in date_columns:
            df[col] = pd.to_datetime(df[col], format='%d/%m/%Y', errors='coerce')
        return df
    
    
    def remove_outliers(df, column):
        """Removes outliers from a specified numeric column using the IQR method."""
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    
    def create_season_column(df, date_column):
        """Creates a season column based on the month of a date column."""
        df['Season'] = df[date_column].dt.month.map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        }).fillna('Unknown')
        return df
    
    def create_model_manufactor_feature(df):
        df['Model_Manufactor'] = df['model'].astype(str) + '_' + df['manufactor'].astype(str)
        return df    
    
    def create_derived_features(df):
        """Creates derived features such as Age, Km_per_year, and Age_Hand_interaction."""
        current_year = pd.Timestamp.now().year
        df['Age'] = current_year - df['Year']
        df['Age_Hand_interaction'] = df['Age'] * df['Hand']
        return df

    def drop_unnecessary_columns(df, columns_to_drop):
        """Drops unnecessary columns from the DataFrame."""
        return df.drop(columns=columns_to_drop, errors='ignore')

    # Define numeric columns
    numeric_columns = ['capacity_Engine', 'Km', 'Pic_num', 'Year', 'Hand', target]

    # Clean and impute numeric columns
    df = clean_and_impute_numeric_columns(df, numeric_columns)

    # Impute categorical columns
    df = impute_categorical_columns(df)
    
     # Remove outliers for 'Year'
    df = remove_outliers(df, 'Year')
    
     # Remove manufactor from model
    df = remove_manufactor_from_model(df)
    
    # Convert date columns
    date_columns = ['Cre_date', 'Repub_date']
    df = convert_date_columns(df, date_columns)

      
    # Drop unnecessary columns
    columns_to_drop = ['Cre_date', 'Repub_date', 'Description', 'Supply_score', 'Test']
    df = drop_unnecessary_columns(df, columns_to_drop)

    return df
