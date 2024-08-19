import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
import pickle
from car_data_prep import prepare_data


# Load data
raw_url = 'dataset (2).csv'
df = pd.read_csv(raw_url, engine='python')

# Prepare data
prepared_df = prepare_data(df)

# Define columns
cat_columns = ['manufactor', 'model', 'Gear', 'Engine_type', 'Area', 'City', 'Color', 'Prev_ownership', 'Curr_ownership']
numeric_columns = ['capacity_Engine', 'Km', 'Pic_num', 'Year', 'Hand']


X = prepared_df.drop(columns=['Price'])
y = prepared_df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_columns),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_columns)
    ])

# Fit and transform the data
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Get feature names
feature_names_cat = []
for i, col in enumerate(cat_columns):
    feature_names_cat.extend([f"{col}_{val}" for val in preprocessor.named_transformers_['cat'].categories_[i]])
feature_names = numeric_columns + feature_names_cat

# Define the model
model = ElasticNet(random_state=42)

# Define parameter grid for GridSearchCV
param_grid = {
    'alpha': [0.1, 0.5, 1.0, 1.5],
    'l1_ratio': [0.3, 0.5, 0.7, 0.9]
}

# Perform GridSearchCV to find best parameters
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train_transformed, y_train)

# Use best parameters to evaluate model using cross-validation on training set
best_model = grid_search.best_estimator_
cv = KFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(best_model, X_train_transformed, y_train, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)

# Fit model on training data using best parameters
best_model.fit(X_train_transformed, y_train)

# Predict on test set
y_pred = best_model.predict(X_test_transformed)
mse = mean_squared_error(y_test, y_pred)

# Calculate permutation importance using training set
importance = permutation_importance(best_model, X_train_transformed, y_train, n_repeats=10, random_state=42)
feature_importances = importance.importances_mean

# Create a dictionary for permutation importance
importance_dict = dict(zip(feature_names, feature_importances))
sorted_importances = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)


import pickle
pickle.dump(best_model, open("trained_model.pkl", "wb"))
pickle.dump(preprocessor, open('preprocessor.pkl', 'wb'))