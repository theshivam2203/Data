import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from math import sqrt
from sklearn.preprocessing import OrdinalEncoder
import joblib

# Load the Excel file
file_path = r'main.output.xlsx'
df = pd.read_excel(file_path)

# Define input and output columns
input_columns = [
    'Technology Acceptance',
    'Level of use of AI based tools',
    'Technology based Tutoring System',
    'Organisational Performance',
    'Student\'s Performance'
]

output_columns = [
    'Technology_Acceptance_Range',
    'Level_of_use_of_AI_based_tools_Range',
    'Technology_based_Tutoring_System_Range',
    'Organisational_Performance_Range',
    'Student\'s_Performance_Range'
]

# Feature engineering
# Exclude non-numeric columns
numeric_columns = df.select_dtypes(include=['number']).columns

# Create interaction terms
interaction_terms = {}
for i in range(len(numeric_columns)):
    for j in range(i+1, len(numeric_columns)):
        interaction_terms[f"{numeric_columns[i]}_{numeric_columns[j]}"] = df[numeric_columns[i]] * df[numeric_columns[j]]

# Convert interaction terms dictionary to DataFrame
interaction_terms_df = pd.DataFrame(interaction_terms)

# Create polynomial features (squared terms)
polynomial_features = {}
for column in numeric_columns:
    polynomial_features[f"{column}_squared"] = df[column] ** 2

# Convert polynomial features dictionary to DataFrame
polynomial_features_df = pd.DataFrame(polynomial_features)

# Concatenate the original dataframe with the new features
df_augmented = pd.concat([df, interaction_terms_df, polynomial_features_df], axis=1)

# Extract input and output data
X = df_augmented[input_columns].values

# Encode the target variable using ordinal encoding
ordinal_encoder = OrdinalEncoder()
y_encoded = ordinal_encoder.fit_transform(df_augmented[output_columns])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Model Selection and Hyperparameter Tuning (Random Forest Regressor)
rf_regressor = RandomForestRegressor(random_state=42)
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10, 15]
}
rf_grid_search = GridSearchCV(estimator=rf_regressor, param_grid=rf_param_grid, cv=5)
rf_grid_search.fit(X_train, y_train)

# Best Random Forest Regressor model
rf_best_model = rf_grid_search.best_estimator_

# Predict on the testing set using Random Forest Regressor
y_pred_rf = rf_best_model.predict(X_test)

# Evaluate Random Forest Regressor model
r2_rf = r2_score(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = sqrt(mse_rf)

print("Random Forest Regressor Metrics:")
print(f'R-squared: {r2_rf}')
print(f'Mean Absolute Error: {mae_rf}')
print(f'Mean Squared Error: {mse_rf}')
print(f'Root Mean Squared Error: {rmse_rf}')

# Save the best Random Forest Regressor model
rf_model_filename = 'random_forest_model.joblib'
joblib.dump(rf_best_model, rf_model_filename)

print(f'Random Forest Regressor model saved as {rf_model_filename}')
