import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load the datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
sample_submission_df = pd.read_csv('sample_submission.csv')

# Features and target variable
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
target = 'SalePrice'

# Prepare the training data
x = train_df[features]
y = train_df[target]

# Split the data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

# Define a transformer for numeric features
numeric_features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])

# Combine transformers into a preprocessor
preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features)])

# Create a pipeline with preprocessor and regressor
model = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', LinearRegression())])

# Fit the model to the training data
model.fit(x_train, y_train)

# Predict and evaluate the model on the validation set
y_val_pred = model.predict(x_val)
mse = mean_squared_error(y_val, y_val_pred)
rmse = mse ** 0.5
print(f'Validation RMSE: {rmse}')

# Predict on the test data
X_test = test_df[features]
test_predictions = model.predict(X_test)

# Create the submission file
submission_df = pd.DataFrame({'Id': test_df['Id'], 'SalePrice': test_predictions})
submission_df.to_csv('submission.csv', index=False)
print('Submission file created: submission.csv')

# Plot Actual vs Predicted SalePrice
plt.figure(figsize=(10, 6))
plt.scatter(y_val, y_val_pred, alpha=0.4)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)
plt.xlabel('Actual SalePrice')
plt.ylabel('Predicted SalePrice')
plt.title('Actual vs Predicted SalePrice')
plt.show()

# Residual plot
residuals = y_val - y_val_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_val_pred, residuals, alpha=0.5)
plt.axhline(0, color='r', linestyle='--', linewidth=2)
plt.xlabel('Predicted SalePrice')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

# Distribution of residuals
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.xlabel('Residuals')
plt.title('Distribution of Residuals')
plt.show()

# Feature importance plot
coefficients = model.named_steps['regressor'].coef_
features_df = pd.DataFrame({'Feature': features, 'Coefficient': coefficients})
plt.figure(figsize=(10, 6))
sns.barplot(x='Feature', y='Coefficient', data=features_df)
plt.title('Feature Importance')
plt.show()
