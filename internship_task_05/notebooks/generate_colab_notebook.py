
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

def generate_colab_notebook():
    nb = new_notebook()

    nb.cells.append(new_markdown_cell("# Regression Modeling: From Scratch to Production"))
    nb.cells.append(new_markdown_cell("## Internship Task 05: Medical Insurance Cost Prediction"))
    nb.cells.append(new_markdown_cell("This notebook demonstrates the implementation of linear and regularized regression models from scratch, benchmarks them against scikit-learn implementations, and provides comprehensive diagnostics and residual analysis. The goal is to predict medical insurance charges based on various personal attributes."))

    nb.cells.append(new_markdown_cell("### 1. Setup and Imports"))
    nb.cells.append(new_code_cell("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import statsmodels.api as sm

# Custom regression implementations (assuming they are in a file named linear_regression_scratch.py)
# We'll include the code directly for Colab self-containment
"""))

    nb.cells.append(new_markdown_cell("### 2. From-Scratch Regression Implementations"))
    nb.cells.append(new_code_cell("""class LinearRegressionFromScratch:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

class RidgeRegressionFromScratch(LinearRegressionFromScratch):
    def __init__(self, learning_rate=0.01, n_iterations=1000, lambda_param=0.1):
        super().__init__(learning_rate, n_iterations)
        self.lambda_param = lambda_param

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y)) + (2 * self.lambda_param / n_samples) * self.weights
            db = (1/n_samples) * np.sum(y_predicted - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

class LassoRegressionFromScratch(LinearRegressionFromScratch):
    def __init__(self, learning_rate=0.01, n_iterations=1000, lambda_param=0.1):
        super().__init__(learning_rate, n_iterations)
        self.lambda_param = lambda_param

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias
            # Subgradient for Lasso
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y)) + self.lambda_param * np.sign(self.weights)
            db = (1/n_samples) * np.sum(y_predicted - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
"""))

    nb.cells.append(new_markdown_cell("### 3. Data Loading and Preprocessing"))
    nb.cells.append(new_code_cell("""# Load the dataset
df = pd.read_csv('insurance.csv')

# Display basic info
print(df.info())
print(df.head())

# Define categorical and numerical features
categorical_features = ['sex', 'smoker', 'region']
numerical_features = ['age', 'bmi', 'children']

# Create a column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# Split data into features (X) and target (y)
X = df.drop('charges', axis=1)
y = df['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply preprocessing to training and testing data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
"""))

    nb.cells.append(new_markdown_cell("### 4. Model Training and Benchmarking"))
    nb.cells.append(new_markdown_cell("#### 4.1. From-Scratch Models"))
    nb.cells.append(new_code_cell("""# Linear Regression From Scratch
lr_scratch = LinearRegressionFromScratch(learning_rate=0.01, n_iterations=5000)
lr_scratch.fit(X_train_processed, y_train.values)
y_pred_lr_scratch = lr_scratch.predict(X_test_processed)

# Ridge Regression From Scratch
ridge_scratch = RidgeRegressionFromScratch(learning_rate=0.01, n_iterations=5000, lambda_param=100)
ridge_scratch.fit(X_train_processed, y_train.values)
y_pred_ridge_scratch = ridge_scratch.predict(X_test_processed)

# Lasso Regression From Scratch
lasso_scratch = LassoRegressionFromScratch(learning_rate=0.01, n_iterations=5000, lambda_param=100)
lasso_scratch.fit(X_train_processed, y_train.values)
y_pred_lasso_scratch = lasso_scratch.predict(X_test_processed)
"""))

    nb.cells.append(new_markdown_cell("#### 4.2. Scikit-learn Models"))
    nb.cells.append(new_code_cell("""# Scikit-learn Linear Regression
lr_sklearn = LinearRegression()
lr_sklearn.fit(X_train_processed, y_train)
y_pred_lr_sklearn = lr_sklearn.predict(X_test_processed)

# Scikit-learn Ridge Regression
ridge_sklearn = Ridge(alpha=100)
ridge_sklearn.fit(X_train_processed, y_train)
y_pred_ridge_sklearn = ridge_sklearn.predict(X_test_processed)

# Scikit-learn Lasso Regression
lasso_sklearn = Lasso(alpha=100, max_iter=5000)
lasso_sklearn.fit(X_train_processed, y_train)
y_pred_lasso_sklearn = lasso_sklearn.predict(X_test_processed)
"""))

    nb.cells.append(new_markdown_cell("### 5. Model Evaluation"))
    nb.cells.append(new_code_cell("""def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n--- {model_name} ---")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R-squared (R2): {r2:.2f}")
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}

metrics = {}
metrics['Linear Regression (Scratch)'] = evaluate_model(y_test, y_pred_lr_scratch, 'Linear Regression (Scratch)')
metrics['Ridge Regression (Scratch)'] = evaluate_model(y_test, y_pred_ridge_scratch, 'Ridge Regression (Scratch)')
metrics['Lasso Regression (Scratch)'] = evaluate_model(y_test, y_pred_lasso_scratch, 'Lasso Regression (Scratch)')
metrics['Linear Regression (Scikit-learn)'] = evaluate_model(y_test, y_pred_lr_sklearn, 'Linear Regression (Scikit-learn)')
metrics['Ridge Regression (Scikit-learn)'] = evaluate_model(y_test, y_pred_ridge_sklearn, 'Ridge Regression (Scikit-learn)')
metrics['Lasso Regression (Scikit-learn)'] = evaluate_model(y_test, y_pred_lasso_sklearn, 'Lasso Regression (Scikit-learn)')

# Create a DataFrame for easy comparison
metrics_df = pd.DataFrame(metrics).T
print("\n--- Model Comparison ---")
print(metrics_df)
"""))

    nb.cells.append(new_markdown_cell("### 6. Diagnostics and Residual Analysis"))
    nb.cells.append(new_markdown_cell("#### 6.1. Residual Plots"))
    nb.cells.append(new_code_cell("""# For Scikit-learn Linear Regression as a representative model
residuals = y_test - y_pred_lr_sklearn

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred_lr_sklearn, y=residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Charges')
plt.ylabel('Residuals')
plt.title('Residuals vs. Predicted Values (Scikit-learn Linear Regression)')
plt.show()

sns.histplot(residuals, kde=True)
plt.title('Distribution of Residuals (Scikit-learn Linear Regression)')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()
"""))

    nb.cells.append(new_markdown_cell("#### 6.2. Q-Q Plot of Residuals"))
    nb.cells.append(new_code_cell("""fig, ax = plt.subplots(figsize=(8, 6))
sm.qqplot(residuals, line='s', ax=ax)
ax.set_title('Q-Q Plot of Residuals (Scikit-learn Linear Regression)')
plt.show()
"""))

    nb.cells.append(new_markdown_cell("#### 6.3. Feature Importance (from Scikit-learn Linear Regression)"))
    nb.cells.append(new_code_cell("""# Get feature names after one-hot encoding and scaling
feature_names = numerical_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))

# Create a DataFrame for coefficients
coefficients_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': lr_sklearn.coef_
})

coefficients_df['Absolute_Coefficient'] = np.abs(coefficients_df['Coefficient'])
coefficients_df = coefficients_df.sort_values(by='Absolute_Coefficient', ascending=False)

plt.figure(figsize=(12, 7))
sns.barplot(x='Coefficient', y='Feature', data=coefficients_df)
plt.title('Feature Importance (Coefficients) - Scikit-learn Linear Regression')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.show()
"""))

    nb.cells.append(new_markdown_cell("### 7. Conclusion"))
    nb.cells.append(new_markdown_cell("This notebook successfully implemented linear and regularized regression models from scratch and benchmarked them against their scikit-learn counterparts. The diagnostics and residual analysis provide insights into model performance and assumptions. The from-scratch implementations, while simpler, provide a foundational understanding of how these algorithms work."))

    with open('internship_task_05/notebooks/regression_from_scratch_to_production.ipynb', 'w') as f:
        nbformat.write(nb, f)

if __name__ == '__main__':
    generate_colab_notebook()
