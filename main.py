# Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

# Load Dataset
df = pd.read_csv('your_dataset.csv')

# Univariate Analysis (for numerical columns)
df['your_column'].hist()
plt.title('Univariate Analysis')
plt.xlabel('Your Column')
plt.ylabel('Frequency')
plt.show()

# Bivariate Analysis
sns.scatterplot(x='your_independent_var', y='your_dependent_var', data=df)
plt.title('Bivariate Analysis')
plt.xlabel('Independent Variable')
plt.ylabel('Dependent Variable')
plt.show()

# Split Dataset into Train and Test
X = df[['your_independent_var']]  # Independent variable
y = df['your_dependent_var']      # Dependent variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Save Model to .pkl file
with open('linear_regression_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Loading the model and passing an input to the model
with open('linear_regression_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Test with a new input
new_input = np.array([[5]])  # Example input
predicted_output = loaded_model.predict(new_input)
print(f'Predicted output for input {new_input[0][0]}: {predicted_output[0]}')
