
## Machine Learning Pipeline

# 1. Load dataset
# 2. Feature selection
# 3. Train-test split
# 4. Train Linear Regression model
# 5. Predict exam score

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load dataset
df = pd.read_csv("student_data.csv")

# Features and target
X = df[["study_hours", "attendance", "assignments"]]
y = df["score"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, predictions)

print("Model trained successfully")
print("Mean Absolute Error:", mae)