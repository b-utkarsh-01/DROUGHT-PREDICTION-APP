import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from preprocessing.preprocess import preprocess_data

# Load and preprocess the dataset
file_path = 'data/soil_data.csv'  # Update with the actual dataset path
data = preprocess_data(file_path)

# Define features (X) and target (y)
X = data.drop(columns=['drought', 'fips', 'lat', 'lon', 'year', 'month'])
y = data['drought']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
rf_model.fit(X_train, y_train)

# Save the trained model to the model folder
joblib.dump(rf_model, 'model/save_model.py')
