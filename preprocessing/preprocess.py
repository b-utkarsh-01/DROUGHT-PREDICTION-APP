import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Normalize continuous features (e.g., elevation, slopes)
    scaler = MinMaxScaler()
    continuous_features = ['elevation', 'slope1', 'slope2', 'slope3', 'slope4', 'slope5', 'slope6', 'slope7', 'slope8']
    data[continuous_features] = scaler.fit_transform(data[continuous_features])
    
    # One-hot encode categorical features
    data_encoded = pd.get_dummies(data, columns=['SQ1', 'SQ2', 'SQ3', 'SQ4', 'SQ5', 'SQ6', 'SQ7'], drop_first=True)
    
    return data_encoded
