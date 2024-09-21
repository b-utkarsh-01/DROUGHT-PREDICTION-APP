import joblib
from sklearn.ensemble import RandomForestClassifier

from preprocessing.train_model import X_train
from preprocessing.train_model import y_train

# Assuming X_train and y_train are already defined
rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
rf_model.fit(X_train, y_train)

# Save the model
joblib.dump(rf_model, 'd:/Projects/drought-prediction-app/model/rf_model.pkl')
