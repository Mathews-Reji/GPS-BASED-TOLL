import joblib

# Load the model once when the module is imported
try:
    model = joblib.load("06_07_lgbm_model.sav")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

def predict(data):
    if model is None:
        raise Exception("Model not loaded")
    return model.predict(data)
