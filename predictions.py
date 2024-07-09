import joblib


def predict(data):
    model = joblib.load("06_07_lgbm_model.sav")
    return model.predict(data)
