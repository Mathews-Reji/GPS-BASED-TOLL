import joblib


def predict(data):
    clf = joblib.load("06_07_lgbm_model.sav")
    return clf.predict(data)
