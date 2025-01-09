from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

class RF:
    def train(X_train, y_train):
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1) # Standard estimator value
        model.fit(X_train, y_train)
        return model

    def predict_and_evaluate(model, X_test, y_test):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        return cm