from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

class DT:
    def train(X_train, y_train):
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)
        return model

    def predict_and_evaluate(model, X_test, y_test):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        return cm