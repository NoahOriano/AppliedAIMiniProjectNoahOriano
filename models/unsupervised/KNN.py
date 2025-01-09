from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

class KNN:
    def train(X_train, y_train):
        # Count the number of classes
        num_classes = len(set(y_train))
        # Count the number of samples
        num_samples = len(y_train)
        # Calculate the number of neighbors to be a max of 20, and a min of 3, and a sqrt of the number of samples per class
        n_neighbors = max(min(20, int((num_samples/num_classes) ** 0.5)), 3)
        model = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
        model.fit(X_train, y_train)
        return model

    def predict_and_evaluate(model, X_test, y_test):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        return cm