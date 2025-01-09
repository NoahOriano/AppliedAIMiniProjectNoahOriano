from sklearn.mixture import GaussianMixture as skGMM
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np

class GMM:
    def train(X_train, y_train):
        y_train = np.squeeze(y_train)
        n_classes = len(np.unique(y_train))
        
        # Initialize the Gaussian Mixture Model with spherical covariance
        classifier = skGMM(
            n_components=n_classes,
            covariance_type='spherical',
            max_iter=20,
            init_params='random'
        )
        
        # Train the GMM on the full training set
        classifier.fit(X_train)

        # Map the GMM components to the classes
        y_pred = classifier.predict(X_train)
        # Convert y_pred to integer
        y_pred = np.array(y_pred)
        y_train = np.array(y_train)
        y_pred = y_pred.astype(int)
        # Convert y_train to integer
        y_train = y_train.astype(int)
        class_mapping = {}
        for i in range(n_classes):
            class_mapping[i] = np.argmax(np.bincount(y_train[y_pred == i]))
        classifier.class_mapping = class_mapping
        
        return (classifier, class_mapping)
    def predict_and_evaluate(model, X_test, y_test):
        (model, class_mapping) = model
        y_pred = model.predict(X_test)
        y_pred = np.array([class_mapping[i] for i in y_pred])
        cm = confusion_matrix(y_test, y_pred)
        return cm
    