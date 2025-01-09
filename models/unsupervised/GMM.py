from sklearn.mixture import GaussianMixture as skGMM
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np

class GMM:
    def train(X_train, y_train):
        n_classes = len(np.unique(y_train))
        # Split the data into training and validation sets
        X_train_split, X_test, y_train_split, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        # Try Gaussian Mixture Models using different types of covariances.
        classifiers = {
            covar_type: skGMM(
                n_components=n_classes,
                covariance_type=covar_type,
                max_iter=4,
                init_params='random'
            )
            for covar_type in ['spherical', 'diag', 'tied', 'full']
        }

        # Train classifiers and compute confusion matrices
        results = {}
        for name, classifier in classifiers.items():
            # Initialize GMM parameters in a supervised manner using training data
            classifier.means_init = np.array([
                X_train_split[y_train_split == i].mean(axis=0) for i in range(n_classes)
            ])
            
            # Train the GMM
            classifier.fit(X_train_split)
            
            # Predict on the test set
            y_test_pred = classifier.predict(X_test)
            
            # Compute the confusion matrix
            cm = confusion_matrix(y_test, y_test_pred)
            results[name] = cm
        
        # Find the best classifier
        best_classifier = max(results, key=lambda key: np.sum(results[key]))
        # Retrain the model on the full training set with the best covariance type
        best_model = skGMM(
                n_components=n_classes,
                covariance_type=best_classifier,
                max_iter=15,
                init_params='random'
        )
        best_model.fit(X_train, y_train)
        return best_model
    def predict_and_evaluate(model, X_test, y_test):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        return cm