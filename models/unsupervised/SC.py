from sklearn.cluster import SpectralClustering
from sklearn.metrics import confusion_matrix
import numpy as np
from Helper import map_clusters_to_labels

class SC:
    def train(X_train, y_train):
        max_iter=2
        n_clusters=len(np.unique(y_train))
        model = SpectralClustering(
            n_clusters=n_clusters,
            assign_labels='kmeans',  # Use k-means for label assignment
            random_state=42,
            n_init=max_iter  # Limit the number of iterations so it doesn't take too long 
        )
        cluster_labels = model.fit_predict(X_train)
        # Convert cluster labels to integer labels from float
        cluster_labels = np.array([int(label) for label in cluster_labels])
        y_train = np.array(y_train)
        y_train = np.array([int(label) for label in y_train])
        cluster_to_label_map = map_clusters_to_labels(cluster_labels, y_train)
        model = (model, cluster_to_label_map)
        return model
    def predict_and_evaluate(model, X_test, y_test):
        model, cluster_to_label_map = model
        y_pred = model.fit_predict(X_test)
        mapped_labels = np.array([
            cluster_to_label_map[label] if label != -1 else -1 for label in y_pred
        ])
        cm = confusion_matrix(y_test, mapped_labels)
        return cm