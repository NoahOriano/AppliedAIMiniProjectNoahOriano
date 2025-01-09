# This class will be responsible for storing, retrieving and displaying the results of each model
import os
import numpy as np
from sklearn.model_selection import train_test_split

def save_results(model_name, dataset_name, results):
    np.save(f"results/{model_name}/{dataset_name}", results)

def load_results(model_name, dataset_name):
    return np.load("results/"+model_name+"/"+dataset_name+ ".npy")

def get_all_models():
    return ["rf", "svm", "ann", "dt", "knn", "som", "gmm"]

def split_dataset(data_x, data_y):
    X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def whiten_data(data_x):
    # Ensure that no NAN values are present
    data_x = np.nan_to_num(data_x)
    # Subtract the mean of each feature from the data
    data_x = data_x - np.mean(data_x, axis=0)
    # Divide each feature by its standard deviation
    if(np.std(data_x, axis=0).all() != 0):
        data_x = data_x / np.std(data_x, axis=0)
    return data_x

def evaluate_all_models_on_dataset(data_x, data_y, dataset_name):
    # Whiten the data
    data_x = whiten_data(data_x)
    # Produce the data splits
    X_train, X_test, y_train, y_test = split_dataset(data_x, data_y)

    # Supervised Models

    # Evaluate RF
    if not is_evaluated("rf", dataset_name):
        from models.supervised.RF import RF
        model = RF.train(X_train, y_train)
        cm = RF.predict_and_evaluate(model, X_test, y_test)
        save_results("rf", dataset_name, cm)
    # Evaluate SVM
    if not is_evaluated("svm", dataset_name):
        from models.supervised.SVM import SVM
        model = SVM.train(X_train, y_train)
        cm = SVM.predict_and_evaluate(model, X_test, y_test)
        save_results("svm", dataset_name, cm)
    # Evaluate ANN
    if not is_evaluated("ann", dataset_name):
        from models.supervised.ANN import ANN
        model = ANN.train(X_train, y_train)
        cm = ANN.predict_and_evaluate(model, X_test, y_test)
        save_results("ann", dataset_name, cm)
    # Evaluate DT
    if not is_evaluated("dt", dataset_name):
        from models.supervised.DT import DT
        model = DT.train(X_train, y_train)
        cm = DT.predict_and_evaluate(model, X_test, y_test)
        save_results("dt", dataset_name, cm)

    # Unsupervised Models

    # Evaluate KNN
    if not is_evaluated("knn", dataset_name):
        from models.unsupervised.KNN import KNN
        model = KNN.train(X_train, y_train)
        cm = KNN.predict_and_evaluate(model, X_test, y_test)
        save_results("knn", dataset_name, cm)
    # Evaluate SOM
    if not is_evaluated("som", dataset_name):
        from models.unsupervised.SOM import SOM
        model = SOM.train(X_train, y_train)
        cm = SOM.predict_and_evaluate(model, X_test, y_test)
        save_results("som", dataset_name, cm)
    # # Evaluate GMM
    # if not is_evaluated("gmm", dataset_name):
    #     from models.unsupervised.GMM import GMM
    #     model = GMM.train(X_train, y_train)
    #     cm = GMM.predict_and_evaluate(model, X_test, y_test)
    #     save_results("gmm", dataset_name, cm)


def is_evaluated(model_name, dataset_name):
    if(os.path.exists(f"results/{model_name}/{dataset_name}.npy")):
        return True
    
def map_clusters_to_labels(cluster_labels, true_labels):
    cluster_to_label = {}
    for cluster in np.unique(cluster_labels):
        if cluster == -1:
            continue  # Skip noise points
        # Find the most common true label for each cluster
        cluster_indices = np.where(cluster_labels == cluster)[0]
        true_labels_in_cluster = true_labels[cluster_indices]
        most_common_label = np.bincount(true_labels_in_cluster).argmax()
        cluster_to_label[cluster] = most_common_label
    return cluster_to_label

def calculate_accuracy(cm):
    return np.trace(cm) / np.sum(cm)