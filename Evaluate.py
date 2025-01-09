import Helper
import seaborn as sns
import numpy as np
import tensorflow as tf

def preprocess_tensor_image(image, label):
    image = tf.image.resize(image, [32, 32])  # Resize to 32x32
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
    return image, label

def randomize_data(data_x, data_y):
    data_index_map = range(0, len(data_x))
    data_index_map = list(data_index_map)
    np.random.shuffle(data_index_map)
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    ret_x = []
    ret_y = []
    for i in range(len(data_x)):
        ret_x.append(data_x[data_index_map[i]])
        ret_y.append(data_y[data_index_map[i]])
    return np.array(ret_x), np.array(ret_y)


def evaulate_models():
    # Note: all of the models and datasets are for classification problems
    # Get the names of all the models
    models = Helper.get_all_models()

    def is_any_model_not_evaluated(dataset_name):
        for model in models:
            if not Helper.is_evaluated(model, dataset_name):
                return True
        return False

    # Check if any of the models have not yet been evaluated on the dataset
    if is_any_model_not_evaluated("iris"):
        # Iris dataset
        from sklearn.datasets import load_iris
        iris = load_iris()
        data_x = iris.data
        data_y = iris.target
        dataset_name = "iris"
        Helper.evaluate_all_models_on_dataset(data_x, data_y, dataset_name)
    
    if is_any_model_not_evaluated("wine"):
        # Wine dataset
        from sklearn.datasets import load_wine
        wine = load_wine()
        data_x = wine.data
        data_y = wine.target
        dataset_name = "wine"
        data_x, data_y = randomize_data(data_x, data_y)
        Helper.evaluate_all_models_on_dataset(data_x, data_y, dataset_name)

    if is_any_model_not_evaluated("breast_cancer"):
        # Breast cancer dataset
        from sklearn.datasets import load_breast_cancer
        breast_cancer = load_breast_cancer()
        data_x = breast_cancer.data
        data_y = breast_cancer.target
        dataset_name = "breast_cancer"
        data_x, data_y = randomize_data(data_x, data_y)
        Helper.evaluate_all_models_on_dataset(data_x, data_y, dataset_name)

    if is_any_model_not_evaluated("digits"):
        # Digits dataset
        from sklearn.datasets import load_digits
        digits = load_digits()
        data_x = digits.data
        data_y = digits.target
        dataset_name = "digits"
        data_x, data_y = randomize_data(data_x, data_y)
        Helper.evaluate_all_models_on_dataset(data_x, data_y, dataset_name)

    if is_any_model_not_evaluated("mnist"):
        # Mnist dataset
        from keras.datasets import mnist
        (data_x, data_y), (X_test, y_test) = mnist.load_data()
        dataset_name = "mnist"
        # Limit to 10000 samples
        data_x = data_x[:10000]
        data_y = data_y[:10000]
        # Reshape the data, which is currently an array of images into an array of features
        data_x = data_x.reshape(data_x.shape[0], data_x.shape[1] * data_x.shape[2])
        data_x, data_y = randomize_data(data_x, data_y)
        Helper.evaluate_all_models_on_dataset(data_x, data_y, dataset_name)

    if is_any_model_not_evaluated("cifar10"):
        # Cifar10 dataset
        from keras.datasets import cifar10
        (data_x, data_y), (X_test, y_test) = cifar10.load_data()
        dataset_name = "cifar10"
        # Limit to 10000 samples
        data_x = data_x[:10000]
        data_y = data_y[:10000]
        # Reshape the data, which is currently an array of images into an array of features
        data_x = data_x.reshape(data_x.shape[0], data_x.shape[1] * data_x.shape[2] * data_x.shape[3])
        data_y = np.array(data_y).reshape(-1)
        data_x, data_y = randomize_data(data_x, data_y)
        Helper.evaluate_all_models_on_dataset(data_x, data_y, dataset_name)

    if is_any_model_not_evaluated("cifar100"):
        # Cifar100 dataset
        from keras.datasets import cifar100
        (data_x, data_y), (X_test, y_test) = cifar100.load_data()
        dataset_name = "cifar100"
        # Limit to 10000 samples
        data_x = data_x[:10000]
        data_y = data_y[:10000]
        # Reshape the data, which is currently an array of images into an array of features
        data_x = data_x.reshape(data_x.shape[0], data_x.shape[1] * data_x.shape[2] * data_x.shape[3])
        data_y = np.array(data_y).reshape(-1)
        data_x, data_y = randomize_data(data_x, data_y)
        Helper.evaluate_all_models_on_dataset(data_x, data_y, dataset_name)

    if is_any_model_not_evaluated("fashion_mnist"):
        # Fashion Mnist dataset
        from keras.datasets import fashion_mnist
        (data_x, data_y), (X_test, y_test) = fashion_mnist.load_data()
        dataset_name = "fashion_mnist"
        # Limit to 10000 samples
        data_x = data_x[:10000]
        data_y = data_y[:10000]
        # Reshape the data, which is currently an array of images into an array of features
        data_x = data_x.reshape(data_x.shape[0], data_x.shape[1] * data_x.shape[2])
        data_y = np.array(data_y).reshape(-1)
        data_x, data_y = randomize_data(data_x, data_y)
        Helper.evaluate_all_models_on_dataset(data_x, data_y, dataset_name)

    if is_any_model_not_evaluated("titanic"):
        # Titanic dataset
        dataset = sns.load_dataset('titanic')
        data_x = dataset.drop(columns=['survived', 'embarked', 'who', 'deck', 'embark_town', 'alive', 'alone', 'adult_male'])
        # Convert male/female to 0/1
        data_x['sex'] = data_x['sex'].map({'male': 0, 'female': 1})
        # Replace class with 1-3 for first, second and third class
        data_x['class'] = data_x['class'].map({'First': 1, 'Second': 2, 'Third': 3})
        data_y = dataset['survived']
        # Replace NAN values with the mean of the column
        data_x = np.array(data_x)
        data_y = np.array(data_y).reshape(-1)
        # Drop rows with NAN values from data_x and data_y
        for i in range(data_x.shape[1]):
            data_y = data_y[~np.isnan(data_x[:, i])]
            data_x = data_x[~np.isnan(data_x[:, i])]
        dataset_name = "titanic"
        data_x, data_y = randomize_data(data_x, data_y)
        Helper.evaluate_all_models_on_dataset(data_x, data_y, dataset_name)

    if is_any_model_not_evaluated("california_housing"):
        # California Housing dataset
        from sklearn.datasets import fetch_california_housing
        california = fetch_california_housing()
        data_x = california.data
        data_y = california.target
        # Make the house prices discrete by taking the logarithm of the price and rounding to the nearest integer
        data_y = np.round(np.log(data_y) * 4)
        # Find the lowest value and subtract it from all values to make them positive and start from 0
        data_y = data_y - np.min(data_y)
        data_y = np.array(data_y)
        data_x = np.array(data_x)
        dataset_name = "california_housing"
        data_x, data_y = randomize_data(data_x, data_y)
        Helper.evaluate_all_models_on_dataset(data_x, data_y, dataset_name)

    if is_any_model_not_evaluated("skin_segmentation"):
        # Skin Segmentation dataset
        from ucimlrepo import fetch_ucirepo 
        skin_segmentation = fetch_ucirepo(id=229)
        data = skin_segmentation.data
        # Randomize the order of data
        data = np.column_stack((data.features, data.targets))
        np.random.shuffle(data)
        data_x = data[:, :-1]
        data_y = data[:, -1]
        # Take the first 10000 samples
        data_x = data_x[:10000]
        data_y = data_y[:10000]
        dataset_name = "skin_segmentation"
        data_x, data_y = randomize_data(data_x, data_y)
        Helper.evaluate_all_models_on_dataset(data_x, data_y, dataset_name)


    if is_any_model_not_evaluated("abalone"):
        # Abalone dataset
        from ucimlrepo import fetch_ucirepo
        abalone = fetch_ucirepo(id=1)
        data_x = abalone.data.features
        data_y = abalone.data.targets
        # Convert the sex category to 0-1-2, 0 for infant, 1 for female, 2 for male
        data_x["Sex"] = data_x["Sex"].map({'I': 0, 'F': 1, 'M': 2})
        # Convert to numpy arrays
        data_x = np.array(data_x)
        data_y = np.array(data_y)
        # Limit to 10000 samples and remove first sample
        data_x = data_x[1:10001]
        data_y = data_y[1:10001]
        dataset_name = "abalone"
        data_x, data_y = randomize_data(data_x, data_y)
        Helper.evaluate_all_models_on_dataset(data_x, data_y, dataset_name)

    if is_any_model_not_evaluated("malaria"):
        # Malaria dataset
        import tensorflow_datasets as tfds
        dataset, info = tfds.load("malaria", with_info=True, as_supervised=True)
        train_data = dataset['train'].take(10000)
        # Randomize the order of data
        train_data = train_data.shuffle(10000)
        data_x = []
        data_y = []
        for image, label in tfds.as_numpy(train_data):
            im, lab = preprocess_tensor_image(image, label)
            data_x.append(im)
            data_y.append(lab)
        data_x = np.array(data_x)
        # Reshape data to array of features from array of images
        data_x = data_x.reshape(data_x.shape[0], data_x.shape[1] * data_x.shape[2] * data_x.shape[3])
        dataset_name = "malaria"
        data_x, data_y = randomize_data(data_x, data_y)
        Helper.evaluate_all_models_on_dataset(data_x, data_y, dataset_name)

    if is_any_model_not_evaluated("eurosat"):
        # EuroSAT dataset
        import tensorflow_datasets as tfds
        dataset, info = tfds.load("eurosat", with_info=True, as_supervised=True)
        eurosat_data = dataset['train'].take(10000)
        data_x = []
        data_y = []
        for image, label in tfds.as_numpy(eurosat_data):
            im, lab = preprocess_tensor_image(image, label)
            data_x.append(im)
            data_y.append(lab)
        data_x = np.array(data_x)
        data_y = np.array(data_y).reshape(-1)
        # Reshape data to array of features from array of images
        data_x = data_x.reshape(data_x.shape[0], data_x.shape[1] * data_x.shape[2] * data_x.shape[3])
        dataset_name = "eurosat"
        data_x, data_y = randomize_data(data_x, data_y)
        Helper.evaluate_all_models_on_dataset(data_x, data_y, dataset_name)

    if is_any_model_not_evaluated("rock_paper_scissors"):
        # Rock Paper Scissors dataset
        import tensorflow_datasets as tfds
        dataset, info = tfds.load("rock_paper_scissors", with_info=True, as_supervised=True)
        rock_paper_scissors_data = dataset['train'].take(10000)
        data_x = []
        data_y = []
        for image, label in tfds.as_numpy(rock_paper_scissors_data):
            im, lab = preprocess_tensor_image(image, label)
            data_x.append(im)
            data_y.append(lab)
        data_x = np.array(data_x)
        data_y = np.array(data_y).reshape(-1)
        # Reshape data to array of features from array of images
        data_x = data_x.reshape(data_x.shape[0], data_x.shape[1] * data_x.shape[2] * data_x.shape[3])
        dataset_name = "rock_paper_scissors"
        data_x, data_y = randomize_data(data_x, data_y)
        Helper.evaluate_all_models_on_dataset(data_x, data_y, dataset_name)

    if is_any_model_not_evaluated("penguins"):
        # Penguins dataset
        dataset = sns.load_dataset('penguins')
        data_x = dataset.drop(columns=['species', 'island', 'sex'])
        data_y = dataset['species']
        data_y = data_y.map({'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2})
        data_y = np.array(data_y).reshape(-1)
        # Convert the species to 0-1-2, 0 for Adelie, 1 for Chinstrap, 2 for Gentoo
        dataset_name = "penguins"
        data_x, data_y = randomize_data(data_x, data_y)
        Helper.evaluate_all_models_on_dataset(data_x, data_y, dataset_name)

    if is_any_model_not_evaluated("mnist_corrupted"):
        # MNIST corrupted dataset from tensorflow_datasets
        dataset, info = tfds.load("mnist_corrupted", with_info=True, as_supervised=True)
        mnist_corrupted_data = dataset['train'].take(10000)
        data_x = []
        data_y = []
        for image, label in tfds.as_numpy(mnist_corrupted_data):
            im, lab = preprocess_tensor_image(image, label)
            data_x.append(im)
            data_y.append(lab)
        data_x = np.array(data_x)
        data_y = np.array(data_y).reshape(-1)
        # Reshape data to array of features from array of images
        data_x = data_x.reshape(data_x.shape[0], data_x.shape[1] * data_x.shape[2] * data_x.shape[3])
        data_y = np.array(data_y).reshape(-1)
        dataset_name = "mnist_corrupted"
        data_x, data_y = randomize_data(data_x, data_y)
        Helper.evaluate_all_models_on_dataset(data_x, data_y, dataset_name)

    if is_any_model_not_evaluated("tf_flowers"):
        # TF-flowers
        dataset, info = tfds.load("tf_flowers", with_info=True, as_supervised=True)
        tf_flowers_data = dataset['train'].take(10000)
        data_x = []
        data_y = []
        for image, label in tfds.as_numpy(tf_flowers_data):
            im, lab = preprocess_tensor_image(image, label)
            data_x.append(im)
            data_y.append(lab)
        data_x = np.array(data_x)
        # Reshape data to array of features from array of images
        data_x = data_x.reshape(data_x.shape[0], data_x.shape[1] * data_x.shape[2] * data_x.shape[3])
        data_y = np.array(data_y).reshape(-1)
        dataset_name = "tf_flowers"
        data_x, data_y = randomize_data(data_x, data_y)
        Helper.evaluate_all_models_on_dataset(data_x, data_y, dataset_name)

    if is_any_model_not_evaluated("wine_quality"):
        # Wine Quality Dataset
        from ucimlrepo import fetch_ucirepo
        wine_quality = fetch_ucirepo(id=186) 
        data_x = wine_quality.data.features
        data_y = wine_quality.data.targets
        data_x = np.array(data_x)
        data_y = np.array(data_y).reshape(-1)
        # Convert labels into integers by mapping them to integers
        labels = np.unique(data_y)
        label_map = {}
        for i, label in enumerate(labels):
            label_map[label] = i
        for i in range(len(data_y)):
            data_y[i] = label_map[data_y[i]]
        dataset_name = "wine_quality"
        data_x, data_y = randomize_data(data_x, data_y)
        Helper.evaluate_all_models_on_dataset(data_x, data_y, dataset_name)

    if is_any_model_not_evaluated("dry_beans"):
        # Dry Beans dataset
        from ucimlrepo import fetch_ucirepo
        dry_beans = fetch_ucirepo(id=602)
        xx = dry_beans.data.features
        yy = dry_beans.data.targets

        # Convert labels into integers by mapping them to integers
        label_map = {}
        for i, label in enumerate(np.unique(yy)):
            label_map[label] = i
        for i in range(len(yy)):
            yy[i] = label_map[yy[i]]

        # Randomize the order of data
        xx, yy = randomize_data(data_x, data_y)
        data_x = xx[:10000]
        data_y = yy[:10000]
        dataset_name = "dry_beans"
        Helper.evaluate_all_models_on_dataset(data_x, data_y, dataset_name)

    # Rice dataset
    if is_any_model_not_evaluated("rice"):
        from ucimlrepo import fetch_ucirepo
        rice = fetch_ucirepo(id=545)
        data_x = rice.data.features
        data_y = rice.data.targets
        data_x = np.array(data_x)
        data_y = np.array(data_y).reshape(-1)
        # Convert labels into integers by mapping them to integers
        labels = np.unique(data_y)
        label_map = {}
        for i, label in enumerate(labels):
            label_map[label] = i
        for i in range(len(data_y)):
            data_y[i] = label_map[data_y[i]]
        dataset_name = "rice"
        data_x, data_y = randomize_data(data_x, data_y)
        Helper.evaluate_all_models_on_dataset(data_x, data_y, dataset_name)
    
    # Parkinsons dataset
    if is_any_model_not_evaluated("parkinsons"):
        from ucimlrepo import fetch_ucirepo
        parkinsons = fetch_ucirepo(id=174) 
        data_x = parkinsons.data.features
        data_y = parkinsons.data.targets
        data_x = np.array(data_x)
        data_y = np.array(data_y).reshape(-1)
        # Remove any NAN values
        data_x = data_x[~np.isnan(data_x).any(axis=1)]
        data_y = data_y[~np.isnan(data_x).any(axis=1)]
        # Convert labels into integers by mapping them to integers from binary
        data_y = [1 if y == True else 0 for y in data_y]
        data_x, data_y = randomize_data(data_x, data_y)
        # Randomize
        dataset_name = "parkinsons"
        Helper.evaluate_all_models_on_dataset(data_x, data_y, dataset_name)
