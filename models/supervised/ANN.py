import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import to_categorical

class ANN:
    def train(X_train, y_train):
        # Define the model
        model = Sequential()
        y_train_onehot = to_categorical(y_train)
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        classes = y_train_onehot.shape[1]
        model.add(Dense(units=32, activation='relu', input_dim=X_train.shape[1]))
        model.add(Dense(units=16, activation='relu'))
        model.add(Dense(units=classes, activation='softmax'))
        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        # Fit the model
        model.fit(X_train, y_train_onehot, epochs=100, verbose=0)
        return model
    def predict_and_evaluate(model, X_test, y_test):
        y_pred = np.argmax(model.predict(X_test), axis=1)
        cm = confusion_matrix(y_test, y_pred)
        return cm
