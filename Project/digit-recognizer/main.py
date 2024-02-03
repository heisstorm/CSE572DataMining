import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
# pip install tensorflow==2.15.0 --ignore-installed
# pip install np_utils

# %matplotlib inline
np.random.seed(2)

if __name__ == '__main__':
    train = pd.read_csv("./train.csv")
    test = pd.read_csv("./test.csv")

    Y_train = train["label"]
    X_train = train.drop(labels=["label"], axis=1)

    Y_train.value_counts()
    X_train.isnull().any().describe()
    test.isnull().any().describe()
    X_train = X_train / 255.0
    test = test / 255.0
    X_train = X_train.values.reshape(-1, 28, 28, 1)
    test = test.values.reshape(-1, 28, 28, 1)
    Y_train = to_categorical(Y_train, num_classes=10)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=5)
    print(X_train)