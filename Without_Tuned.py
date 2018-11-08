import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, \
    confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
import warnings
import sklearn.metrics as metrics
import time

warnings.filterwarnings('ignore')

DATASET_PATH = r"C:\Users\Terry\Documents\Software_Defect_Tuning\testDataset\xerces-1.3.csv"


def read_data(path):
    data = pd.read_csv(path, skipinitialspace=True)
    data = data.iloc[:, 3:]
    return data


def data_statistics(dataset):
    # print the first 5 rows of data
    print(dataset.head(5))
    print('The shape of our dataset is:', dataset.shape)
    # Descriptive statistics for each column
    print(dataset.describe())


def split_dataset(dataset, train_percentage):
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1].map(lambda x: 'False' if x == 0 else 'True')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_percentage, random_state=42)
    return X_train, X_test, y_train, y_test


def random_forest(features, target):
    # rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
    #                             max_depth=9, max_features=0.8, max_leaf_nodes=23,
    #                             min_impurity_decrease=0.0, min_impurity_split=None,
    #                             min_samples_leaf=3, min_samples_split=4,
    #                             min_weight_fraction_leaf=0.0, n_estimators=108, n_jobs=None,
    #                             oob_score=False, random_state=0, verbose=0, warm_start=False)
    rf = RandomForestClassifier()
    rf.fit(features, target)
    return rf


def multilayer_perceptron(features, target):
    # mlpn = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
    #                      beta_1=0.9, beta_2=0.999, early_stopping=False,
    #                      epsilon=1e-08, hidden_layer_sizes=(5, 2),
    #                      learning_rate='constant', learning_rate_init=0.001,
    #                      max_iter=200, momentum=0.9, n_iter_no_change=10,
    #                      nesterovs_momentum=True, power_t=0.5, random_state=1,
    #                      shuffle=True, solver='lbfgs', tol=0.0001,
    #                      validation_fraction=0.1, verbose=False, warm_start=False)
    mlpn = MLPClassifier()
    mlpn.fit(features, target)
    return mlpn


def KNN(features, target):
    # knn = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(features, target)
    return knn


def result_statistics(predictions):
    print("Precision  :: ", precision_score(test_y, predictions, average='macro'))
    print("recall score ::", recall_score(test_y, predictions, average='macro'))
    print("f1 score :: ", f1_score(test_y, predictions, average='macro'))

    #
    # precision, recall, fscore, support = score(test_y, predictions)
    # print('precision: {}'.format(precision))
    # print('recall: {}'.format(recall))
    # print('fscore: {}'.format(fscore))
    # print('support: {}'.format(support))


def main():
    start_time = time.time()

    dataset = read_data(DATASET_PATH)

    # Get basic statistics of the loaded dataset
    # data_statistics(dataset)

    global train_x, test_x, train_y, test_y

    train_x, test_x, train_y, test_y = split_dataset(dataset, 0.25)
    # Train and Test dataset size details
    print("Train_x Shape :: ", train_x.shape)
    print("Train_y Shape :: ", train_y.shape)
    print("Test_x Shape :: ", test_x.shape)
    print("Test_y Shape :: ", test_y.shape)

    print("")
    print("----------Random Forest----------")
    rf = random_forest(train_x, train_y)
    print("Trained model:", rf)

    rf_predictions = rf.predict(test_x)
    # print("Train Accuracy :: ", accuracy_score(train_y, rf.predict(train_x)))
    result_statistics(rf_predictions)

    print("")
    print("----------Multilayer Perceptron----------")
    mlpn = multilayer_perceptron(train_x, train_y)
    print("Trained model:", mlpn)

    mlpn_predictions = mlpn.predict(test_x)
    # print("Train Accuracy :: ", accuracy_score(train_y, mlpn.predict(train_x)))
    result_statistics(mlpn_predictions)

    print("")
    print("----------KNN----------")
    knn = KNN(train_x, train_y)
    print("Trained model:", knn)

    knn_predictions = knn.predict(test_x)
    # print("Train Accuracy :: ", accuracy_score(train_y, knn.predict(train_x)))
    result_statistics(knn_predictions)

    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()
