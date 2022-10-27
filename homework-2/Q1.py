import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


def load_magic_data():


    Ynames = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym',
             'fM3Long', 'fM3Trans', 'fAlpha', 'fDist', 'identity']

    featureNames = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym',
             'fM3Long', 'fM3Trans', 'fAlpha', 'fDist']

    filepath = 'magic04.data'

    data = pd.read_csv(filepath, names=Ynames, header=None)

    data['identity'][data['identity'] == 'g'] = 1
    data['identity'][data['identity'] == 'h'] = 0

    X = data[featureNames].values
    Y = data['identity'].values.astype('int64')

    return X, Y


def main():
    X, Y = load_magic_data()
    X = StandardScaler().fit_transform(X)
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.20, random_state=None)

    accuracy = list()
    upperRange = 50
    for i in range(1,upperRange):
        # defaults as a l2 SVM
        clf = linear_model.SGDClassifier(max_iter=i, tol=1e-3)
        clf.fit(Xtrain, Ytrain)
        accuracy.append(clf.score(Xtest,Ytest))
        print("accuracy = " + str(clf.score(Xtest,Ytest)))

    print(accuracy)
    plt.plot(range(1,upperRange), accuracy)

    plt.xlabel('iterations')
    plt.ylabel('accuracy')
    plt.title('Accuracy over iterations')
    plt.show()

    x = list()
    accuracy = list()

    for i in range(1, 99, 2):
        i2 = i/100
        x.append(i2)
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=i2, random_state=None)
        # defaults as a l2 SVM
        clf = linear_model.SGDClassifier(max_iter=10, tol=1e-3)
        clf.fit(Xtrain, Ytrain)
        accuracy.append(clf.score(Xtest,Ytest))
        print("accuracy = " + str(clf.score(Xtest,Ytest)))


    plt.plot(x, accuracy)

    plt.xlabel('test size')
    plt.ylabel('accuracy')
    plt.title('Accuracy over test splits')
    plt.show()

if __name__=="__main__":
    main()
