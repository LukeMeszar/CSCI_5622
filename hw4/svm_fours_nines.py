import argparse
import numpy as np
import random

from svm import weight_vector, find_support, find_slack
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import classification_report

class FoursAndNines:
    """
    Class to store MNIST data
    """

    def __init__(self, location):
        # You shouldn't have to modify this class, but you can if
        # you'd like.

        import pickle, gzip

        # Load the dataset
        f = gzip.open(location, 'rb')

        train_set, valid_set, test_set = pickle.load(f)

        self.x_train = train_set[0][np.where(np.logical_or( train_set[1]==4, train_set[1] == 9))[0],:]
        self.y_train = train_set[1][np.where(np.logical_or( train_set[1]==4, train_set[1] == 9))[0]]

        shuff = np.arange(self.x_train.shape[0])
        np.random.shuffle(shuff)
        self.x_train = self.x_train[shuff,:]
        self.y_train = self.y_train[shuff]

        self.x_valid = valid_set[0][np.where(np.logical_or( valid_set[1]==4, valid_set[1] == 9))[0],:]
        self.y_valid = valid_set[1][np.where(np.logical_or( valid_set[1]==4, valid_set[1] == 9))[0]]

        self.x_test  = test_set[0][np.where(np.logical_or( test_set[1]==4, test_set[1] == 9))[0],:]
        self.y_test  = test_set[1][np.where(np.logical_or( test_set[1]==4, test_set[1] == 9))[0]]

        f.close()

def mnist_digit_show(flatimage, label, outname=None):

    import matplotlib.pyplot as plt

    image = np.reshape(flatimage, (-1,28))

    plt.matshow(image, cmap=plt.cm.binary)
    plt.xticks([])
    plt.yticks([])
    plt.title("Correct Label %s" % label)
    if outname:
        plt.savefig(outname)
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SVM classifier options')
    parser.add_argument('--limit', type=int, default=-1,
                        help="Restrict training to this many examples")
    args = parser.parse_args()

    data = FoursAndNines("../data/mnist.pkl.gz")
    #linear grid search
    # grid_params_linear = [{'C': [0.05,0.1,0.15], 'kernel': ['linear']}]
    # svm = SVC()
    # clf = GridSearchCV(estimator=svm,param_grid=grid_params_linear,cv=5,n_jobs=4,verbose=3)
    # clf.fit(data.x_train,data.y_train)
    # print("best linear params (C): ", clf.best_params_['C'])
    # y_true,y_pred = data.y_test, clf.predict(data.x_test)
    # #print("support vectors:",clf.support_vectors_)
    # print("classification_report:\n", classification_report(y_true,y_pred))
    # best_linear_svm = SVC(kernel='linear', C = clf.best_params_['C'])
    # best_linear_svm.fit(data.x_train,data.y_train)
    # print(best_linear_svm.score(data.x_test,data.y_test))

    #polynomial grid search
    # grid_params_poly = [{'C': [1,100,200,300,400,500, 600,700,800,900,1000,2000,3000,4000,5000,6000,7000,8000,9000], 'kernel': ['poly'],'degree': [1,2,3]}]
    # svm = SVC()
    # clf = GridSearchCV(estimator=svm,param_grid=grid_params_poly,cv=3,n_jobs=8,verbose=3)
    # clf.fit(data.x_train,data.y_train)
    # print("best poly params (C): ", clf.best_params_['C'])
    # print("best poly params (degree): ", clf.best_params_['degree'])
    # y_true,y_pred = data.y_test, clf.predict(data.x_test)
    # #print("support vectors:",clf.support_vectors_)
    # print("classification_report:\n", classification_report(y_true,y_pred))
    # best_poly_svm = SVC(kernel='poly', C = clf.best_params_['C'],degree=clf.best_params_['degree'])
    # best_poly_svm.fit(data.x_train,data.y_train)
    # print(best_poly_svm.score(data.x_test,data.y_test))

    #rbf grid search
    # grid_params_rbf = [{'C': [1,10,20,30,40,50,75,100], 'kernel': ['rbf'],'gamma': [0.001,0.01,0.1]}]
    # svm = SVC()
    # clf = GridSearchCV(estimator=svm,param_grid=grid_params_rbf,cv=3,n_jobs=8,verbose=3)
    # clf.fit(data.x_train,data.y_train)
    # print("best rbf params (C): ", clf.best_params_['C'])
    # print("best rbf params (gamma): ", clf.best_params_['gamma'])
    # y_true,y_pred = data.y_test, clf.predict(data.x_test)
    # #print("support vectors:",clf.support_vectors_)
    # print("classification_report:\n", classification_report(y_true,y_pred))
    # best_rbf_svm = SVC(kernel='rbf', C = clf.best_params_['C'],gamma=clf.best_params_['gamma'])
    # best_rbf_svm.fit(data.x_train,data.y_train)
    # print(best_rbf_svm.score(data.x_test,data.y_test))

    # grid_params_all = [{'C': [0.05,0.1,0.15], 'kernel': ['linear']},{'C': [1000,2000,3000,4000,5000], 'kernel': ['poly'],'degree': [2,3,4]},{'C': [10,20,30,40,50], 'kernel': ['rbf'],'gamma': [0.001,0.01,0.1]}]
    # svm = SVC()
    # clf = GridSearchCV(estimator=svm,param_grid=grid_params_all,cv=3,n_jobs=8,verbose=3)
    # clf.fit(data.x_train,data.y_train)
    # print("best params : ", clf.best_params_)
    # y_true,y_pred = data.y_test, clf.predict(data.x_test)
    # print("classification_report:\n", classification_report(y_true,y_pred))
    # best_rbf_svm = SVC(kernel='rbf', C = clf.best_params_['C'],gamma=clf.best_params_['gamma'])
    # best_rbf_svm.fit(data.x_train,data.y_train)
    # print(best_rbf_svm.score(data.x_test,data.y_test))
    best_svm = SVC(kernel='rbf',C=40,gamma=0.01)
    best_svm.fit(data.x_train,data.y_train)
    print(best_svm.score(data.x_test,data.y_test))
    random.shuffle(best_svm.support_)
    for i in range(0,4):
        filename = 'support_vector' + str(i)
        mnist_digit_show(data.x_train[best_svm.support_[i]],str(data.y_train[best_svm.support_[i]]),filename)



    # -----------------------------------
    # Plotting Examples
    # -----------------------------------


    # Display in on screen
    #mnist_digit_show(data.x_train[ 0,:])

    # Plot image to file
    #mnist_digit_show(data.x_train[1,:], "mnistfig.png")
