from Preprocessing import initialization, fillInMissingValue, fixOutliers, Standardization, oneHotEncoding, \
    deleteMissingValue
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, plot_precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

def dataProcessing(path, col_catergorical_na, col_numerical_na, col_numerical, col_catergorical):
    df = initialization(path)
    '''option: fill the missing values or delete them. Currently: deleted'''
    # df_filled = fillInMissingValue(df,col_catergorical_na,col_numerical_na)
    # df_ss,scaler = Standardization(df_filled,col_numerical)

    df_filled = deleteMissingValue(df, col_catergorical_na, col_numerical_na)
    '''option: fix the outliers or not. Currently: fixed'''
    df_fixed = fixOutliers(df_filled, col_numerical)
    df_ss, scaler = Standardization(df_fixed, col_numerical)
    df_encoding = oneHotEncoding(df_ss)
    X = df_encoding.iloc[:, 0:df_encoding.shape[1] - 1]
    Y = df_encoding.iloc[:, -1]
    '''create training set and test set'''
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
    return X_train, X_test, Y_train, Y_test


def logisticRegression(X_train, X_test, Y_train, Y_test):
    '''classification'''
    # LR = LogisticRegression(penalty='none',class_weight='balanced',max_iter=600)
    LR = LogisticRegression(penalty='l2', class_weight=None, max_iter=800,tol=1e-6)
    # LR = LogisticRegression(penalty='none', class_weight=None, max_iter=800,tol=1e-8)

    LR.fit(X_train, Y_train)
    Y_predict = LR.predict(X_test)
    '''confusion matrix'''
    print(confusion_matrix(Y_test, Y_predict, labels=[1, 0]))
    print(LR.score(X_test, Y_test))
    '''plot_precision_recall_curve'''
    # APC = average_precision_score(Y_test, LR.decision_function(X_test))
    # PRC = plot_precision_recall_curve(LR, X_test, Y_test)
    # PRC.ax_.set_title('Precision-Recall-Curve')
    # plt.savefig('PRC.jpg')
    # plt.show()
    # print(APC)
    '''predict_proba() predict() and decision_function()'''
    # print(LR.predict_proba(X_test))
    # print(Y_predict)
    # print(LR.decision_function(X_test))


if __name__ == "__main__":
    '''Property setting'''
    path = 'Data/crx.data'
    col_numerical = ['A2', 'A3', 'A8', 'A11', 'A14', 'A15']
    col_catergorical = ['A1', 'A4', 'A5', 'A6', 'A7', 'A9', 'A10', 'A12', 'A13']
    col_numerical_na = ['A2', 'A14']
    col_catergorical_na = ['A1', 'A4', 'A5', 'A6', 'A7']
    X_train, X_test, Y_train, Y_test = dataProcessing(path, col_catergorical_na, col_numerical_na, col_numerical,
                                                      col_catergorical)
    logisticRegression(X_train, X_test, Y_train, Y_test)
