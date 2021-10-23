import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
import seaborn as sns

pd.options.mode.chained_assignment = None

def initialization(path):
    df = pd.read_csv(path)
    df.columns = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16']
    ''' replace ? with NaN '''
    df = df.replace('?', np.nan)
    # transform A2 A14 to numerical
    df[['A2', 'A14']] = df[['A2', 'A14']].astype('float')
    # print(df.isnull().any())
    return df


def deleteMissingValue(df, col_catergorical_na, col_numerical_na):
    df_deleted = df.dropna()
    return df_deleted


def fillInMissingValue(df, col_catergorical_na, col_numerical_na):
    '''Clear the records that are missing more than 3'''
    df_deleted = df.dropna(thresh=14)
    # print(data_df_deleted['A1'].value_counts())
    '''Fill in the missing values, use the mode for classification, and use the median for values'''
    values = {}
    for column in col_catergorical_na:
        values[column] = df_deleted[column].mode()[0]
    df_deleted = df_deleted.fillna(values)
    values = {}
    for column in col_numerical_na:
        values[column] = df_deleted[column].median()
    df_filled = df_deleted.fillna(values)
    return df_filled


#   print(df_deleted.isnull().any())


def fixOutliers(df_filled, col_numerical):
    '''box plot'''
    # print(data_df_deleted.describe())
    # data_df_A2 =pd.DataFrame(data_df_deleted,columns=['A2'])
    # data_df_A14 =pd.DataFrame(data_df_deleted,columns=['A14'])
    # data_df_A2.plot(y=data_df_A2.columns,kind = 'box')
    # data_df_A14.plot(y=data_df_A14.columns,kind = 'box')
    # plt.show()
    # df_deleted_numeric = df_deleted[['A2','A3','A8','A11','A14','A15']]

    '''Normality test'''
    # for column in ['A2','A3','A8','A11','A14','A15']:
    #     std = df_deleted[column].std()
    #     mean = df_deleted[column].mean()
    #     p = stats.kstest(df_deleted[column],'norm',(mean,std))
    #     if p.pvalue > 0.05:
    #         print('yes')
    #     else:
    #         print('no')
    '''fix'''
    for column in col_numerical:
        Q1 = df_filled[column].describe()['25%']
        Q3 = df_filled[column].describe()['75%']
        median = df_filled[column].describe()['50%']
        IQR = Q3 - Q1
        minimun = Q1 - 1.5 * IQR
        maximun = Q3 + 1.5 * IQR
        df_filled.loc[(df_filled[column] < minimun) | (df_filled[column] > maximun), column] = median
    # print(df_deleted.head())
    return df_filled

'''standardization '''
def Standardization(df_fixed, col_numerical):
    df_ss = df_fixed
    scaler = StandardScaler()
    scaler.fit(df_ss[col_numerical])
    df_ss.loc[:, col_numerical] = scaler.transform(df_ss[col_numerical])
    # print(df_ss.mean(axis=0))
    # print(df_ss.std(axis=0))

    return df_ss, scaler
    # df_deleted = scaler.inverse_transform(df_ss)


def oneHotEncoding(df_ss):
    '''one hot encoding'''
    df_ss = df_ss.replace('+', 1)
    df_ss = df_ss.replace('-', 0)
    df_ss = df_ss.replace('t', 1)
    df_ss = df_ss.replace('f', 0)
    '''Pearson corr and its heat map'''
    # sns.heatmap(data=df_ss.corr(), cmap="RdBu_r")
    # plt.savefig('Heatmap.jpg')
    # plt.show()
    df_encoding = pd.get_dummies(df_ss)
    target = df_encoding['A16']
    df_encoding.drop(['A16'], axis=1, inplace=True)
    df_encoding['A16'] = target
    df_encoding = df_encoding.astype('float')
    # print(df_encoding.corr())

    return df_encoding
