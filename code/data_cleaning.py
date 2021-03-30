import pandas as pd
import copy
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
import numpy as np
from numpy import nan
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split


def data_clean(df, isTrain):
    """
    :param df: dataframe, Input dataframe
    :param isTrain: bool, see whether the input df is training dataset
    :return: feature vectors and labels
    """
    """
    Step 0: Check the summary of the input data frame
    Step 1: After observation, I found that there are 6 columns whose data type are 'object'. To get these columns, a subset dataframe is get from the raw_data, namely df_obj.
    Step 2: To have a closer look at df_obj, I first print out the unique values for all features (['x34', 'x35', 'x41', 'x45', 'x68', 'x93']) in df_obj.
    Step 3: Now converting objects to float type    
        (1) ['x41', 'x45'] can be easily converted to float type;
        (2) ['x35'] contains values that share the same meaning；
        (3) ['x34', 'x68', 'x93'] just do it
        (4) convert category type to float without touching missing values
        (5) double check the converted data frame
    Step 4: Dealing with missing values. There are many ways to handle missing values, such as drop rows containing n/a, fill n/a with aggreate values (such as mean, median, etc.), and impute
            n/a with imputers (such as IterativeImputer, KNNImputer, SimpleImputer, etc.)
            Here I chose KNNImputer to handle missing values. Note that the n_neighbors is considered as a hyperparameter. I used 5, but other values can be applied as well.
    Step 5: Split the given training data into training and validation dataset. Validation dataset will be used for hyperparameter tuning. Then get X and y and return.
    Step 6: Dealing with data imbalance for training samples. There are several ways to handle data imbalance and the intuitive method is data re-sampling. Since under-sampling may cause overfitting and poor generalization,
            I chose to use over-sampling technique, which is described as adding more copies to the minority class.
    Step 7: Normalize the data features with zero mean and unit variance for all training, validation and test dataset.
    """
    # Step 0: first check the summary of df
    print(df.info())
    print("================================")

    # Step1. subset a dataframe that contains 'object' values
    df_obj = df.select_dtypes(include='object')

    # Step2. take a closer look at the subset, df_obj, for more information.
    for item in list(df_obj):
        print(item, df_obj[item].unique())
    print("================================")

    # Step3(1): simply remove unnecessary symbols and convert to float type
    df['x41'] = df['x41'].str.replace('$', '').astype(float)
    df['x45'] = df['x45'].str.replace('%', '').astype(float)

    # Step3(2): combine values which have the same meaning and convert to category type
    df['x35'] = df['x35'].replace('thur', 'thurday').astype('category')
    df['x35'] = df['x35'].replace('wed', 'wednesday').astype('category')
    df['x35'] = df['x35'].replace('fri', 'friday').astype('category')

    # Step3(3): directly convert these features from object type to category type
    df['x34'] = df['x34'].astype('category')
    df['x68'] = df['x68'].astype('category')
    df['x93'] = df['x93'].astype('category')

    # Step3(4): convert category type to float without touching missing values
    cat_columns = df.select_dtypes(['category']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
    for item in list(df.select_dtypes(include='int8')):
        df[item] = df[item].replace(-1, nan)

    # Step3(5): double check the converted data frame
    print(df.info())
    print("================================")

    # Step4: handling missing values by building a KNN imputer
    knn = KNNImputer(n_neighbors=5)
    df1 = pd.DataFrame(knn.fit_transform(df), columns=list(df))

    # again, check whether the clean data contains missing values
    print("Does the dataset contain missing values?(T/F): ", df1.isnull().values.any())

    # Step5&6&7: split training data into training and validation, handle imbalanced training data, normalize all data features
    if isTrain:
        # split data in to training data and validation data
        X_train, X_val, y_train, y_val = train_test_split(df1.drop('y', axis=1), df1['y'], test_size=0.20, random_state=101)
        # handle imbalanced training data
        print("Training data before oversampling: ", Counter(y_train))
        ros = RandomOverSampler(random_state=42)
        X_train, y_train = ros.fit_resample(X_train, y_train)
        print("Training data after oversampling: ", Counter(y_train))
        # training data normalization
        X_train = (X_train - X_train.mean()) / X_train.std()

        print("Check whether the validation data is balance or not (T/F): ", Counter(y_val))
        # validation data normalization
        X_val = (X_val - X_val.mean()) / X_val.std()
        print("================================")
        return X_train, X_val, y_train, y_val
    else:
        X_test = (df1 - df1.mean()) / df1.std()
        return X_test
