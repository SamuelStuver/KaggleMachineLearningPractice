"""
Columns:

PassengerId - type should be integers
Survived - Survived or Not
Pclass - Class of Travel
Name - Name of Passenger
Sex - Gender
Age - Age of Passengers
SibSp - Number of Sibling/Spouse aboard
Parch - Number of Parent/Child aboard
Ticket
Fare
Cabin
Embarked - The port in which a passenger has embarked. C - Cherbourg, S - Southampton, Q = Queenstown
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def scoreDataset(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_absolute_error(y_test, preds)


def test_DropMissing(X_train, X_test, y_train, y_test):
    '''
    cols_with_missing = []
    for col in X_train.columns:
        print(X_train[col])
        for val in X_train[col]:
            print(val)
            if val.isnull():
                cols_with_missing.append(col)
    '''
    cols_with_missing = [
        col for col in X_train[X_train.columns].columns if X_train[col].isnull().any()]
    reduced_X_train = X_train.drop(cols_with_missing, axis=1)
    reduced_X_test = X_test.drop(cols_with_missing, axis=1)
    print("Mean Absolute Error from dropping columns with Missing Values:")
    score = scoreDataset(reduced_X_train, reduced_X_test, y_train, y_test)
    print(score)
    return score


def test_Imputation(X_train, X_test, y_train, y_test):
    my_imputer = SimpleImputer()
    imputed_X_train = my_imputer.fit_transform(X_train)
    imputed_X_test = my_imputer.transform(X_test)
    print("Mean Absolute Error from Imputation:")
    score = scoreDataset(imputed_X_train, imputed_X_test, y_train, y_test)
    print(score)
    return score


def reduce_DropMissing(X):
    cols_with_missing = [
        col for col in X.columns if X[col].isnull().any()]
    reduced_X = X.drop(cols_with_missing, axis=1)
    return reduced_X, cols_with_missing


def reduce_Imputation(X):
    my_imputer = SimpleImputer()
    imputed_X = my_imputer.fit_transform(X)
    return imputed_X, my_imputer


def main():
    # Load data
    trainingData = pd.read_csv('train.csv')
    trainingDataOriginal = trainingData.copy()
    # print(trainingData.columns)
    # print(trainingData.describe())

    #trainingData = trainingData.replace({'Sex': {'male': 1, 'female': 0}})
    #trainingData = trainingData.replace({'Embarked': {'C': 0, 'S': 1, 'Q': 2}})

    target = trainingData.Survived
    predictors = trainingData.drop(['Survived', 'Ticket'], axis=1)

    # For the sake of keeping the example simple, we'll use only numeric predictors.
    #numeric_predictors = predictors.select_dtypes(exclude=['object'])
    stringCols = ['Sex', 'Embarked']
    one_hot_encoded_predictors = pd.get_dummies(predictors, columns=stringCols)
    final_predictors = pd.concat(
        [predictors, one_hot_encoded_predictors], axis=1).drop(stringCols, axis=1)

    final_predictors = final_predictors.select_dtypes(
        exclude=['object'])
    final_predictors = final_predictors.loc[:,
                                            ~final_predictors.columns.duplicated()]
    X_train, X_test, y_train, y_test = train_test_split(final_predictors,
                                                        target,
                                                        train_size=0.7,
                                                        test_size=0.3,
                                                        random_state=0)

    # X_train, X_test = X_train.align(X_test,
    #                                join='left',
    #                                axis=1)

    score_Drop = test_DropMissing(X_train, X_test, y_train, y_test)
    score_Impute = test_Imputation(X_train, X_test, y_train, y_test)

    impute = False
    if score_Drop < score_Impute:
        print("Dropping Columns with missing values...")
        reduceFunc = reduce_DropMissing
    else:
        impute = True
        print("Imputing missing values...")
        reduceFunc = reduce_Imputation

    if impute:
        final_predictors, imputer = reduceFunc(
            final_predictors)
    else:
        final_predictors, droppedCols = reduceFunc(
            final_predictors)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(final_predictors, target)

    # load test data
    testData = pd.read_csv('test.csv')
    testDataOriginal = testData.copy()

    testData = testData.drop(['Ticket'], axis=1)

    #testData = testData.replace({'Sex': {'male': 1, 'female': 0}})
    #testData = testData.replace({'Embarked': {'C': 0, 'S': 1, 'Q': 2}})

    test_numeric_predictors_OHE = pd.get_dummies(testData, columns=stringCols)
    test_final_predictors = pd.concat(
        [testData, test_numeric_predictors_OHE], axis=1).drop(stringCols, axis=1)

    test_final_predictors = test_final_predictors.select_dtypes(exclude=[
                                                                'object'])

    test_final_predictors = test_final_predictors.loc[:,
                                                      ~test_final_predictors.columns.duplicated()]

    if impute:
        test_final_predictors = imputer.transform(
            test_final_predictors)
    else:
        test_final_predictors = test_final_predictors.drop(
            droppedCols, axis=1)

    # predict outcomes
    predictions = model.predict(test_final_predictors)
    results = testData['PassengerId'].to_frame()
    results['Survived'] = pd.Series(predictions, index=results.index)

    #results['Survived'] = pd.Series(map(int, predictions), index=results.index)

    print(results)
    print(results['Survived'].mean())
    results.to_csv('submission_new.csv', index=False)


if __name__ == "__main__":
    main()
