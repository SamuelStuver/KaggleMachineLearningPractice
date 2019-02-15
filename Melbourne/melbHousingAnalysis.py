import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def scoreDataset(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_absolute_error(y_test, preds)


def reduce_DropMissing(X_train, X_test, y_train, y_test):
    cols_with_missing = [
        col for col in X_train.columns if X_train[col].isnull().any()]
    reduced_X_train = X_train.drop(cols_with_missing, axis=1)
    reduced_X_test = X_test.drop(cols_with_missing, axis=1)
    print("Mean Absolute Error from dropping columns with Missing Values:")
    score = scoreDataset(reduced_X_train, reduced_X_test, y_train, y_test)
    print(score)
    return reduced_X_train, reduced_X_test, y_train, y_test


def reduce_Imputation(X_train, X_test, y_train, y_test):
    my_imputer = SimpleImputer()
    imputed_X_train = my_imputer.fit_transform(X_train)
    imputed_X_test = my_imputer.transform(X_test)
    print("Mean Absolute Error from Imputation:")
    score = scoreDataset(imputed_X_train, imputed_X_test, y_train, y_test)
    print(score)
    return imputed_X_train, imputed_X_test, y_train, y_test


def main():
    # Load data
    melb_data = pd.read_csv('melb_data.csv')

    melb_target = melb_data.Price
    melb_predictors = melb_data.drop(['Price'], axis=1)

    # For the sake of keeping the example simple, we'll use only numeric predictors.
    melb_numeric_predictors = melb_predictors.select_dtypes(exclude=['object'])

    X_train, X_test, y_train, y_test = train_test_split(melb_numeric_predictors,
                                                        melb_target,
                                                        train_size=0.7,
                                                        test_size=0.3,
                                                        random_state=0)

    reduce_DropMissing(X_train, X_test, y_train, y_test)
    reduce_Imputation(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
