from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.impute import SimpleImputer
import pandas as pd


CARDINALITY_THRESHOLD: int = 10


def extract_numerical_features(X: pd.DataFrame) -> pd.DataFrame:
    X_num = X.select_dtypes(exclude = ['object'])
    return X_num


def extract_low_cardinality_categorical_features(X: pd.DataFrame) -> pd.DataFrame:
    target_cols = [col_name for col_name in X.columns
                   if X[col_name].nunique() <= CARDINALITY_THRESHOLD
                   and X[col_name].dtype == 'object']
    X_cat_locard = X[target_cols]
    return X_cat_locard


def encode_low_cardinality_categorical_features(
    X_train_locard: pd.DataFrame, X_valid_locard: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:

    full_dataset = pd.concat([X_train_locard, X_valid_locard], axis = 0)

    encoder = OneHotEncoder(handle_unknown = 'ignore', sparse = False)
    encoder.fit(full_dataset)

    X_train_locard_encoded = pd.DataFrame(encoder.transform(X_train_locard))
    X_valid_locard_encoded = pd.DataFrame(encoder.transform(X_valid_locard))

    X_train_locard_encoded.columns = X_train_locard_encoded.columns.astype(str)
    X_valid_locard_encoded.columns = X_valid_locard_encoded.columns.astype(str)

    X_train_locard_encoded.index = X_train_locard.index
    X_valid_locard_encoded.index = X_valid_locard.index

    return X_train_locard_encoded, X_valid_locard_encoded


def extract_high_cardinality_categorical_features(X: pd.DataFrame) -> pd.DataFrame:
    target_cols = [col_name for col_name in X.columns
                   if X[col_name].nunique() > CARDINALITY_THRESHOLD
                   and X[col_name].dtype == 'object']
    X_cat_hicard = X[target_cols]
    return X_cat_hicard


def encode_high_cardinality_categorical_features(
    X_train_hicard: pd.DataFrame, X_valid_hicard: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:

    full_dataset = pd.concat([X_train_hicard, X_valid_hicard], axis = 0)

    encoder = OrdinalEncoder()
    encoder.fit(full_dataset)

    X_train_hicard_encoded = pd.DataFrame(encoder.transform(X_train_hicard))
    X_valid_hicard_encoded = pd.DataFrame(encoder.transform(X_valid_hicard))

    X_train_hicard_encoded.columns = X_train_hicard_encoded.columns.astype(str)
    X_valid_hicard_encoded.columns = X_valid_hicard_encoded.columns.astype(str)

    X_train_hicard_encoded.index = X_train_hicard.index
    X_valid_hicard_encoded.index = X_valid_hicard.index

    return X_train_hicard_encoded, X_valid_hicard_encoded


def impute_predictors(
    X_train_nonimp: pd.DataFrame, X_valid_nonimp: pd.DataFrame
) -> tuple:

    full_dataset = pd.concat([X_train_nonimp, X_valid_nonimp], axis = 0)

    imputer = SimpleImputer(strategy = 'mean')
    imputer.fit(full_dataset)

    X_train = pd.DataFrame(imputer.transform(X_train_nonimp))
    X_valid = pd.DataFrame(imputer.transform(X_valid_nonimp))

    return X_train, X_valid


def split_and_preprocess_data(
    X: pd.DataFrame, y: pd.Series
) -> tuple:

    X_train_raw, X_valid_raw, y_train, y_valid = train_test_split(X, y)

    # select numerical features
    X_train_num = extract_numerical_features(X_train_raw)
    X_valid_num = extract_numerical_features(X_valid_raw)


    # preprocess low cardinality categorical features
    X_train_locard_raw = extract_low_cardinality_categorical_features(X_train_raw)
    X_valid_locard_raw = extract_low_cardinality_categorical_features(X_valid_raw)

    X_train_locard, X_valid_locard = encode_low_cardinality_categorical_features(
        X_train_locard_raw, X_valid_locard_raw
    )


    # preprocess high cardinality categorical features
    X_train_hicard_raw = extract_high_cardinality_categorical_features(X_train_raw)
    X_valid_hicard_raw = extract_high_cardinality_categorical_features(X_valid_raw)

    X_train_hicard, X_valid_hicard = encode_high_cardinality_categorical_features(
        X_train_hicard_raw, X_valid_hicard_raw
    )


    # merge predictive features into training and validation datasets
    training_data = [X_train_num, X_train_locard, X_train_hicard]
    X_train_nonimputed = pd.concat(training_data, axis = 1)

    validation_data = [X_valid_num, X_valid_locard, X_valid_hicard]
    X_valid_nonimputed = pd.concat(validation_data, axis = 1)


    # impute missing data
    X_train, X_valid = impute_predictors(X_train_nonimputed, X_valid_nonimputed)

    return X_train, X_valid, y_train, y_valid


def main() -> None:
    housing_data: pd.DataFrame = pd.read_csv("./data/melb_data.csv")

    y = housing_data['Price']
    X = housing_data.drop(['Price'], axis = 1)

    X_train, X_valid, y_train, y_valid = split_and_preprocess_data(X, y)

    model = RandomForestRegressor(n_estimators = 500, random_state = 0)
    model.fit(X_train, y_train)

    predictions = model.predict(X_valid)
    mae = mean_absolute_percentage_error(y_valid, predictions)

    print(round(mae, 5))


if __name__ == "__main__":
    main()
