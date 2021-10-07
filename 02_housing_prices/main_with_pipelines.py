from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

import pandas as pd


CARDINALITY_THRESHOLD: int = 10


def main() -> None:

    data = pd.DataFrame(
        pd.read_csv('./data/melb_data.csv')
    )

    y = data['Price']
    X = data.drop(['Price'], axis = 1)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, random_state = 0, test_size = 0.2, train_size = 0.8,
    )

    cat_cols_locard = [col_name for col_name in X_train.columns
                       if  X_train[col_name].nunique() < CARDINALITY_THRESHOLD
                       and X_train[col_name].dtype == 'object']


    num_cols = [col_name for col_name in X_train.columns
                if  X_train[col_name].dtype in ['int64', 'float64']]


    num_transformer = SimpleImputer(strategy = 'constant')

    cat_locard_transformer = Pipeline(steps = [
        ('imputer', SimpleImputer(strategy = 'most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown = 'ignore')),
    ])

    preprocessor = ColumnTransformer(transformers = [
        ('num', num_transformer, num_cols),
        ('cat_locard', cat_locard_transformer, cat_cols_locard),
    ])

    model = RandomForestRegressor(n_estimators = 100, random_state = 0)

    model_with_preprocessor = Pipeline(steps = [
        ('preprocessor', preprocessor),
        ('model', model),
    ])

    score = -1 * cross_val_score(
        model_with_preprocessor,
        X, y, cv = 5,
        scoring = 'neg_mean_absolute_error',
    )

    print(score.mean())

    # model_with_preprocessor.fit(X_train, y_train)
    # preds = model_with_preprocessor.predict(X_valid)
    # score = mean_absolute_percentage_error(preds, y_valid)
    # print(score)


if __name__ == "__main__":
    main()
