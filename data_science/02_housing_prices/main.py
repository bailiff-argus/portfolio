from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from xgboost import XGBRegressor

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


    cols = {
        'numerical':   [col_name for col_name in X_train.columns
                        if  X_train[col_name].dtype in ('int64', 'float64')],

        'categorical': [col_name for col_name in X_train.columns
                        if  X_train[col_name].dtype == 'object'
                        and X_train[col_name].nunique() <= CARDINALITY_THRESHOLD],
    }

    transformers = {
        'numerical':   SimpleImputer(strategy = 'constant'),

        'categorical': Pipeline(steps = [
            ('impute', SimpleImputer(strategy = 'most_frequent')),
            ('encode', OneHotEncoder(handle_unknown = 'ignore')),
        ]),
    }


    preprocessor = ColumnTransformer(transformers = [
        ('num', transformers['numerical'],   cols['numerical']),
        ('cat', transformers['categorical'], cols['categorical']),
    ])


    preprocessor_pipe = Pipeline(steps = [
        ('preprocessor', preprocessor),
    ])


    model = XGBRegressor(n_estimators = 1000, learning_rate = 0.05, random_state = 0)

    model_pipeline = Pipeline(steps = [
        ('preprocessor', preprocessor),
        ('model', model),
    ])


    # pipeline doesn't process the eval_set
    X_valid_eval = X_valid.copy()
    X_valid_eval = preprocessor_pipe.fit(X_train, y_train).transform(X_valid_eval)


    model_pipeline.fit(
        X_train, y_train,
        model__early_stopping_rounds = 5,
        model__eval_set = [(X_valid_eval, y_valid)],
        model__verbose = False,
    )

    preds = model_pipeline.predict(X_valid)
    score = mean_absolute_error(preds, y_valid)

    print(score)


if __name__ == "__main__":
    main()
