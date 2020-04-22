import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEng(BaseEstimator, TransformerMixin):
    def __init__(self, periods =[1], column = "signal", add_neg =False, fill_value=False):
        self.periods = periods
        self.column = column
        self.add_neg = add_neg
        self.fill_value = fill_value
        # self.copy = copy

    def fit(self, X, y):
        return self

    def transform(self,
     X: pd.DataFrame,
     y=None):

        periods = np.array(self.periods, dtype=np.int32)

        if self.add_neg:
            periods = np.append(periods, -periods)

        X_transformed = X.copy()
        for p in periods:
            X_transformed[f"{self.column}_shifted_{p}"] = X_transformed[self.column].shift(
                periods=p, fill_value= self.fill_value
            )
            # X_transformed[f"{self.column}_shifted_{p}_p4"] = X_transformed[f"{self.column}_shifted_{p}"]**4
        print ('Shape', X_transformed.shape)
        return X_transformed


class FeatureSelection(BaseEstimator, TransformerMixin):
    def __init__(self, drop_columns):
        self.drop_columns = drop_columns

    def fit(self, X,y):
        # pass
        return self

    def transform(self, X: pd.DataFrame, y=None):
        X = X[[c for c in X.columns if c not in self.drop_columns]]
        return X


# def add_category(train, test):
#     train["category"] = 0
#     test["category"] = 0

#     # train segments with more then 9 open channels classes
#     train.loc[2_000_000:2_500_000-1, 'category'] = 1
#     train.loc[4_500_000:5_000_000-1, 'category'] = 1

#     # test segments with more then 9 open channels classes (potentially)
#     test.loc[500_000:600_000-1, "category"] = 1
#     test.loc[700_000:800_000-1, "category"] = 1

#     return train, test
