
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from preprocessing import FeatureEng, FeatureSelection
import readingData
import preprocessing
from sklearn.metrics import f1_score, accuracy_score
from lightgbm import LGBMClassifier


def save_submission(y_test):
    submission = pd.read_csv("../mySubmission/sample_submission.csv")
    submission["open_channels"] = np.asarray(y_test, dtype=np.int32)
    submission.to_csv("submission_rfwithSquare.csv", index=False, float_format="%.4f")


if __name__ == "__main__":
    PATH = '../data/data-without-drift'
    shifted_rfc = make_pipeline(
        FeatureEng(
            periods=range(1, 20),
            add_neg=True,
            fill_value=0
        ),
        FeatureSelection(
            drop_columns=["open_channels", "time"]
        ),
        RandomForestClassifier(
            n_estimators=256,
            max_depth=12,
            max_features="auto",
            n_jobs=10,
            verbose=0
        )
    )
    train, test = readingData.readData(PATH, 'train_clean.csv', 'test_clean.csv' )
    train, test = preprocessing.add_category(train, test)
    shifted_rfc.fit(train, train.open_channels)
    open_channels = shifted_rfc.predict(test)
    predictions_train =  shifted_rfc.predict(train)
    f1accuracy = f1_score(train.open_channels,predictions_train , average = 'macro')
    acc = accuracy_score(train.open_channels,predictions_train )
    print (f"Accuracy: {acc} f1: {f1accuracy}")
    save_submission(open_channels)