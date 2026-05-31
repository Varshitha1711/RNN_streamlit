import pickle

import pandas as pd

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import load_model

from preprocessing import clean_text

MAX_LEN = 50

df = pd.read_csv(
    "data/spam.csv",
    encoding="latin-1"
)

df = df.iloc[:, :2]

df.columns = [
    "label",
    "message"
]

df["message"] = df["message"].apply(
    clean_text
)

df["label"] = df["label"].map(
    {
        "ham": 0,
        "spam": 1
    }
)

X = df["message"]

y = df["label"]

_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

with open(
    "models/tokenizer.pkl",
    "rb"
) as f:

    tokenizer = pickle.load(
        f
    )

X_test_seq = tokenizer.texts_to_sequences(
    X_test
)

X_test_pad = pad_sequences(
    X_test_seq,
    maxlen=MAX_LEN
)

model = load_model(
    "models/rnn_model.keras"
)

pred = (
    model.predict(
        X_test_pad
    ) > 0.5
).astype(int)

print(
    classification_report(
        y_test,
        pred
    )
)

print(
    confusion_matrix(
        y_test,
        pred
    )
)