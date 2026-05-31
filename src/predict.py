import pickle

from tensorflow.keras.models import load_model

from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.preprocessing import clean_text

MAX_LEN = 50

model = load_model(
    "models/rnn_model.keras"
)

with open(
    "models/tokenizer.pkl",
    "rb"
) as f:

    tokenizer = pickle.load(
        f
    )


def predict_message(text):

    text = clean_text(
        text
    )

    seq = tokenizer.texts_to_sequences(
        [text]
    )

    padded = pad_sequences(
        seq,
        maxlen=MAX_LEN
    )

    score = float(
        model.predict(
            padded,
            verbose=0
        )[0][0]
    )

    if score > 0.5:

        return (
            "Spam",
            score
        )

    return (
        "Ham",
        1 - score
    )