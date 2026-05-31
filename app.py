import streamlit as st

from src.predict import predict_message

st.set_page_config(
    page_title="SMS Spam Detector"
)

st.title(
    "📩 SMS Spam Detection using SimpleRNN"
)

message = st.text_area(
    "Enter SMS Message"
)

if st.button(
    "Predict"
):

    if message:

        label, confidence = predict_message(
            message
        )

        if label == "Spam":

            st.error(
                f"{label}"
            )

        else:

            st.success(
                f"{label}"
            )

        st.write(
            f"Confidence: {confidence:.2%}"
        )