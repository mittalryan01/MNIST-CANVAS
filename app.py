import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from streamlit_drawable_canvas import st_canvas

st.set_page_config(
    page_title="CNN Digit Recognizer",
    layout="wide"
)

st.markdown("""
<style>

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

.block-container {
    padding-top: 2rem;
    max-width: 1200px;
}

</style>
""", unsafe_allow_html=True)


if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "model/mnist_cnn.keras"
    )

model = load_model()

st.markdown(
    """
    <h1 style='text-align:center;'>
         CNN Digit Recognizer
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <p style='text-align:center; font-size:18px;'>
        Draw a digit (0-9) and click Predict
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

left, center, right = st.columns([1, 2, 1])

with center:

    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=20,
        stroke_color="white",
        background_color="black",
        width=350,
        height=350,
        drawing_mode="freedraw",
        display_toolbar=False,
        key=f"canvas_{st.session_state.canvas_key}"
    )

    btn1, btn2 = st.columns(2)

    with btn1:
        predict_btn = st.button(
            "Predict",
            use_container_width=True
        )

    with btn2:
        clear_btn = st.button(
            "🗑 Clear Canvas",
            use_container_width=True
        )

if clear_btn:
    st.session_state.canvas_key += 1
    st.rerun()

if predict_btn:

    if canvas_result.image_data is not None:

        image = canvas_result.image_data

        image = cv2.cvtColor(
            image.astype("uint8"),
            cv2.COLOR_RGBA2GRAY
        )

        image = cv2.resize(
            image,
            (28, 28)
        )

        image = image.astype("float32") / 255.0

        processed = image.copy()

        image = image.reshape(
            1,
            28,
            28,
            1
        )

        probs = model.predict(
            image,
            verbose=0
        )[0]

        pred = np.argmax(probs)

        st.divider()

        m1, m2 = st.columns(2)

        with m1:
            st.metric(
                "Predicted Digit",
                int(pred)
            )

        with m2:
            st.metric(
                "Confidence",
                f"{probs[pred] * 100:.2f}%"
            )

        st.divider()

        col1, col2 = st.columns(2)

        with col1:

            st.subheader(
                "Processed 28×28 Image"
            )

            st.image(
                processed,
                width=250
            )

        with col2:

            st.subheader(
                "Top 3 Predictions"
            )

            top3 = np.argsort(
                probs
            )[::-1][:3]

            for idx in top3:

                st.write(
                    f"**Digit {idx}** → {probs[idx] * 100:.2f}%"
                )

        st.divider()

        st.subheader(
            "Confidence Distribution"
        )

        for i, p in enumerate(probs):

            st.write(
                f"**Digit {i}** : {p * 100:.2f}%"
            )

            st.progress(
                float(p)
            )

    else:

        st.warning(
            "Please draw a digit first."
        )