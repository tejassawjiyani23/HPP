
import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

FEATURES = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "grade",
    "yr_built"
]

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models",
    "best_model.pkl"
)


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


model = load_model()

st.title("🏠 House Price Prediction")
st.subheader("Machine Learning Based Real Estate Price Estimator")

st.write(
    "Enter the property details below to estimate the "
    "house price using the trained machine learning model."
)

st.divider()

if model is None:
    st.error(
        "Trained model not found. Please run the notebook first "
        "to generate models/best_model.pkl."
    )
    st.stop()

with st.form("house_prediction_form"):
    st.subheader("Property Details")

    col1, col2 = st.columns(2)

    with col1:
        bedrooms = st.number_input(
            "Number of Bedrooms",
            min_value=0,
            max_value=20,
            value=3,
            step=1
        )

        bathrooms = st.number_input(
            "Number of Bathrooms",
            min_value=0.0,
            max_value=10.0,
            value=2.5,
            step=0.5
        )

        sqft_living = st.number_input(
            "Living Area (sq. ft.)",
            min_value=1,
            max_value=50000,
            value=1800,
            step=100
        )

        sqft_lot = st.number_input(
            "Lot Area (sq. ft.)",
            min_value=1,
            max_value=1000000,
            value=5000,
            step=500
        )

        floors = st.number_input(
            "Number of Floors",
            min_value=1.0,
            max_value=4.0,
            value=1.5,
            step=0.5
        )

    with col2:
        waterfront_option = st.selectbox(
            "Waterfront Property",
            ["No", "Yes"]
        )

        view = st.selectbox(
            "View Rating",
            options=[0, 1, 2, 3, 4],
            index=0
        )

        condition = st.selectbox(
            "House Condition (1–5)",
            options=[1, 2, 3, 4, 5],
            index=2
        )

        grade = st.selectbox(
            "Construction Grade (1–13)",
            options=list(range(1, 14)),
            index=6
        )

        yr_built = st.number_input(
            "Year Built",
            min_value=1800,
            max_value=2026,
            value=1990,
            step=1
        )

    predict_button = st.form_submit_button(
        "Predict House Price",
        use_container_width=True
    )

if predict_button:
    waterfront = 1 if waterfront_option == "Yes" else 0

    input_data = pd.DataFrame(
        [[
            bedrooms,
            bathrooms,
            sqft_living,
            sqft_lot,
            floors,
            waterfront,
            view,
            condition,
            grade,
            yr_built
        ]],
        columns=FEATURES
    )

    try:
        prediction = float(model.predict(input_data)[0])

        st.divider()
        st.subheader("Estimated House Price")

        if prediction >= 0:
            st.success(f"Estimated Price: ${prediction:,.2f}")
        else:
            st.warning(
                "The model returned a negative estimate. "
                "Please try different property details."
            )

        st.caption(
            "This is a machine learning estimate, not a guaranteed "
            "market price. Actual prices may vary."
        )

        with st.expander("View Submitted Property Details"):
            st.dataframe(input_data, use_container_width=True)

    except Exception as error:
        st.error(f"Prediction failed: {error}")

st.divider()
st.caption("House Price Prediction | Machine Learning Project")