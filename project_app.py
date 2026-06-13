import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Crop Yield Prediction System",
    page_icon="🌾",
    layout="wide"
)

# ---------------- TITLE ----------------
st.title("🌾 Crop Yield Prediction System")
st.markdown(
    "### AI-Powered Agricultural Decision Support System"
)

st.divider()

# ---------------- MODEL PERFORMANCE ----------------
R2_SCORE = 0.8795
MAE_SCORE = 11.92

st.subheader("📈 Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric("R² Score", f"{R2_SCORE:.4f}")

with col2:
    st.metric("MAE", f"{MAE_SCORE:.2f}")

st.divider()

# ---------------- LOAD FILES ----------------
try:
    model = pickle.load(open("xgb_crop_yield_model.pkl", "rb"))

    le_state = pickle.load(open("le_state.pkl", "rb"))
    le_crop = pickle.load(open("le_crop.pkl", "rb"))
    le_season = pickle.load(open("le_season.pkl", "rb"))

except Exception as e:
    st.error(f"❌ Error loading model files: {e}")
    st.stop()

# ---------------- LOAD RAINFALL DATA ----------------
try:
    rainfall_df = pd.read_csv("rainfall_cleaned.csv")

except Exception as e:
    st.error(f"❌ Failed to load rainfall data: {e}")
    st.stop()

# ---------------- VALIDATE DATA ----------------
required_cols = {"state", "year", "rainfall"}

if not required_cols.issubset(rainfall_df.columns):
    st.error(
        f"❌ rainfall_cleaned.csv must contain columns: {required_cols}"
    )
    st.stop()

# ---------------- SIDEBAR ----------------
st.sidebar.header("🌱 Farm Inputs")

state = st.sidebar.selectbox(
    "Select State",
    le_state.classes_
)

crop = st.sidebar.selectbox(
    "Select Crop",
    le_crop.classes_
)

season = st.sidebar.selectbox(
    "Select Season",
    le_season.classes_
)

area = st.sidebar.number_input(
    "Cultivated Area (hectares)",
    min_value=1.0,
    max_value=100000.0,
    value=100.0
)

year = st.sidebar.number_input(
    "Cultivation Year",
    min_value=2015,
    max_value=2050,
    value=2026,
    step=1
)

# ---------------- AUTO RAINFALL ----------------

rainfall_row = rainfall_df[
    (rainfall_df["state"] == state)
    &
    (rainfall_df["year"] == year)
]

if not rainfall_row.empty:

    rainfall = rainfall_row["rainfall"].mean()
    rainfall_source = "Exact State-Year Match"

else:

    state_data = rainfall_df[
        rainfall_df["state"] == state
    ]

    if len(state_data) > 0:

        rainfall = state_data["rainfall"].mean()
        rainfall_source = "Historical State Average"

    else:

        rainfall = rainfall_df["rainfall"].mean()
        rainfall_source = "Global Dataset Average"

# ---------------- SIDEBAR RAINFALL INFO ----------------
st.sidebar.success(
    f"🌧️ Estimated Rainfall\n\n"
    f"{rainfall:.2f} mm\n\n"
    f"Source: {rainfall_source}"
)

# ---------------- FARM SUMMARY ----------------
st.subheader("📋 Farm Summary")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("State", state)

with col2:
    st.metric("Crop", crop)

with col3:
    st.metric("Season", season)

with col4:
    st.metric("Area", f"{area:.1f} ha")

with col5:
    st.metric("Year", int(year))

st.divider()

# ---------------- ENCODING ----------------
try:

    state_enc = le_state.transform([state])[0]
    crop_enc = le_crop.transform([crop])[0]
    season_enc = le_season.transform([season])[0]

except Exception as e:

    st.error(f"❌ Encoding Error: {e}")
    st.stop()

# ---------------- PREDICT BUTTON ----------------
if st.button("🚀 Predict Yield", use_container_width=True):

    try:

        # Must match training order:
        # state, year, Season, Crop, Area, rainfall

        input_data = np.array([[
            state_enc,
            year,
            season_enc,
            crop_enc,
            area,
            rainfall
        ]])

        prediction = model.predict(input_data)[0]

        # Prevent negative values
        prediction = max(0, prediction)

        estimated_production = prediction * area

        st.divider()

        st.subheader("🌾 Prediction Results")

        result1, result2 = st.columns(2)

        with result1:
            st.metric(
                "Predicted Yield",
                f"{prediction:.2f} t/ha"
            )

        with result2:
            st.metric(
                "Estimated Production",
                f"{estimated_production:.2f} tons"
            )

        st.divider()

        # ---------------- INTERPRETATION ----------------

        st.subheader("📊 Yield Interpretation")

        if prediction < 1:

            st.warning(
                "⚠️ Low yield expected under the selected conditions."
            )

        elif prediction < 3:

            st.info(
                "ℹ️ Moderate yield expected under average conditions."
            )

        else:

            st.success(
                "✅ High yield expected under favorable conditions."
            )

        # ---------------- MODEL DETAILS ----------------

        with st.expander("🔍 Model Information"):

            st.write("### Model Used")
            st.write("XGBoost Regressor")

            st.write("### Project")
            st.write(
                "Crop Yield Prediction Using Machine Learning"
            )

            st.write("### Input Features")
            st.write("- State")
            st.write("- Cultivation Year")
            st.write("- Season")
            st.write("- Crop")
            st.write("- Area")
            st.write("- Rainfall (Automatic)")

            st.write("### Output")
            st.write("- Predicted Yield (tons/hectare)")
            st.write("- Estimated Production (tons)")

            st.write("### Model Performance")
            st.write(f"R² Score: {R2_SCORE:.4f}")
            st.write(f"MAE: {MAE_SCORE:.2f}")

            st.write("### Feature Importance")

            feature_importance = pd.DataFrame({
                "Feature": [
                    "Rainfall",
                    "Crop",
                    "Year",
                    "State",
                    "Area",
                    "Season"
                ],
                "Importance": [
                    49.52,
                    18.31,
                    15.53,
                    8.91,
                    7.66,
                    0.06
                ]
            })

            st.dataframe(
                feature_importance,
                use_container_width=True
            )

    except Exception as e:

        st.error(f"❌ Prediction Failed: {e}")