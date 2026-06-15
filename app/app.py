import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Page config
st.set_page_config(
    page_title="CriticalCare AI",
    page_icon="🏥",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    model = joblib.load('models/best_model.pkl')
    features = joblib.load('models/feature_names.pkl')
    return model, features

model, feature_names = load_model()

# Header
st.title("🏥 CriticalCare AI")
st.markdown("### ICU Patient Mortality Risk Predictor")
st.markdown("*Powered by LightGBM + SHAP Explainability*")
st.divider()

# Sidebar inputs
st.sidebar.header("🔬 Patient Vitals Input")

age = st.sidebar.slider("Age", 18, 100, 65)
ventilated = st.sidebar.selectbox("Ventilated?", [0, 1], format_func=lambda x: "Yes" if x else "No")
gcs_motor = st.sidebar.slider("GCS Motor Score (1-6)", 1, 6, 4)
gcs_verbal = st.sidebar.slider("GCS Verbal Score (1-5)", 1, 5, 3)
gcs_eyes = st.sidebar.slider("GCS Eyes Score (1-4)", 1, 4, 3)
d1_spo2_min = st.sidebar.slider("Min SpO2 (%)", 50, 100, 92)
d1_heartrate_min = st.sidebar.slider("Min Heart Rate", 20, 150, 60)
d1_heartrate_max = st.sidebar.slider("Max Heart Rate", 40, 200, 110)
d1_resprate_min = st.sidebar.slider("Min Resp Rate", 5, 40, 12)
d1_resprate_max = st.sidebar.slider("Max Resp Rate", 10, 60, 24)
d1_sysbp_min = st.sidebar.slider("Min Systolic BP", 50, 200, 90)
d1_temp_min = st.sidebar.slider("Min Temperature (°C)", 30.0, 42.0, 36.5)
elective_surgery = st.sidebar.selectbox("Elective Surgery?", [0, 1], format_func=lambda x: "Yes" if x else "No")
pre_icu_los_days = st.sidebar.slider("Pre-ICU LOS (days)", 0, 30, 1)

# Predict button
if st.sidebar.button("🔍 Predict Mortality Risk", type="primary"):

    # Build input — fill all features with median, override known ones
    input_data = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)

    known = {
        'age': age, 'ventilated_apache': ventilated,
        'gcs_motor_apache': gcs_motor, 'gcs_verbal_apache': gcs_verbal,
        'gcs_eyes_apache': gcs_eyes, 'd1_spo2_min': d1_spo2_min,
        'd1_heartrate_min': d1_heartrate_min, 'd1_heartrate_max': d1_heartrate_max,
        'd1_resprate_min': d1_resprate_min, 'd1_resprate_max': d1_resprate_max,
        'd1_sysbp_min': d1_sysbp_min, 'd1_temp_min': d1_temp_min,
        'elective_surgery': elective_surgery, 'pre_icu_los_days': pre_icu_los_days
    }
    for col, val in known.items():
        if col in input_data.columns:
            input_data[col] = val

    # Predict
    prob = model.predict_proba(input_data)[0][1]
    risk_pct = prob * 100

    # Result
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Mortality Risk", f"{risk_pct:.1f}%")

    with col2:
        if risk_pct < 20:
            st.success("🟢 LOW RISK")
        elif risk_pct < 50:
            st.warning("🟡 MODERATE RISK")
        else:
            st.error("🔴 HIGH RISK")

    with col3:
        st.metric("Survival Probability", f"{100 - risk_pct:.1f}%")

    st.divider()

    # SHAP explanation
    st.subheader("🧠 Why this prediction? (SHAP)")
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(input_data)

    fig, ax = plt.subplots(figsize=(10, 4))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_vals[0] if isinstance(shap_vals, list) else shap_vals[0],
            base_values=explainer.expected_value if not isinstance(explainer.expected_value, list) else explainer.expected_value[1],
            data=input_data.iloc[0],
            feature_names=feature_names
        ),
        show=False
    )
    st.pyplot(fig)

else:
    st.info("👈 Patient vitals sidebar mein fill karo aur **Predict** dabao!")
    
    # Show SHAP summary
    st.subheader("📊 Model Insights")
    col1, col2 = st.columns(2)
    with col1:
        st.image("models/shap_summary.png", caption="Feature Importance")
    with col2:
        st.image("models/shap_beeswarm.png", caption="SHAP Beeswarm")