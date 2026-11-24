from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="Organic Farming Suitability Prediction",
    layout="wide"
)

# File paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_organic_farming_pipeline.joblib"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"
DATA_PATH = BASE_DIR / "data" / "organic_farming_suitability_1000_source_calibrated.csv"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"
BENCHMARK_PATH = BASE_DIR / "outputs" / "reports" / "model_benchmark_results.csv"

# Header section
st.title("Machine Learning-Based Organic Farming Suitability Prediction")
st.write("**Name:** Nehalata Maji")
st.write("**Roll No:** BWU/BCA/24/541")
st.write("**Course:** Bachelor of Computer Applications (BCA)")
st.write("**Department:** Department of Computational Sciences, Brainware University, Kolkata")
st.write("---")

st.subheader("1. Project Overview")
st.write(
    "The goal of this project is to build a binary classification model that predicts whether a given "
    "agricultural field is Suitable or Not Suitable for organic farming based on soil nutrient levels, "
    "climatic factors, and chemical input history. "
    "The dataset consists of 1,000 source-calibrated instances prepared for educational purposes based on "
    "Soil Health Card parameters. As per instructions, Suitability_Score has been removed from model training "
    "to avoid target leakage."
)

# Load model and supporting data
@st.cache_resource
def load_all_resources():
    if not MODEL_PATH.exists():
        return None, None, None, None
    clf_pipeline = joblib.load(MODEL_PATH)
    raw_dataset = pd.read_csv(DATA_PATH) if DATA_PATH.exists() else None
    meta_info = json.load(open(METADATA_PATH)) if METADATA_PATH.exists() else {}
    bench_data = pd.read_csv(BENCHMARK_PATH) if BENCHMARK_PATH.exists() else None
    return clf_pipeline, raw_dataset, meta_info, bench_data

pipeline, data_df, metadata, benchmark_df = load_all_resources()

if pipeline is None:
    st.error("Model file not found. Please run: python run_pipeline.py")
    st.stop()

# Model benchmark summary
st.subheader("2. Model Comparison and Evaluation Summary")
if benchmark_df is not None:
    st.write("Summary of candidate models tested on the held-out test set (200 records):")
    st.dataframe(
        benchmark_df[["Model", "CV_Accuracy_Mean", "CV_F1_Mean", "Test_Accuracy", "Test_Precision", "Test_Recall", "Test_F1_Score", "Test_ROC_AUC"]],
        use_container_width=True
    )
    st.write(
        "Based on cross-validation and test evaluation, Gradient Boosting achieved the highest "
        "Test Accuracy (83.00%), Test F1-Score (0.8000), and Test ROC-AUC (0.9003) and was chosen as the final model."
    )

st.write("---")

# Feature input form
st.subheader("3. Test Sample Prediction")
st.write("Enter the soil, environmental, and past chemical input parameters below:")

# Dynamic options from data
states_list = sorted(data_df["State"].dropna().unique().tolist()) if data_df is not None else ["West Bengal", "Bihar", "Andhra Pradesh", "Karnataka", "Madhya Pradesh", "Rajasthan", "Haryana", "Tamil Nadu", "Telangana"]
soil_list = sorted(data_df["Soil_Type"].dropna().unique().tolist()) if data_df is not None else ["Alluvial", "Black", "Clay Loam", "Laterite", "Loamy", "Red", "Sandy"]
crop_list = sorted(data_df["Crop"].dropna().unique().tolist()) if data_df is not None else ["Chickpea", "Groundnut", "Mustard", "Pulses", "Rice", "Soybean", "Sugarcane", "Tomato", "Wheat"]
fert_options = ["Organic", "Mixed", "Chemical"]
pest_options = ["Low", "Moderate", "High"]

with st.form("input_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**Field and General Parameters**")
        state_val = st.selectbox("State", states_list, index=states_list.index("West Bengal") if "West Bengal" in states_list else 0)
        soil_val = st.selectbox("Soil Type", soil_list, index=soil_list.index("Alluvial") if "Alluvial" in soil_list else 0)
        crop_val = st.selectbox("Crop", crop_list, index=crop_list.index("Rice") if "Rice" in crop_list else 0)
        ph_val = st.number_input("Soil pH", min_value=3.0, max_value=11.0, value=6.8, step=0.1)
        oc_val = st.number_input("Organic Carbon (%)", min_value=0.05, max_value=3.0, value=0.85, step=0.01)
        ec_val = st.number_input("Electrical Conductivity (dS/m)", min_value=0.01, max_value=5.0, value=0.35, step=0.01)

    with col2:
        st.write("**Primary and Micronutrients**")
        n_val = st.number_input("Available Nitrogen (kg/ha)", min_value=10.0, max_value=600.0, value=250.0, step=5.0)
        p_val = st.number_input("Available Phosphorus (kg/ha)", min_value=1.0, max_value=120.0, value=35.0, step=1.0)
        k_val = st.number_input("Available Potassium (kg/ha)", min_value=10.0, max_value=600.0, value=210.0, step=5.0)
        s_val = st.number_input("Available Sulphur (mg/kg)", min_value=1.0, max_value=60.0, value=18.0, step=0.5)
        zn_val = st.number_input("Available Zinc (mg/kg)", min_value=0.05, max_value=10.0, value=1.10, step=0.05)
        fe_val = st.number_input("Available Iron (mg/kg)", min_value=0.5, max_value=60.0, value=12.0, step=0.5)

    with col3:
        st.write("**Weather and Input History**")
        mn_val = st.number_input("Available Manganese (mg/kg)", min_value=0.5, max_value=60.0, value=8.0, step=0.5)
        cu_val = st.number_input("Available Copper (mg/kg)", min_value=0.05, max_value=10.0, value=0.80, step=0.05)
        b_val = st.number_input("Available Boron (mg/kg)", min_value=0.05, max_value=6.0, value=0.60, step=0.05)
        rainfall_val = st.number_input("Annual Rainfall (mm)", min_value=200.0, max_value=3000.0, value=1200.0, step=25.0)
        temp_val = st.number_input("Mean Temperature (C)", min_value=10.0, max_value=45.0, value=27.0, step=0.5)
        moisture_val = st.number_input("Soil Moisture (%)", min_value=5.0, max_value=60.0, value=32.0, step=1.0)
        fert_val = st.selectbox("Previous Fertilizer Usage", fert_options, index=0)
        pest_val = st.selectbox("Previous Pesticide Usage", pest_options, index=0)
        years_chem_val = st.number_input("Years of Chemical Input", min_value=0, max_value=25, value=1, step=1)

    submit_button = st.form_submit_button("Run Prediction", use_container_width=True)

# Prediction output
if submit_button:
    sample_record = pd.DataFrame([{
        "State": state_val,
        "Soil_Type": soil_val,
        "Crop": crop_val,
        "pH": ph_val,
        "Organic_Carbon_pct": oc_val,
        "Available_N_kg_ha": n_val,
        "Available_P_kg_ha": p_val,
        "Available_K_kg_ha": k_val,
        "Available_S_mg_kg": s_val,
        "Available_Zn_mg_kg": zn_val,
        "Available_Fe_mg_kg": fe_val,
        "Available_Mn_mg_kg": mn_val,
        "Available_Cu_mg_kg": cu_val,
        "Available_B_mg_kg": b_val,
        "EC_dS_m": ec_val,
        "Annual_Rainfall_mm": rainfall_val,
        "Mean_Temperature_C": temp_val,
        "Soil_Moisture_pct": moisture_val,
        "Previous_Fertilizer_Usage": fert_val,
        "Previous_Pesticide_Usage": pest_val,
        "Years_Chemical_Input": years_chem_val
    }])

    pred = pipeline.predict(sample_record)[0]
    label = "Suitable" if pred == 1 else "Not Suitable"

    st.write("---")
    st.subheader("4. Prediction Output")

    out_col1, out_col2 = st.columns([1, 1.2])

    with out_col1:
        if label == "Suitable":
            st.success("Result: Suitable for Organic Farming")
        else:
            st.error("Result: Not Suitable for Organic Farming")

        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba(sample_record)[0]
            st.write(f"- Probability (Suitable): {probs[1] * 100:.2f}%")
            st.write(f"- Probability (Not Suitable): {probs[0] * 100:.2f}%")

    with out_col2:
        st.write("**Observations:**")
        if label == "Suitable":
            st.write(
                f"- Organic carbon is {oc_val}%, which supports soil biological activity.\n"
                f"- Soil pH is {ph_val}, which is in a favorable range for nutrient uptake.\n"
                f"- Previous chemical usage is low ({years_chem_val} years), meaning the soil can transition smoothly."
            )
        else:
            notes = []
            if oc_val < 0.60:
                notes.append(f"Organic carbon ({oc_val}%) is below the desirable level of 0.75%.")
            if years_chem_val >= 6:
                notes.append(f"Chemical input history ({years_chem_val} years) is high, indicating residual chemical load.")
            if ph_val < 6.0 or ph_val > 8.0:
                notes.append(f"Soil pH ({ph_val}) is outside the normal neutral range (6.5 to 7.5).")
            if fert_val == "Chemical":
                notes.append("Previous fertilizer practice was exclusively chemical.")
            if not notes:
                notes.append("Nutrient levels and soil conditions are currently imbalanced for organic requirements.")

            for item in notes:
                st.write(f"- {item}")
            st.write("A transition period of 2 to 3 years with organic manure (FYM or vermicompost) is suggested before full conversion.")

# Collapsible evaluation section
st.write("---")
with st.expander("5. View Model Evaluation Plots"):
    plot_tab1, plot_tab2, plot_tab3 = st.tabs(["Confusion Matrix", "ROC Curves", "Feature Importance"])

    with plot_tab1:
        cm_file = FIGURES_DIR / "09_confusion_matrices_all_models.png"
        if cm_file.exists():
            st.image(str(cm_file), caption="Confusion matrices for all six classification models on the test set", use_container_width=True)

    with plot_tab2:
        roc_file = FIGURES_DIR / "10_combined_roc_auc_curves.png"
        if roc_file.exists():
            st.image(str(roc_file), caption="Receiver Operating Characteristic (ROC) curves", use_container_width=True)

    with plot_tab3:
        imp_file = FIGURES_DIR / "13_permutation_importance_held_out.png"
        if imp_file.exists():
            st.image(str(imp_file), caption="Feature importance calculated using permutation importance on test data", use_container_width=True)
