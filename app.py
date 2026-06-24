# ============================================================
# DATA 690 CAPSTONE PROJECT
# Elevated Diabetes Risk Prediction Application
# Author: Shankar Mohankumar
#
# IMPORTANT ENVIRONMENT INFORMATION
# Environment Name: diabetes_streamlit_py312
# Python: 3.12
# TensorFlow: 2.20.0
# Keras: 3.13.2
# ============================================================

import streamlit as st
import joblib
import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np
import shap

st.set_page_config(
    page_title="Diabetes Risk Prediction App",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Elevated Diabetes Risk Prediction App")
st.subheader("Created by Shankar Mohankumar")

try:
    scaler = joblib.load("models/scaler.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")

    lr_model = joblib.load("models/logistic_regression_model.pkl")
    rf_model = joblib.load("models/random_forest_model.pkl")
    xgb_model = joblib.load("models/xgboost_model.pkl")
    ann_model = load_model("models/ann_model.keras", compile=False)

    st.success("✅ All models loaded successfully")

except Exception as e:
    st.error("❌ Error loading models")
    st.exception(e)
    st.stop()


def yes_no_input(label):
    return st.radio(label, options=["No", "Yes"], horizontal=True)


def yes_no_to_binary(value):
    return 1.0 if value == "Yes" else 0.0


def age_to_bin(age):
    if age <= 24:
        return 1.0
    elif age <= 29:
        return 2.0
    elif age <= 34:
        return 3.0
    elif age <= 39:
        return 4.0
    elif age <= 44:
        return 5.0
    elif age <= 49:
        return 6.0
    elif age <= 54:
        return 7.0
    elif age <= 59:
        return 8.0
    elif age <= 64:
        return 9.0
    elif age <= 69:
        return 10.0
    elif age <= 74:
        return 11.0
    elif age <= 79:
        return 12.0
    else:
        return 13.0

st.info(
    "Privacy Notice: Information entered into this application is used only to generate the current prediction and is NOT STORED."
)
st.divider()

st.header("Patient / User Health Inputs")
st.divider()
st.subheader("Demographics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    age_actual = st.slider("Age", min_value=18, max_value=100, value=35)
    Age = age_to_bin(age_actual)

with col2:
    Sex_label = st.radio("Sex", options=["Female", "Male"], horizontal=True)
    Sex = 1.0 if Sex_label == "Male" else 0.0

with col3:
    Education_label = st.selectbox(
        "Education",
        options=[
            "Never attended school / Kindergarten",
            "Grades 1-8",
            "Grades 9-11",
            "High School Graduate",
            "Some College / Technical School",
            "College Graduate"
        ]
    )

    Education = {
        "Never attended school / Kindergarten": 1.0,
        "Grades 1-8": 2.0,
        "Grades 9-11": 3.0,
        "High School Graduate": 4.0,
        "Some College / Technical School": 5.0,
        "College Graduate": 6.0
    }[Education_label]

with col4:
    Income_label = st.selectbox(
        "Income",
        options=[
            "Less than $10,000",
            "$10,000-$14,999",
            "$15,000-$19,999",
            "$20,000-$24,999",
            "$25,000-$34,999",
            "$35,000-$49,999",
            "$50,000-$74,999",
            "$75,000+"
        ]
    )

    Income = {
        "Less than $10,000": 1.0,
        "$10,000-$14,999": 2.0,
        "$15,000-$19,999": 3.0,
        "$20,000-$24,999": 4.0,
        "$25,000-$34,999": 5.0,
        "$35,000-$49,999": 6.0,
        "$50,000-$74,999": 7.0,
        "$75,000+": 8.0
    }[Income_label]

st.divider()
st.subheader("Body Mass Index (BMI)")

bmi_method = st.radio(
    "How would you like to enter BMI?",
    options=[
        "Enter BMI directly",
        "Use height and weight (Imperial)",
        "Use height and weight (Metric)"
    ]
)

if bmi_method == "Enter BMI directly":
    BMI = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=100.0,
        value=25.0,
        step=0.1
    )

elif bmi_method == "Use height and weight (Imperial)":
    col1, col2, col3 = st.columns(3)

    with col1:
        feet = st.number_input("Height (feet)", min_value=3, max_value=8, value=5)

    with col2:
        inches = st.number_input("Additional inches", min_value=0, max_value=11, value=8)

    with col3:
        weight_lbs = st.number_input("Weight (lbs)", min_value=50.0, max_value=700.0, value=170.0)

    total_inches = (feet * 12) + inches
    BMI = (weight_lbs / (total_inches ** 2)) * 703
    st.info(f"Calculated BMI = {BMI:.1f}")

else:
    col1, col2 = st.columns(2)

    with col1:
        height_cm = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0)

    with col2:
        weight_kg = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0, value=75.0)

    height_m = height_cm / 100
    BMI = weight_kg / (height_m ** 2)
    st.info(f"Calculated BMI = {BMI:.1f}")

st.divider()
st.subheader("General Health")

GenHlth_label = st.selectbox(
    "What would you say that in general your health is?",
    options=[
        "Excellent",
        "Very Good",
        "Good",
        "Fair",
        "Poor"
    ]
)

GenHlth = {
    "Excellent": 1.0,
    "Very Good": 2.0,
    "Good": 3.0,
    "Fair": 4.0,
    "Poor": 5.0
}[GenHlth_label]

st.divider()
st.subheader("Mental Health")
mental_col = st.columns([3, 2])

with mental_col[0]:

    st.write(
        "Now thinking about your mental health, which includes stress, depression, and problems with emotions,"
    )

    st.markdown(
        "**for how many days during the past 30 days was your mental health not good?**"
    )

    MentHlth = st.slider(
    "",
    min_value=0,
    max_value=30,
    value=0,
    key="mental_health_slider"
)
st.divider()
st.subheader("Physical Health")
physical_col = st.columns([3, 2])

with physical_col[0]:

    st.write(
        "Now thinking about your physical health, which includes physical illness and injury,"
    )

    st.markdown(
        "**for how many days during the past 30 days was your physical health not good?**"
    )

    PhysHlth = st.slider(
    "",
    min_value=0,
    max_value=30,
    value=0,
    key="physical_health_slider"
)
st.divider()
st.subheader("Health Conditions")

col1, col2 = st.columns(2)

with col1:
    HighBP = yes_no_to_binary(
        yes_no_input("Do you have High Blood Pressure?")
    )

    HighChol = yes_no_to_binary(
        yes_no_input("Do you have High Cholesterol?")
    )

    CholCheck = yes_no_to_binary(
        yes_no_input("Have you had your Cholesterol Checked with in the past 5 years?")
    )

    Stroke = yes_no_to_binary(
        yes_no_input("Do you have any History of Stroke")
    )

with col2:
    HeartDiseaseorAttack = yes_no_to_binary(
        yes_no_input("Do you have any history of Heart Disease or Heart Attack")
    )

    DiffWalk = yes_no_to_binary(
        yes_no_input("Do you have serious Difficulty Walking or climbing stairs?")
    )

    AnyHealthcare = yes_no_to_binary(
        yes_no_input("Do you have any kind of health care coverage, including health insurance, prepaid plans such as HMOs, or government plans")
    )

    NoDocbcCost = yes_no_to_binary(
        yes_no_input("Was there a time in the past 12 months when you needed to see a doctor but could not because of cost?")
    )

st.divider()
st.subheader("Lifestyle Factors")

col1, col2, col3 = st.columns(3)

with col1:
    Smoker = yes_no_to_binary(
        yes_no_input("Are you a Smoker or have any History of smoking at least 100 cigarettes in your entire life?")
    )

    PhysActivity = yes_no_to_binary(
        yes_no_input("Have you done any physical activity or exercise during the past 30 days other than your regular job?")
    )

with col2:
    Fruits = yes_no_to_binary(
        yes_no_input("Do you consume Fruit 1 or more times per day?")
    )

    Veggies = yes_no_to_binary(
        yes_no_input("Do you consume Vegetables 1 or more times per day?")
    )

with col3:
    HvyAlcoholConsump = yes_no_to_binary(
        yes_no_input("Do you have history of heavy alochol consumption? (adult men having more than 14 drinks per week and adult women having more than 7 drinks per week)")
    )


st.divider()

st.success("Complete all fields above, then click the button below to generate the prediction.")

predict_button = st.button(
    "🔍 Predict Diabetes Risk",
    use_container_width=True
)


# ============================================================

if predict_button:

    input_data = pd.DataFrame([{
        "HighBP": HighBP,
        "HighChol": HighChol,
        "CholCheck": CholCheck,
        "BMI": float(BMI),
        "Smoker": Smoker,
        "Stroke": Stroke,
        "HeartDiseaseorAttack": HeartDiseaseorAttack,
        "PhysActivity": PhysActivity,
        "Fruits": Fruits,
        "Veggies": Veggies,
        "HvyAlcoholConsump": HvyAlcoholConsump,
        "AnyHealthcare": AnyHealthcare,
        "NoDocbcCost": NoDocbcCost,
        "GenHlth": GenHlth,
        "MentHlth": float(MentHlth),
        "PhysHlth": float(PhysHlth),
        "DiffWalk": DiffWalk,
        "Sex": Sex,
        "Age": Age,
        "Education": Education,
        "Income": Income
    }])

    input_data = input_data[feature_columns]

    st.success("Input data successfully created")

    #st.dataframe(input_data)

# ================LR============================================

    st.divider()


    input_data_scaled = scaler.transform(input_data)

    # Logistic Regression
    lr_probability = lr_model.predict_proba(input_data_scaled)[0][1]
    lr_prediction = (
        "Elevated Diabetes Risk"
        if lr_probability >= 0.50
        else "Lower Diabetes Risk"
    )

    # Random Forest
    rf_probability = rf_model.predict_proba(input_data)[0][1]
    rf_prediction = (
        "Elevated Diabetes Risk"
        if rf_probability >= 0.50
        else "Lower Diabetes Risk"
    )

    # XGBoost
    xgb_probability = xgb_model.predict_proba(input_data)[0][1]
    xgb_prediction = (
        "Elevated Diabetes Risk"
        if xgb_probability >= 0.50
        else "Lower Diabetes Risk"
    )

    # ANN
    ann_probability = ann_model.predict(
        input_data_scaled,
        verbose=0
    )[0][0]

    ann_prediction = (
        "Elevated Diabetes Risk"
        if ann_probability >= 0.50
        else "Lower Diabetes Risk"
    )


    st.subheader("Model Prediction Results")

    results_df = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Random Forest",
            "XGBoost",
            "Artificial Neural Network"
        ],
        "Prediction": [
            f"<span style='color:red'><b>{lr_prediction}</b></span>"
            if lr_prediction == "Elevated Diabetes Risk"
            else f"<span style='color:limegreen'><b>{lr_prediction}</b></span>",

            f"<span style='color:red'><b>{rf_prediction}</b></span>"
            if rf_prediction == "Elevated Diabetes Risk"
            else f"<span style='color:limegreen'><b>{rf_prediction}</b></span>",

            f"<span style='color:red'><b>{xgb_prediction}</b></span>"
            if xgb_prediction == "Elevated Diabetes Risk"
            else f"<span style='color:limegreen'><b>{xgb_prediction}</b></span>",

            f"<span style='color:red'><b>{ann_prediction}</b></span>"
            if ann_prediction == "Elevated Diabetes Risk"
            else f"<span style='color:limegreen'><b>{ann_prediction}</b></span>"
        ],
        "Risk Probability": [
            f"{lr_probability:.1%}",
            f"{rf_probability:.1%}",
            f"{xgb_probability:.1%}",
            f"{ann_probability:.1%}"
        ]
    })

    st.markdown(
        """
        <style>
        table {
            width: 100%;
            font-size: 24px;
        }

        th {
            font-size: 28px !important;
            text-align: center !important;
        }

        td {
            font-size: 24px !important;
            text-align: center !important;
            padding: 12px !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        results_df.to_html(index=False, escape=False),
        unsafe_allow_html=True
    )

    st.caption(
        "Risk probability represents the likelihood of elevated diabetes risk predicted by each model."
    )

    elevated_count = sum([
        lr_prediction == "Elevated Diabetes Risk",
        rf_prediction == "Elevated Diabetes Risk",
        xgb_prediction == "Elevated Diabetes Risk",
        ann_prediction == "Elevated Diabetes Risk"
    ])

    average_probability = (
        lr_probability +
        rf_probability +
        xgb_probability +
        ann_probability
    ) / 4

    st.markdown(
        f"""
        <div style='text-align:center; margin-top:20px; font-size:36px; font-weight:bold;'>
        {elevated_count} of 4 models predict Elevated Diabetes Risk
        <br>
        Average Risk Probability: {average_probability:.1%}
        </div>
        """,
        unsafe_allow_html=True
    )

#----------------------------------------------
    st.divider()
    st.subheader("Interpretability Analysis")
    st.info(
        "Interpretability analysis may take several seconds to run, especially for the Artificial Neural Network model."
    )

    with st.expander("🔎 Click to view which inputs had the highest influence on the model"):

        
        st.subheader("Top 5 Influential Features by Model")

        st.caption(
            "Positive influence values (+) push the model toward elevated diabetes risk. Negative influence values (-) push the model toward lower diabetes risk. Larger absolute values indicate a stronger influence on the model's decision."
        )

        try:
            shap_background_raw = joblib.load("models/shap_background_raw.pkl")
            shap_background_scaled = joblib.load("models/shap_background_scaled.pkl")

            def extract_class_1_shap(shap_values):
                if isinstance(shap_values, list):
                    values = shap_values[1]
                else:
                    values = shap_values

                values = np.array(values)

                if values.ndim == 3:
                    values = values[:, :, 1]

                return values[0]

            def make_influence_table(values):
                influence_df = pd.DataFrame({
                    "Feature": feature_columns,
                    "Influence": values
                })

                influence_df["Absolute Influence"] = influence_df["Influence"].abs()

                influence_df = influence_df.sort_values(
                    by="Absolute Influence",
                    ascending=False
                )

                influence_df["Influence"] = influence_df["Influence"].map(
                    lambda x: f"{x:+.4f}"
                )

                return influence_df[["Feature", "Influence"]]

            # Logistic Regression
            lr_explainer = shap.LinearExplainer(
                lr_model,
                shap_background_scaled
            )
            lr_shap_values = lr_explainer.shap_values(input_data_scaled)
            lr_values = extract_class_1_shap(lr_shap_values)
            lr_influence = make_influence_table(lr_values)

            # Random Forest
            rf_explainer = shap.TreeExplainer(rf_model)
            rf_shap_values = rf_explainer.shap_values(input_data)
            rf_values = extract_class_1_shap(rf_shap_values)
            rf_influence = make_influence_table(rf_values)

            # XGBoost
            xgb_explainer = shap.TreeExplainer(xgb_model)
            xgb_shap_values = xgb_explainer.shap_values(input_data)
            xgb_values = extract_class_1_shap(xgb_shap_values)
            xgb_influence = make_influence_table(xgb_values)

            # ANN
            ann_background = shap_background_scaled[:50]

            ann_explainer = shap.KernelExplainer(
                lambda x: ann_model.predict(x, verbose=0).ravel(),
                ann_background
            )

            ann_shap_values = ann_explainer.shap_values(
                input_data_scaled,
                nsamples=100
            )

            ann_values = extract_class_1_shap(ann_shap_values)
            ann_influence = make_influence_table(ann_values)

            table_col, spacer_col = st.columns([1, 1])

            with table_col:
                st.markdown("### Logistic Regression")
                st.dataframe(
                    lr_influence.head(5),
                    use_container_width=True,
                    hide_index=True
                )

                st.markdown("### Random Forest")
                st.dataframe(
                    rf_influence.head(5),
                    use_container_width=True,
                    hide_index=True
                )

                st.markdown("### XGBoost")
                st.dataframe(
                    xgb_influence.head(5),
                    use_container_width=True,
                    hide_index=True
                )

                st.markdown("### Artificial Neural Network")
                st.dataframe(
                    ann_influence.head(5),
                    use_container_width=True,
                    hide_index=True
                )

            with st.expander("Show all feature influences"):

                table_col, spacer_col = st.columns([1, 1])

                with table_col:

                    tab1, tab2, tab3, tab4 = st.tabs([
                        "Logistic Regression",
                        "Random Forest",
                        "XGBoost",
                        "ANN"
                    ])

                    with tab1:
                        st.table(
                            lr_influence
                        )

                    with tab2:
                        st.table(
                            rf_influence
                        )

                    with tab3:
                        st.table(
                            xgb_influence
                        )

                    with tab4:
                        st.table(
                            ann_influence
                        )

        except Exception as e:
            st.error(
                "SHAP influence section could not be loaded."
            )
            st.exception(e)

# ============================================================
# Footer
# ============================================================

st.divider()

st.markdown(
    """
    <div style='text-align:center; color:gray;'>
        <p><b>DATA 690 Capstone Project</b></p>
        <p>Predictive Modeling of Elevated Diabetes Risk Using Health Indicators</p>
        <p>Shankar Mohankumar | University of Maryland Global Campus</p>
        <p>Models: Logistic Regression | Random Forest | XGBoost | Artificial Neural Network</p>
    </div>
    """,
    unsafe_allow_html=True
)