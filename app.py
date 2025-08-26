import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

# Define mappings at the top
region_map = {"North": 0, "South": 1, "East": 2, "West": 3, "Central": 4}
sex_map = {"M": 0, "F": 1}

st.set_page_config(page_title="Clinical Obesity Prediction Dashboard", layout="wide")
st.title("🏥 Clinical Obesity Prediction Dashboard")
st.markdown("---")

# Load the trained model
@st.cache_resource
def load_model():
    try:
        with open('best_model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Model file 'best_model.pkl' not found. Please ensure the model is trained first.")
        return None

model = load_model()

if model is None:
    st.stop()

# Function to calculate BMI
def calculate_bmi(height_cm, weight_kg):
    if height_cm > 0 and weight_kg > 0:
        return weight_kg / ((height_cm / 100) ** 2)
    return None

# Function to determine clinical obesity label
def determine_clinical_obesity(bmi, adl_flag, organ_flag):
    if pd.isna(bmi) or bmi <= 0:
        return "Invalid BMI"
    
    if bmi >= 30 and (adl_flag == 1 or organ_flag == 1):
        return 2  # Clinical Obesity
    elif 25 <= bmi < 30 and adl_flag == 0 and organ_flag == 0:
        return 1  # Preclinical Obesity
    elif bmi < 25:
        return 0  # Normal
    else:
        return 1  # Preclinical Obesity (fallback)

# Function to prepare features for prediction
def prepare_features(height_cm, weight_kg, adl_flag, organ_flag, age=None, sex=None, region=None):
    bmi = calculate_bmi(height_cm, weight_kg)
    
    features = {
        'BMI': bmi,
        'adl_limitation_flag': adl_flag,
        'organ_dysfunction_flag': organ_flag
    }
    
    if age is not None:
        features['age'] = age
    if sex is not None:
        features['sex'] = 0 if sex == 'M' else 1
    if region is not None:
        features['region'] = region
    
    return features, bmi

# Sidebar for input method selection
st.sidebar.header("📊 Input Method")
input_method = st.sidebar.radio(
    "Choose input method:",
    ["Manual Entry", "File Upload"]
)

if input_method == "Manual Entry":
    st.header("📝 Manual Data Entry")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Physical Measurements")
        height_cm = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.1)
        weight_kg = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0, value=70.0, step=0.1)
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=30)
    
    with col2:
        st.subheader("Health Indicators")
        adl_flag = st.selectbox("ADL Limitation", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        organ_flag = st.selectbox("Organ Dysfunction", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        sex = st.selectbox("Sex", ["M", "F"])
        region = st.selectbox("Region", ["North", "South", "East", "West", "Central"])
    
    # Calculate BMI and prepare features
    features, bmi = prepare_features(height_cm, weight_kg, adl_flag, organ_flag, age, sex, region)
    
    if st.button("🔍 Predict Clinical Obesity", type="primary"):
        if bmi is not None:
            # Map region and sex to numeric codes
            features['region'] = region_map.get(region, -1)
            features['sex'] = sex_map.get(sex, -1)
            # Create DataFrame for prediction
            df_pred = pd.DataFrame([features])
            
            # Ensure all features are numeric
            df_pred = df_pred.apply(pd.to_numeric, errors='coerce')
            
            # Make prediction
            prediction = model.predict(df_pred)[0]
            clinical_label = determine_clinical_obesity(bmi, adl_flag, organ_flag)
            
            # Display results
            st.success("✅ Prediction Complete!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("BMI", f"{bmi:.2f}")
                if bmi < 18.5:
                    st.info("Underweight")
                elif bmi < 25:
                    st.success("Normal weight")
                elif bmi < 30:
                    st.warning("Overweight")
                else:
                    st.error("Obese")
            
            with col2:
                st.metric("Clinical Obesity Label", clinical_label)
                if clinical_label == 0:
                    st.success("Normal")
                elif clinical_label == 1:
                    st.warning("Preclinical Obesity")
                else:
                    st.error("Clinical Obesity")
            
            with col3:
                st.metric("Model Prediction", prediction)
                if prediction == 0:
                    st.success("Normal")
                elif prediction == 1:
                    st.warning("Preclinical Obesity")
                else:
                    st.error("Clinical Obesity")
            
            # Create visualization
            fig = go.Figure()
            fig.add_trace(go.Indicator(
                mode="gauge+number+delta",
                value=bmi,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "BMI Gauge"},
                delta={'reference': 25},
                gauge={
                    'axis': {'range': [None, 40]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 18.5], 'color': "lightgray"},
                        {'range': [18.5, 25], 'color': "green"},
                        {'range': [25, 30], 'color': "yellow"},
                        {'range': [30, 40], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 30
                    }
                }
            ))
            st.plotly_chart(fig, use_container_width=True)

else:  # File Upload
    st.header("📁 File Upload")
    
    uploaded_file = st.file_uploader(
        "Upload CSV file with columns: height_cm, weight_kg, adl_limitation_flag, organ_dysfunction_flag, age, sex, region",
        type=["csv"]
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
            
            # Display original data
            st.subheader("📋 Original Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Check required columns
            required_cols = ['height_cm', 'weight_kg', 'adl_limitation_flag', 'organ_dysfunction_flag']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {missing_cols}")
            else:
                # Calculate BMI
                df['BMI'] = df.apply(lambda row: calculate_bmi(row['height_cm'], row['weight_kg']), axis=1)
                
                # Prepare features for prediction
                features_cols = ['BMI', 'adl_limitation_flag', 'organ_dysfunction_flag']
                optional_cols = ['age', 'sex', 'region']
                
                for col in optional_cols:
                    if col in df.columns:
                        features_cols.append(col)
                
                X = df[features_cols].copy()
                
                # Encode categorical features
                if isinstance(X, pd.DataFrame):
                    if 'sex' in X.columns:
                        X['sex'] = X['sex'].map(lambda x: sex_map.get(x, -1)).fillna(-1).astype(int)
                    if 'region' in X.columns:
                        X['region'] = X['region'].astype('category').cat.codes
                
                # Ensure all features are numeric
                X = X.apply(pd.to_numeric, errors='coerce')
                
                # Make predictions
                predictions = model.predict(X)
                df['predicted_label'] = predictions
                
                # Calculate clinical obesity labels
                df['clinical_obesity_label'] = df.apply(
                    lambda row: determine_clinical_obesity(
                        row['BMI'], 
                        row['adl_limitation_flag'], 
                        row['organ_dysfunction_flag']
                    ), axis=1
                )
                
                # Display results
                st.subheader("🎯 Prediction Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Label Distribution:**")
                    label_counts = df['predicted_label'].value_counts().sort_index()
                    st.bar_chart(label_counts)
                
                with col2:
                    st.write("**BMI Distribution:**")
                    bmi_vals = df['BMI'].dropna()
                    if len(bmi_vals) > 0:
                        hist = np.histogram(bmi_vals, bins=10)
                        st.bar_chart(hist[0])
                
                # Display detailed results
                st.subheader("📊 Detailed Results")
                st.dataframe(df, use_container_width=True)
                
                # Download options
                col1, col2 = st.columns(2)
                
                with col1:
                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions (CSV)",
                        data=csv_data,
                        file_name="clinical_obesity_predictions.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    # Save for Tableau
                    df.to_csv("final_results.csv", index=False)
                    st.success("💾 Results saved as 'final_results.csv' for Tableau")
                
                # Summary statistics
                st.subheader("📈 Summary Statistics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Records", len(df))
                    st.metric("Valid BMI Records", df['BMI'].notna().sum())
                
                with col2:
                    st.metric("Normal (0)", (df['predicted_label'] == 0).sum())
                    st.metric("Preclinical (1)", (df['predicted_label'] == 1).sum())
                
                with col3:
                    st.metric("Clinical (2)", (df['predicted_label'] == 2).sum())
                    st.metric("Avg BMI", f"{df['BMI'].mean():.2f}")
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}") 

# Footer: Show ML model comparison with visualizations
st.markdown("---")
st.header("🤖 Model Comparison on This Dataset")
try:
    # Try to load model comparison table if it exists
    comparison_df = pd.read_csv('model_comparison.csv')
    st.dataframe(comparison_df, use_container_width=True)

    # Plotly bar charts for each metric
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
    for metric in metrics:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=comparison_df['Model'],
            y=comparison_df[metric],
            marker_color=["#FFD700" if v == comparison_df[metric].max() else "#1f77b4" for v in comparison_df[metric]],
            text=[f"{v:.2f}" for v in comparison_df[metric]],
            textposition='auto',
        ))
        fig.update_layout(title=f"{metric} Comparison", yaxis_title=metric, xaxis_title="Model", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
except Exception:
    st.info("Model comparison table not found. Please rerun the training script to generate 'model_comparison.csv' with metrics for KNN, Logistic Regression, Random Forest, and XGBoost.") 