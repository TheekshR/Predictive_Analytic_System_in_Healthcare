import streamlit as st
import pickle
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from datetime import datetime
import io
import matplotlib.pyplot as plt

# ===================== LOAD MODEL + SCALER =====================
with open('multi_output_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# ===================== PAGE CONFIG =====================
st.set_page_config(page_title="Health Prediction Dashboard", layout="wide")

st.markdown("""
    <style>
    .main-title {text-align: center; font-size: 2.2rem; color: #2E86C1; font-weight: 800; padding-bottom: 0.5rem;}
    .sub-text {text-align: center; font-size: 1rem; color: #5D6D7E; margin-bottom: 1rem;}
    .stTabs [data-baseweb="tab-list"] {justify-content: center;}
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🩺 Global Health Prediction Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-text'>Predict and visualize multiple health indicators including mortality, recovery, access, and hospital beds per 1000.</div>", unsafe_allow_html=True)

# ===================== LAYOUT =====================
col1, col2 = st.columns([1, 1])

# ---------------- INPUT SECTION ----------------
with col1:
    st.markdown("### 🧮 Input Parameters")
    
    year = st.number_input("📅 Year", min_value=1990, max_value=2030, value=2020)

    countries = ["USA", "India", "China", "UK", "Germany", "France", "Brazil", "Japan", "Australia", "Canada"]
    country = st.selectbox("🌍 Country", countries)
    country_mapping = {name: idx for idx, name in enumerate(countries)}
    country_encoded = country_mapping[country]

    disease_categories = ["Respiratory", "Genetic", "Parasitic", "Bacterial", "Viral", "Cardiovascular"]
    disease_cat = st.selectbox("🦠 Disease Category", disease_categories)
    disease_cat_mapping = {name: idx for idx, name in enumerate(disease_categories)}
    disease_cat_encoded = disease_cat_mapping[disease_cat]

    disease_name = st.text_input("🧫 Disease Name ", "COVID-19")
    disease_name_encoded = 0  # for simplicity

    treatment_types = ["Medication", "Surgery", "Vaccination", "Therapy"]
    treatment = st.selectbox("💊 Treatment Type", treatment_types)
    treatment_mapping = {name: idx for idx, name in enumerate(treatment_types)}
    treatment_encoded = treatment_mapping[treatment]

    doctors = st.number_input("👨‍⚕️ Doctors per 1000", min_value=0.0, value=2.0, step=0.1)
    income = st.number_input("💰 Per Capita Income (USD)", min_value=0, value=10000, step=100)
    education = st.slider("🎓 Education Index", 0.0, 1.0, 0.7, 0.01)
    urban = st.slider("🏙️ Urbanization Rate (%)", 0.0, 100.0, 60.0, 0.1)

    predict_btn = st.button("🔮 Predict Health Indicators", use_container_width=True)

# ---------------- OUTPUT SECTION ----------------
with col2:
    st.markdown("### 📊 Prediction Results & Visualizations")

    if predict_btn:
        input_data = np.array([[year, country_encoded, disease_cat_encoded, disease_name_encoded, treatment_encoded,
                                doctors, income, education, urban]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)

        mortality, recovery, access, beds = prediction[0]
        st.success("✅ Prediction Complete!")

        tab1, tab2, tab3, tab4 = st.tabs([
            "Mortality Rate (%)", "Recovery Rate (%)", "Healthcare Access (%)", "Hospital Beds per 1000"
        ])

        # Gauge chart function
        def gauge_chart(value, title, max_value=100):
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=value,
                title={'text': title, 'font': {'size': 18}},
                gauge={
                    'axis': {'range': [0, max_value]},
                    'bar': {'color': '#2E86C1'},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "#D6EAF8",
                    'steps': [
                        {'range': [0, max_value * 0.5], 'color': '#EBF5FB'},
                        {'range': [max_value * 0.5, max_value * 0.8], 'color': '#AED6F1'},
                        {'range': [max_value * 0.8, max_value], 'color': '#2E86C1'}
                    ],
                }
            ))
            fig.update_layout(height=300, margin=dict(l=40, r=40, t=40, b=40))
            buf = io.BytesIO()
            fig.write_image(buf, format='png')
            buf.seek(0)
            return fig, buf

        charts = {}
        charts['mortality'], buf_mort = gauge_chart(mortality, "Mortality Rate (%)")
        charts['recovery'], buf_rec = gauge_chart(recovery, "Recovery Rate (%)")
        charts['access'], buf_acc = gauge_chart(access, "Healthcare Access (%)")
        charts['beds'], buf_beds = gauge_chart(beds, "Hospital Beds per 1000", max_value=10)

        with tab1:
            st.plotly_chart(charts['mortality'], use_container_width=True)
            st.info(f"Predicted Mortality Rate: **{mortality:.2f}%**")
        with tab2:
            st.plotly_chart(charts['recovery'], use_container_width=True)
            st.info(f"Predicted Recovery Rate: **{recovery:.2f}%**")
        with tab3:
            st.plotly_chart(charts['access'], use_container_width=True)
            st.info(f"Predicted Healthcare Access: **{access:.2f}%**")
        with tab4:
            st.plotly_chart(charts['beds'], use_container_width=True)
            st.info(f"Predicted Beds per 1000: **{beds:.2f}**")

        # ---------------- FEATURE IMPORTANCE SECTION ----------------
        st.markdown("### 🧠 Feature Importance")

        try:
            feature_names = ["Year", "Country", "Disease Cat", "Disease Name", "Treatment",
                             "Doctors/1000", "Income", "Education", "Urbanization"]
            outputs = ["Mortality Rate", "Recovery Rate", "Healthcare Access", "Hospital Beds"]

            if hasattr(model, "estimators_"):  # MultiOutputRegressor
                fig, axes = plt.subplots(2, 2, figsize=(10, 8))
                fig.suptitle("Feature Importance for Each Output", fontsize=16)

                for i, ax in enumerate(axes.flatten()):
                    importances = model.estimators_[i].feature_importances_
                    sorted_idx = np.argsort(importances)
                    ax.barh(np.array(feature_names)[sorted_idx], importances[sorted_idx], color='#2E86C1')
                    ax.set_title(outputs[i])
                    ax.set_xlabel("Importance Score")

                plt.tight_layout(rect=[0, 0, 1, 0.96])
                st.pyplot(fig)

            elif hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                fig_imp = px.bar(
                    x=importances, y=feature_names, orientation='h',
                    title="Feature Importance", color=importances, color_continuous_scale="Blues"
                )
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.info("ℹ️ Feature importance visualization not available for this model type.")
        except Exception as e:
            st.error(f"Error displaying feature importances: {e}")

        # ---------------- RECOMMENDATIONS ----------------
        st.markdown("### 💡 Recommendations")
        if mortality > 30:
            st.error("🚨 High mortality detected! Increase **healthcare spending**, **medical training**, and improve **early diagnosis systems**.")
        elif recovery < 50:
            st.warning("⚠️ Low recovery rate. Focus on **therapy access**, **hospital infrastructure**, and **disease-specific medication research**.")
        elif access < 60:
            st.info("ℹ️ Moderate healthcare access. Consider expanding **rural health centers** and **digital health services**.")
        else:
            st.success("🌟 Overall good health indicators! Continue maintaining **investment in education and healthcare.**")

        # Store in session for PDF
        st.session_state['input_values'] = {
            'Year': year, 'Country': country, 'Disease Category': disease_cat,
            'Disease Name': disease_name, 'Treatment Type': treatment, 'Doctors per 1000': doctors,
            'Per Capita Income (USD)': income, 'Education Index': education, 'Urbanization Rate (%)': urban
        }
        st.session_state['predictions'] = {
            'Mortality Rate (%)': mortality, 'Recovery Rate (%)': recovery,
            'Healthcare Access (%)': access, 'Hospital Beds per 1000': beds
        }
        st.session_state['charts'] = [buf_mort, buf_rec, buf_acc, buf_beds]

# ---------------- PDF DOWNLOAD ----------------
if 'predictions' in st.session_state:
    def generate_pdf_with_charts():
        pdf_file = "Health_Report.pdf"
        c = canvas.Canvas(pdf_file, pagesize=A4)
        width, height = A4

        c.setFont("Helvetica-Bold", 18)
        c.drawString(150, 800, "Global Health Prediction Report")

        c.setFont("Helvetica", 12)
        c.drawString(50, 760, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        c.line(50, 755, 550, 755)

        c.drawString(50, 730, "Input Features:")
        y = 710
        for key, value in st.session_state['input_values'].items():
            if y < 100:
                c.showPage()
                y = 800
            c.drawString(70, y, f"{key}: {value}")
            y -= 18

        c.drawString(50, y - 10, "Predictions & Visualizations:")
        y -= 30

        charts_buf = st.session_state['charts']
        labels = list(st.session_state['predictions'].keys())
        for idx, key in enumerate(labels):
            if y - 180 < 50:
                c.showPage()
                y = 800
            c.drawString(70, y, f"{key}: {st.session_state['predictions'][key]:.2f}")
            y -= 18
            img = ImageReader(charts_buf[idx])
            c.drawImage(img, 70, y - 180, width=400, height=180)
            y -= 200

        c.save()
        return pdf_file

    pdf_file = generate_pdf_with_charts()
    with open(pdf_file, "rb") as f:
        st.download_button(
            label="📄 Download Report",
            data=f,
            file_name=pdf_file,
            mime="application/pdf"
        )
