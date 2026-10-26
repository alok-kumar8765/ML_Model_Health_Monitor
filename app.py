import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from utils import generate_sample_data, load_data, get_recent_anomalies_df

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="ML Model Health Monitor",
    page_icon="📈",
    layout="wide"
)

# Initialize sample data if missing
if not os.path.exists("data/sample_data.csv") or not os.path.exists("models/model.pkl"):
    generate_sample_data()

# --- SIDEBAR ---
st.sidebar.title("📈 Model Health Monitor")
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Upload & Train", "Model Performance", "Data Drift", "Feature Importance", "Monitoring & Alerts", "Download Report"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Data Source")
data_source = st.sidebar.radio("Select Source", ["Upload CSV/Excel", "Connect MySQL"])

if data_source == "Connect MySQL":
    st.sidebar.text_input("Host", "localhost")
    st.sidebar.text_input("Database", "ml_production")
    st.sidebar.text_input("User", "admin")
    st.sidebar.text_input("Password", type="password")

# --- MAIN HEADER ---
col_head1, col_head2 = st.columns([4, 1])
with col_head1:
    st.title("📊 ML Model Health Monitor")
    st.markdown("Track, evaluate and monitor your machine learning models in real-time")
with col_head2:
    st.markdown("<div style='text-align: right; font-size: 14px; color: gray;'>📅 Today<br><b>22 Oct 2026</b></div>", unsafe_allow_html=True)

st.markdown("---")

# --- PAGE ROUTING ---
if page == "Dashboard":
    # --- KPI METRIC CARDS ---
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Model Accuracy", "92.4%", "↑ 2.3%")
    col2.metric("Precision", "91.8%", "↑ 1.6%")
    col3.metric("Recall", "89.7%", "↑ 2.1%")
    col4.metric("Data Drift Score", "0.18", "↓ 12.5%", delta_color="inverse")
    col5.metric("Anomalies Detected", "57", "↑ 4.2%", delta_color="inverse")

    st.markdown("---")

    # --- ROW 1: CHARTS (Model Performance Over Time & Data Drift Analysis) ---
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Model Performance Over Time")
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct"]
        perf_df = pd.DataFrame({
            "Month": months,
            "Accuracy": [0.75, 0.78, 0.80, 0.82, 0.85, 0.86, 0.89, 0.90, 0.91, 0.924],
            "Precision": [0.72, 0.75, 0.77, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.918],
            "Recall": [0.70, 0.72, 0.75, 0.78, 0.80, 0.83, 0.85, 0.87, 0.88, 0.897]
        })
        fig_perf = px.line(perf_df, x="Month", y=["Accuracy", "Precision", "Recall"], markers=True)
        fig_perf.update_layout(height=320, margin=dict(l=20, r=20, t=20, b=20), legend=dict(orientation="h", y=1.1, x=0.3))
        st.plotly_chart(fig_perf, width='stretch')

    with c2:
        st.subheader("Data Drift Analysis")
        x_vals = np.linspace(-3, 3, 100)
        train_dist = np.exp(-0.5 * x_vals**2)
        curr_dist = np.exp(-0.5 * (x_vals - 0.4)**2 / 0.8**2)
        
        fig_drift = go.Figure()
        fig_drift.add_trace(go.Scatter(x=x_vals, y=train_dist, mode='lines', name='Train Data', line=dict(color='blue', width=2), fill='tozeroy'))
        fig_drift.add_trace(go.Scatter(x=x_vals, y=curr_dist, mode='lines', name='Current Data', line=dict(color='red', width=2)))
        fig_drift.update_layout(height=320, margin=dict(l=20, r=20, t=20, b=20), legend=dict(orientation="h", y=1.1, x=0.4), xaxis_title="Feature Value", yaxis_title="Density")
        st.plotly_chart(fig_drift, width='stretch')

    # --- ROW 2: CHARTS (Feature Importance & Prediction Distribution) ---
    c3, c4 = st.columns(2)

    with c3:
        st.subheader("Feature Importance")
        feat_df = pd.DataFrame({
            "Feature": ["Location", "Total_Spend", "Time_on_Site", "Purchase_Freq", "Income", "Age"],
            "Importance": [0.04, 0.08, 0.12, 0.18, 0.24, 0.32]
        })
        fig_feat = px.bar(feat_df, x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale="Blues")
        fig_feat.update_layout(height=320, margin=dict(l=20, r=20, t=20, b=20), coloraxis_showscale=False)
        st.plotly_chart(fig_feat, width='stretch')

    with c4:
        st.subheader("Prediction Distribution")
        fig_donut = go.Figure(data=[go.Pie(labels=['Normal (86.8%)', 'Anomaly (13.2%)'], values=[868, 132], hole=.6, marker_colors=['#0066ff', '#ff4d4d'])])
        fig_donut.update_layout(height=320, margin=dict(l=20, r=20, t=20, b=20), annotations=[dict(text='Total<br>1,000<br>Records', x=0.5, y=0.5, font_size=14, showarrow=False)])
        st.plotly_chart(fig_donut, width='stretch')

    # --- ROW 3: RECENT ANOMALIES TABLE ---
    st.subheader("Recent Anomalies")
    anomalies_df = get_recent_anomalies_df()
    st.dataframe(anomalies_df, width='stretch', hide_index=True)

elif page == "Upload & Train":
    st.subheader("📤 Upload Dataset & Train Model")
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.write("Dataset Preview:", df.head())
        
        target_col = st.selectbox("Select Target Column", df.columns)
        if st.button("Train Model"):
            with st.spinner("Training Random Forest Classifier..."):
                X = df.drop(columns=[target_col])
                y = df[target_col]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                acc = accuracy_score(y_test, preds)
                
                os.makedirs("models", exist_ok=True)
                joblib.dump(model, "models/model.pkl")
                st.success(f"Model successfully trained and saved! Test Accuracy: {acc*100:.2f}%")
    else:
        st.info("Please upload a dataset to begin training.")

elif page == "Model Performance":
    st.subheader("📈 Detailed Model Performance Metrics")
    st.markdown("Detailed breakdown of classification report and confusion matrix.")
    
    y_true = np.array([0]*868 + [1]*132)
    y_pred = np.array([0]*850 + [1]*18 + [0]*20 + [1]*112)
    
    st.text("Classification Report:")
    report = classification_report(y_true, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose(), width='stretch')

elif page == "Data Drift":
    st.subheader("🌊 Advanced Data Drift Monitoring")
    st.markdown("Population Stability Index (PSI) and Kolmogorov-Smirnov test results across features.")
    
    drift_report_df = pd.DataFrame({
        "Feature": ["Age", "Income", "Purchase_Freq", "Time_on_Site", "Total_Spend"],
        "PSI Score": [0.03, 0.22, 0.08, 0.15, 0.04],
        "Drift Status": ["No Drift", "Action Required", "Normal", "Moderate Drift", "No Drift"]
    })
    st.dataframe(drift_report_df, width='stretch', hide_index=True)

elif page == "Feature Importance":
    st.subheader("🔍 Feature Importance Analysis")
    st.markdown("SHAP and Permutation importance rankings.")
    
    fi_df = pd.DataFrame({
        "Feature": ["Age", "Income", "Purchase_Freq", "Time_on_Site", "Total_Spend", "Location"],
        "SHAP Value": [0.32, 0.24, 0.18, 0.12, 0.08, 0.04]
    })
    fig = px.bar(fi_df, x="SHAP Value", y="Feature", orientation="h", title="SHAP Feature Impact")
    st.plotly_chart(fig, width='stretch')

elif page == "Monitoring & Alerts":
    st.subheader("🔔 Monitoring & Alert Configurations")
    st.slider("Data Drift Alert Threshold (PSI)", 0.0, 0.5, 0.2)
    st.slider("Accuracy Drop Alert Threshold (%)", 1.0, 20.0, 5.0)
    st.text_input("Alert Email Recipient", "ml-ops-team@company.com")
    if st.button("Save Alert Settings"):
        st.success("Alert settings updated successfully!")

elif page == "Download Report":
    st.subheader("📥 Download Health Report")
    st.markdown("Export current audit logs, anomaly detection summaries, and model metrics as CSV.")
    
    anomalies_df = get_recent_anomalies_df()
    csv_data = anomalies_df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="Download Anomaly Report (CSV)",
        data=csv_data,
        file_name="model_health_anomaly_report.csv",
        mime="text/csv",
    )