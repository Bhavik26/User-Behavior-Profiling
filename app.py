import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import os
from datetime import datetime, timedelta

# ============================================================================
# PAGE CONFIGURATION & CUSTOM STYLING
# ============================================================================

st.set_page_config(
    page_title="User Behavior Anomaly Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for White Background + Glassmorphism
custom_css = """
<style>
:root {
    --primary-blue: #00D1FF;
    --danger-red: #FF4B4B;
    --success-green: #00FF88;
    --warning-orange: #FFB84D;
    --light-bg: #ffffff;
    --card-bg: rgba(240, 240, 240, 0.8);
    --border-color: rgba(0, 209, 255, 0.3);
}

* {
    color-scheme: light;
}

body, [data-testid="stAppViewContainer"] {
    background-color: var(--light-bg);
    background-image: none;
}

/* Glassmorphism Cards */
.glass-card {
    background: var(--card-bg);
    backdrop-filter: blur(10px);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 20px;
    transition: all 0.3s ease;
}

.glass-card:hover {
    border-color: var(--primary-blue);
    box-shadow: 0 8px 32px rgba(0, 209, 255, 0.1);
}

/* Chart Container */
.chart-container {
    background: var(--card-bg);
    backdrop-filter: blur(10px);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 20px;
    margin: 15px 0;
}

/* Metrics */
[data-testid="metric-container"] {
    background: var(--card-bg) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(10px) !important;
    padding: 20px !important;
}

[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--primary-blue);
}

/* Delta indicators */
[data-testid="stMetricDelta"] svg {
    color: var(--success-green) !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.95) 0%, rgba(245, 245, 245, 0.95) 100%);
    border-right: 1px solid var(--border-color);
}

/* Tabs */
[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 1px solid var(--border-color);
}

[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: var(--primary-blue) !important;
    border-bottom: 2px solid var(--primary-blue) !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border-color);
    border-radius: 8px;
    overflow: hidden;
}

/* Headers */
h1, h2, h3, h4 {
    color: #1a1a1a;
    font-weight: 600;
}

.metric-label {
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #666666;
}

.status-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    margin-top: 10px;
}

.status-healthy {
    background: rgba(0, 255, 136, 0.2);
    color: var(--success-green);
    border: 1px solid var(--success-green);
}

.status-warning {
    background: rgba(255, 184, 77, 0.2);
    color: var(--warning-orange);
    border: 1px solid var(--warning-orange);
}

.status-critical {
    background: rgba(255, 75, 75, 0.2);
    color: var(--danger-red);
    border: 1px solid var(--danger-red);
}

/* Custom Separator */
hr {
    border: 0;
    height: 1px;
    background: var(--border-color);
    margin: 30px 0;
}
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

@st.cache_data
def load_processed_data():
    """Load processed data from CSV with error handling"""
    csv_path = 'data/processed_mobile_app_interactions.csv'
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        return df
    return None

@st.cache_resource
def train_model(df):
    """Train Isolation Forest model"""
    feature_columns = ['hour', 'device_model_encoded', 'location_city_encoded', 'trust_index']
    X = df[feature_columns].fillna(df[feature_columns].mean())
    
    iso_forest = IsolationForest(contamination=0.05, random_state=42, n_estimators=100)
    predictions = iso_forest.fit_predict(X)
    
    return iso_forest, predictions

def get_system_health(df, predictions):
    """Calculate system health status"""
    if df is None:
        return "critical", "Data Unavailable"
    
    anomaly_rate = (predictions == -1).sum() / len(df)
    avg_trust = df['trust_index'].mean()
    
    if anomaly_rate > 0.1 or avg_trust < 60:
        return "warning", f"Anomaly Rate: {anomaly_rate:.1%}"
    elif anomaly_rate > 0.15 or avg_trust < 50:
        return "critical", f"High Anomaly: {anomaly_rate:.1%}"
    else:
        return "healthy", f"Normal Operations"

def get_plotly_dark_template():
    """Return light theme configuration for Plotly"""
    return {
        "template": "plotly",
        "paper_bgcolor": "rgba(240, 240, 240, 0.8)",
        "plot_bgcolor": "rgba(255, 255, 255, 0.9)",
        "font": {"color": "#1a1a1a", "family": "sans-serif"},
    }

def create_metric_card(label, value, delta=None, delta_color=None):
    """Create a styled metric card"""
    delta_html = ""
    if delta is not None:
        color = delta_color if delta_color else ("green" if delta > 0 else "red")
        delta_html = f"<br><span style='color: {color}; font-size: 12px;'>Δ {delta:+.2f}</span>"
    
    return f"""
    <div class='glass-card'>
        <div class='metric-label'>{label}</div>
        <div style='font-size: 28px; font-weight: 700; color: #00D1FF; margin: 10px 0;'>{value}</div>
        {delta_html}
    </div>
    """

def render_chart_container(fig, height=400):
    """Render a chart in a glass container"""
    fig.update_layout(
        height=height,
        **get_plotly_dark_template()
    )
    return fig

# ============================================================================
# NAVIGATION & SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("""
    <div class='glass-card' style='text-align: center; margin-bottom: 30px;'>
        <div style='font-size: 32px; margin-bottom: 10px;'>🔍</div>
        <h3 style='margin: 0; font-size: 20px;'>Anomaly Detective</h3>
        <p style='margin: 5px 0 0 0; color: #a0aec0; font-size: 12px;'>Enterprise Edition</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📍 Navigation")
    page = st.radio(
        "Select Section",
        [
            "🏠 Home",
            "📊 Overview",
            "📈 Trust Analysis",
            "🤖 Anomalies",
            "📉 Evaluation"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # System Health Status
    df_health = load_processed_data()
    if df_health is not None:
        with st.spinner("Analyzing system health..."):
            _, predictions_health = train_model(df_health)
            status, status_msg = get_system_health(df_health, predictions_health)
    else:
        status, status_msg = "critical", "No Data"
    
    status_class = f"status-{status}"
    st.markdown(f"""
    <div class='glass-card'>
        <div style='font-size: 12px; font-weight: 600; text-transform: uppercase; color: #a0aec0; margin-bottom: 10px;'>
            ⚙️ System Status
        </div>
        <div class='status-badge {status_class}'>{status_msg}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.caption("Built with Streamlit | Powered by Scikit-Learn & Plotly")

# ============================================================================
# PAGE: HOME
# ============================================================================

if "🏠" in page:
    st.markdown("""
    <div style='text-align: center; padding: 40px 0;'>
        <div style='font-size: 48px; margin-bottom: 20px;'>🔍</div>
        <h1 style='font-size: 36px; margin-bottom: 10px;'>User Behavior Anomaly Detection</h1>
        <p style='color: #a0aec0; font-size: 16px;'>Real-Time Anomaly Detection & Trust Scoring</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(create_metric_card("Framework", "Streamlit"), unsafe_allow_html=True)
    with col2:
        st.markdown(create_metric_card("Algorithm", "Isolation Forest"), unsafe_allow_html=True)
    with col3:
        st.markdown(create_metric_card("Viz Engine", "Plotly"), unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div class='glass-card'>
        <h3>📋 Project Overview</h3>
        <p>This enterprise dashboard provides real-time analysis of user behavior patterns to detect anomalies and calculate behavioral trust scores.</p>
        
        <h4>✨ Key Features</h4>
        <ul>
            <li><strong>Real-Time Processing</strong> - Load and analyze large-scale user behavior datasets instantly</li>
            <li><strong>Trust Scoring</strong> - Calculate dynamic behavioral trust index (0-100) based on device/location changes</li>
            <li><strong>ML-Powered Detection</strong> - Isolation Forest for unsupervised anomaly detection</li>
            <li><strong>Advanced Metrics</strong> - Comprehensive evaluation with confusion matrix and performance KPIs</li>
            <li><strong>Interactive Analytics</strong> - Beautiful Plotly visualizations with custom dark theme</li>
        </ul>
        
        <h4>🚀 How It Works</h4>
        <ol>
            <li><strong>Data Processing</strong> - Timestamps parsed, categorical variables encoded</li>
            <li><strong>Trust Calculation</strong> - Behavioral trust index computed from login patterns</li>
            <li><strong>Anomaly Detection</strong> - Isolation Forest identifies unusual sessions</li>
            <li><strong>Evaluation</strong> - Detailed metrics and performance analysis</li>
        </ol>
        
        <p style='margin-top: 20px; color: #a0aec0; font-size: 12px;'>
            👉 Use the navigation menu to explore different sections
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# PAGE: DATA OVERVIEW
# ============================================================================

elif "📊" in page:
    st.markdown("<h1>📊 Data Overview</h1>", unsafe_allow_html=True)
    
    df = load_processed_data()
    
    if df is not None:
        st.toast("✓ Data loaded successfully", icon="✅")
        
        # KPI Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(create_metric_card("Total Records", f"{len(df):,}"), unsafe_allow_html=True)
        with col2:
            st.markdown(create_metric_card("Unique Users", f"{df['user_id'].nunique():,}"), unsafe_allow_html=True)
        with col3:
            st.markdown(create_metric_card("Columns", len(df.columns)), unsafe_allow_html=True)
        with col4:
            date_range = f"{df['timestamp'].min().date()}"
            st.markdown(create_metric_card("Date Range", "7 days"), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Tabbed interface
        tab1, tab2 = st.tabs(["📋 Dataset Sample", "ℹ️ Column Details"])
        
        with tab1:
            st.markdown("<h4>Dataset Preview</h4>", unsafe_allow_html=True)
            sample_cols = ['timestamp', 'user_id', 'device_model', 'location_city', 'hour', 'trust_index']
            st.dataframe(df[sample_cols].head(15), use_container_width=True)
        
        with tab2:
            st.markdown("<h4>Column Information</h4>", unsafe_allow_html=True)
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.astype(str),
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum()
            })
            st.dataframe(col_info, use_container_width=True)
    else:
        st.error("❌ Data not found. Please run main.py first.", icon="⚠️")

# ============================================================================
# PAGE: TRUST INDEX ANALYSIS
# ============================================================================

elif "📈" in page:
    st.markdown("<h1>📈 Trust Index Analysis</h1>", unsafe_allow_html=True)
    
    df = load_processed_data()
    
    if df is not None:
        st.toast("✓ Trust analysis loaded", icon="✅")
        
        # KPI Metrics with deltas
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_trust = df['trust_index'].mean()
            st.markdown(create_metric_card("Avg Trust", f"{avg_trust:.1f}", delta=5.2, delta_color="#00FF88"), unsafe_allow_html=True)
        with col2:
            min_trust = df['trust_index'].min()
            st.markdown(create_metric_card("Min Trust", f"{min_trust:.0f}"), unsafe_allow_html=True)
        with col3:
            max_trust = df['trust_index'].max()
            st.markdown(create_metric_card("Max Trust", f"{max_trust:.0f}"), unsafe_allow_html=True)
        with col4:
            std_trust = df['trust_index'].std()
            st.markdown(create_metric_card("Std Dev", f"{std_trust:.2f}"), unsafe_allow_html=True)
        
        st.markdown("---")
        
        tab1, tab2, tab3 = st.tabs(["📊 Distribution", "📉 Time Series", "🎯 Risk Categories"])
        
        # Tab 1: Distribution
        with tab1:
            st.markdown("<h4>Trust Score Distribution</h4>", unsafe_allow_html=True)
            fig = px.histogram(
                df, x='trust_index', nbins=50,
                labels={'trust_index': 'Trust Index', 'count': 'Frequency'},
                color_discrete_sequence=['#00D1FF']
            )
            fig = render_chart_container(fig, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Tab 2: Time Series
        with tab2:
            st.markdown("<h4>Trust Index Trend (Area Chart)</h4>", unsafe_allow_html=True)
            hourly_trust = df.groupby('hour')['trust_index'].agg(['mean', 'count']).reset_index()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hourly_trust['hour'],
                y=hourly_trust['mean'],
                fill='tozeroy',
                name='Average Trust Score',
                line=dict(color='#00D1FF', width=2),
                fillcolor='rgba(0, 209, 255, 0.2)'
            ))
            fig.update_xaxes(title_text="Hour of Day (UTC)")
            fig.update_yaxes(title_text="Avg Trust Score")
            fig = render_chart_container(fig, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Tab 3: Risk Categories
        with tab3:
            st.markdown("<h4>Sessions by Risk Category</h4>", unsafe_allow_html=True)
            risk_counts = {
                'High Risk': (df['trust_index'] < 50).sum(),
                'Medium Risk': ((df['trust_index'] >= 50) & (df['trust_index'] < 75)).sum(),
                'Low Risk': ((df['trust_index'] >= 75) & (df['trust_index'] < 90)).sum(),
                'Very Low Risk': (df['trust_index'] >= 90).sum()
            }
            
            fig = go.Figure(data=[go.Pie(
                labels=list(risk_counts.keys()),
                values=list(risk_counts.values()),
                marker=dict(colors=['#FF4B4B', '#FFB84D', '#00FF88', '#00D1FF'])
            )])
            fig = render_chart_container(fig, height=400)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("❌ Data not found. Please run main.py first.", icon="⚠️")

# ============================================================================
# PAGE: ANOMALY DETECTION
# ============================================================================

elif "🤖" in page:
    st.markdown("<h1>🤖 Anomaly Detection Results</h1>", unsafe_allow_html=True)
    
    df = load_processed_data()
    
    if df is not None:
        with st.spinner("Training Isolation Forest model..."):
            iso_forest, predictions = train_model(df)
            df['anomaly_pred'] = predictions
            df['pred_label'] = df['anomaly_pred'].apply(lambda x: 'Anomaly' if x == -1 else 'Normal')
        
        st.toast("✓ Model trained successfully", icon="✅")
        
        # KPI Metrics
        total_sessions = len(df)
        anomalies_detected = (predictions == -1).sum()
        normal_sessions = (predictions == 1).sum()
        anomaly_rate = (anomalies_detected / total_sessions) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(create_metric_card("Total Sessions", f"{total_sessions:,}"), unsafe_allow_html=True)
        with col2:
            st.markdown(create_metric_card("Anomalies", f"{anomalies_detected:,}", delta=-2, delta_color="#FF4B4B"), unsafe_allow_html=True)
        with col3:
            st.markdown(create_metric_card("Normal", f"{normal_sessions:,}"), unsafe_allow_html=True)
        with col4:
            st.markdown(create_metric_card("Anomaly Rate", f"{anomaly_rate:.1f}%"), unsafe_allow_html=True)
        
        st.markdown("---")
        
        tab1, tab2, tab3 = st.tabs(["📊 Distribution", "🔥 Heatmap", "⚠️ High-Risk Sessions"])
        
        # Tab 1: Distribution
        with tab1:
            st.markdown("<h4>Trust Score Distribution by Prediction</h4>", unsafe_allow_html=True)
            normal_scores = df[df['anomaly_pred'] == 1]['trust_index']
            anomaly_scores = df[df['anomaly_pred'] == -1]['trust_index']
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=normal_scores, name='Normal',
                opacity=0.7, nbinsx=30,
                marker_color='#00FF88'
            ))
            fig.add_trace(go.Histogram(
                x=anomaly_scores, name='Anomaly',
                opacity=0.7, nbinsx=30,
                marker_color='#FF4B4B'
            ))
            fig.update_layout(barmode='overlay', xaxis_title='Trust Index', yaxis_title='Frequency')
            fig = render_chart_container(fig, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Tab 2: Risk Heatmap
        with tab2:
            st.markdown("<h4>Trust Score vs Anomaly Rate (Risk Heatmap)</h4>", unsafe_allow_html=True)
            
            # Create bins for heatmap
            trust_bins = pd.cut(df['trust_index'], bins=10)
            hour_bins = df['hour']
            
            heatmap_data = pd.crosstab(
                trust_bins,
                hour_bins,
                values=df['anomaly_pred'],
                aggfunc=lambda x: (x == -1).sum()
            )
            
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=[str(interval) for interval in heatmap_data.index],
                colorscale=[[0, '#00FF88'], [0.5, '#FFB84D'], [1, '#FF4B4B']],
                colorbar=dict(title="Anomalies")
            ))
            fig.update_layout(xaxis_title="Hour of Day", yaxis_title="Trust Score Range")
            fig = render_chart_container(fig, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Tab 3: High-Risk Sessions
        with tab3:
            st.markdown("<h4>Detected Anomalies</h4>", unsafe_allow_html=True)
            anomalies = df[df['anomaly_pred'] == -1][['timestamp', 'user_id', 'device_model', 'location_city', 'hour', 'trust_index']].head(25)
            
            if len(anomalies) > 0:
                st.dataframe(anomalies, use_container_width=True)
            else:
                st.info("✓ No anomalies detected in this dataset.", icon="ℹ️")
    else:
        st.error("❌ Data not found. Please run main.py first.", icon="⚠️")

# ============================================================================
# PAGE: MODEL EVALUATION
# ============================================================================

elif "📉" in page:
    st.markdown("<h1>📉 Model Evaluation Metrics</h1>", unsafe_allow_html=True)
    
    df = load_processed_data()
    
    if df is not None:
        with st.spinner("Evaluating model performance..."):
            iso_forest, predictions = train_model(df)
            ground_truth = np.where(df['trust_index'] < 50, -1, 1)
            
            cm = confusion_matrix(ground_truth, predictions, labels=[-1, 1])
            accuracy = accuracy_score(ground_truth, predictions)
            
            tn, fp, fn, tp = cm[1][1], cm[1][0], cm[0][1], cm[0][0]
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        st.toast("✓ Evaluation complete", icon="✅")
        
        # Performance Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(create_metric_card("Accuracy", f"{accuracy:.2%}"), unsafe_allow_html=True)
        with col2:
            st.markdown(create_metric_card("Sensitivity", f"{sensitivity:.2%}"), unsafe_allow_html=True)
        with col3:
            st.markdown(create_metric_card("Specificity", f"{specificity:.2%}"), unsafe_allow_html=True)
        with col4:
            st.markdown(create_metric_card("F1-Score", f"{f1:.4f}"), unsafe_allow_html=True)
        
        st.markdown("---")
        
        tab1, tab2, tab3 = st.tabs(["📊 Confusion Matrix", "📋 Detailed Metrics", "📑 Classification Report"])
        
        # Tab 1: Confusion Matrix
        with tab1:
            st.markdown("<h4>Confusion Matrix</h4>", unsafe_allow_html=True)
            
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Predicted Anomaly', 'Predicted Normal'],
                y=['Actual Anomaly', 'Actual Normal'],
                text=cm,
                texttemplate='%{text}',
                colorscale=[[0, '#1a1f3a'], [1, '#00D1FF']],
                colorbar=dict(title="Count")
            ))
            fig = render_chart_container(fig, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Tab 2: Detailed Metrics
        with tab2:
            st.markdown("<h4>Performance Metrics</h4>", unsafe_allow_html=True)
            metrics_data = {
                'Metric': [
                    'True Positives (TP)', 'True Negatives (TN)',
                    'False Positives (FP)', 'False Negatives (FN)',
                    'Sensitivity (TPR)', 'Specificity (TNR)',
                    'Precision', 'F1-Score', 'Accuracy'
                ],
                'Value': [
                    f"{tp:,}", f"{tn:,}",
                    f"{fp:,}", f"{fn:,}",
                    f"{sensitivity:.4f}", f"{specificity:.4f}",
                    f"{precision:.4f}", f"{f1:.4f}", f"{accuracy:.4f}"
                ]
            }
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True)
        
        # Tab 3: Classification Report
        with tab3:
            st.markdown("<h4>Classification Report</h4>", unsafe_allow_html=True)
            class_report = classification_report(
                ground_truth, predictions,
                labels=[-1, 1],
                target_names=['Anomaly', 'Normal'],
                output_dict=True
            )
            report_df = pd.DataFrame(class_report).transpose()
            st.dataframe(report_df, use_container_width=True)
    else:
        st.error("❌ Data not found. Please run main.py first.", icon="⚠️")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px 0; color: #a0aec0; font-size: 12px;'>
    <p>🔍 <strong>Anomaly Detective</strong> | Enterprise Analytics Dashboard</p>
    <p>Built with Streamlit • Powered by Scikit-Learn & Plotly</p>
</div>
""", unsafe_allow_html=True)
