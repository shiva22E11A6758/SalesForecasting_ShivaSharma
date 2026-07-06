import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# Page config
st.set_page_config(page_title="Demand Intelligence System", page_icon="📈", layout="wide")

# Custom CSS for professional UI
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6
    }
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        color: #1e3d59;
    }
    .kpi-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('datasets/train.csv')
    if 'Order Date' in df.columns:
        df['Order Date'] = pd.to_datetime(df['Order Date'], format='mixed', dayfirst=True)
        df['Year'] = df['Order Date'].dt.year
        df['Month'] = df['Order Date'].dt.to_period('M').astype(str)
    return df

@st.cache_data
def load_outputs():
    outputs = {}
    try:
        outputs['metrics'] = pd.read_csv('outputs/model_metrics.csv')
    except: outputs['metrics'] = None
    
    try:
        outputs['forecast'] = pd.read_csv('outputs/final_3m_forecast.csv')
    except: outputs['forecast'] = None
    
    try:
        outputs['segment_forecasts'] = pd.read_csv('outputs/segment_forecasts.csv')
    except: outputs['segment_forecasts'] = None
    
    try:
        outputs['anomalies'] = pd.read_csv('outputs/anomaly_report.csv')
    except: outputs['anomalies'] = None
    
    try:
        outputs['clusters'] = pd.read_csv('outputs/demand_segmentation.csv')
    except: outputs['clusters'] = None
    
    return outputs

df = load_data()
outputs = load_outputs()

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page:", 
                        ["Sales Dashboard", "Forecast Explorer", "Anomaly Report", "Demand Segmentation"])

st.sidebar.markdown("---")
st.sidebar.info("End-to-End Sales Forecasting & Demand Intelligence System - Internship Project")

# ==========================================
# PAGE 1: SALES DASHBOARD
# ==========================================
if page == "Sales Dashboard":
    st.markdown('<div class="main-header">📊 Interactive Sales Dashboard</div>', unsafe_allow_html=True)
    
    # Filters
    st.sidebar.subheader("Filters")
    filtered_df = df.copy()
    if 'Year' in df.columns:
        selected_year = st.sidebar.multiselect("Select Year", df['Year'].unique(), default=df['Year'].unique())
        filtered_df = filtered_df[filtered_df['Year'].isin(selected_year)]
    if 'Region' in df.columns:
        selected_region = st.sidebar.multiselect("Select Region", df['Region'].dropna().unique(), default=df['Region'].dropna().unique())
        filtered_df = filtered_df[filtered_df['Region'].isin(selected_region)]
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        val = f"${filtered_df['Sales'].sum():,.0f}" if 'Sales' in filtered_df.columns else "N/A"
        st.markdown(f"""<div class="kpi-card">
            <h3 style='color: gray; font-size: 1.2rem'>Total Sales</h3>
            <h2 style='color: #2c3e50'>{val}</h2>
            </div>""", unsafe_allow_html=True)
    with col2:
        if 'Sales' in filtered_df.columns and 'Order ID' in filtered_df.columns:
            aov = filtered_df['Sales'].sum() / filtered_df['Order ID'].nunique() if filtered_df['Order ID'].nunique() > 0 else 0
            val = f"${aov:,.2f}"
        else:
            val = "N/A"
        st.markdown(f"""<div class="kpi-card">
            <h3 style='color: gray; font-size: 1.2rem'>Avg Order Value</h3>
            <h2 style='color: #2c3e50'>{val}</h2>
            </div>""", unsafe_allow_html=True)
    with col3:
        val = f"{filtered_df['Order ID'].nunique()}" if 'Order ID' in filtered_df.columns else "N/A"
        st.markdown(f"""<div class="kpi-card">
            <h3 style='color: gray; font-size: 1.2rem'>Total Orders</h3>
            <h2 style='color: #2c3e50'>{val}</h2>
            </div>""", unsafe_allow_html=True)
    with col4:
        val = f"{filtered_df['Customer ID'].nunique()}" if 'Customer ID' in filtered_df.columns else "N/A"
        st.markdown(f"""<div class="kpi-card">
            <h3 style='color: gray; font-size: 1.2rem'>Unique Customers</h3>
            <h2 style='color: #2c3e50'>{val}</h2>
            </div>""", unsafe_allow_html=True)
            
    st.markdown("---")
    
    # Charts
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        if 'Month' in filtered_df.columns and 'Sales' in filtered_df.columns:
            monthly_trend = filtered_df.groupby('Month')['Sales'].sum().reset_index()
            fig1 = px.line(monthly_trend, x='Month', y='Sales', title='Monthly Sales Trend', markers=True)
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.warning("Insufficient data for Monthly Sales Trend.")
        
        if 'Region' in filtered_df.columns and 'Sales' in filtered_df.columns:
            region_sales = filtered_df.groupby('Region')['Sales'].sum().reset_index()
            fig3 = px.pie(region_sales, values='Sales', names='Region', title='Sales by Region', hole=0.4)
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.warning("Insufficient data for Region Sales.")
            
    with col_chart2:
        if 'Year' in filtered_df.columns and 'Sales' in filtered_df.columns:
            yearly_trend = filtered_df.groupby('Year')['Sales'].sum().reset_index()
            fig2 = px.bar(yearly_trend, x='Year', y='Sales', title='Yearly Sales Trend', color='Year')
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("Insufficient data for Yearly Sales Trend.")
        
        if 'Category' in filtered_df.columns and 'Sales' in filtered_df.columns:
            cat_sales = filtered_df.groupby('Category')['Sales'].sum().reset_index()
            fig4 = px.bar(cat_sales, x='Category', y='Sales', title='Sales by Category', color='Category')
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.warning("Insufficient data for Category Sales.")
        
    if 'Product Name' in filtered_df.columns and 'Sales' in filtered_df.columns:
        st.subheader("Top Products")
        top_products = filtered_df.groupby('Product Name')['Sales'].sum().nlargest(10).reset_index()
        fig5 = px.bar(top_products, x='Sales', y='Product Name', orientation='h', title='Top 10 Products by Revenue')
        st.plotly_chart(fig5, use_container_width=True)

# ==========================================
# PAGE 2: FORECAST EXPLORER
# ==========================================
elif page == "Forecast Explorer":
    st.markdown('<div class="main-header">🔮 Forecast Explorer</div>', unsafe_allow_html=True)
    
    if outputs['metrics'] is None or outputs['segment_forecasts'] is None:
        st.warning("⚠️ Forecasting outputs not found. Please run the analysis.ipynb notebook first to generate the models and forecasts.")
    else:
        st.subheader("Model Performance Metrics")
        st.dataframe(outputs['metrics'], use_container_width=True)
        
        st.markdown("---")
        st.subheader("Interactive Segment Forecast")
        
        seg_df = outputs['segment_forecasts']
        if 'Segment' in seg_df.columns:
            segments_available = seg_df['Segment'].unique()
            
            selected_seg = st.selectbox("Select Category / Region for Forecast", segments_available)
            
            st.write(f"Showing 3-Month Forecast for **{selected_seg}**")
            
            plot_df = seg_df[seg_df['Segment'] == selected_seg]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=plot_df['ds'], y=plot_df['yhat'], mode='lines+markers', name='Forecasted Sales', line=dict(color='orange', width=3)))
            fig.update_layout(title=f'{selected_seg} Forecast', xaxis_title='Date', yaxis_title='Sales ($)', template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("The forecasts are generated using Facebook Prophet with a multiplicative seasonality model.")
        else:
            st.warning("Segment data not available in outputs.")

# ==========================================
# PAGE 3: ANOMALY REPORT
# ==========================================
elif page == "Anomaly Report":
    st.markdown('<div class="main-header">🚨 Anomaly Report</div>', unsafe_allow_html=True)
    
    if outputs['anomalies'] is None:
        st.warning("⚠️ Anomaly output not found. Please run the analysis.ipynb notebook first.")
    else:
        anomaly_df = outputs['anomalies']
        if 'Order Date' in anomaly_df.columns and 'Sales' in anomaly_df.columns:
            anomaly_df['Order Date'] = pd.to_datetime(anomaly_df['Order Date'])
            
            st.subheader("Isolation Forest Anomalies")
            if 'Iso_Anomaly' in anomaly_df.columns:
                fig_iso = px.scatter(anomaly_df, x='Order Date', y='Sales', color=anomaly_df['Iso_Anomaly'].astype(str), 
                                     color_discrete_map={'-1': 'red', '1': 'blue'}, 
                                     title="Anomalies Detected (Red = Anomaly)",
                                     labels={'color': 'Normal(1) / Anomaly(-1)'})
                fig_iso.add_trace(go.Scatter(x=anomaly_df['Order Date'], y=anomaly_df['Sales'], mode='lines', name='Sales Trend', line=dict(color='lightgray', width=1)))
                st.plotly_chart(fig_iso, use_container_width=True)
            
            st.subheader("Z-Score Anomalies")
            if 'Z_Anomaly' in anomaly_df.columns:
                fig_z = px.scatter(anomaly_df, x='Order Date', y='Sales', color=anomaly_df['Z_Anomaly'].astype(str), 
                                     color_discrete_map={'-1': 'orange', '1': 'green'}, 
                                     title="Z-Score Anomalies (Orange = Anomaly)",
                                     labels={'color': 'Normal(1) / Anomaly(-1)'})
                fig_z.add_trace(go.Scatter(x=anomaly_df['Order Date'], y=anomaly_df['Sales'], mode='lines', name='Sales Trend', line=dict(color='lightgray', width=1)))
                st.plotly_chart(fig_z, use_container_width=True)
            
            st.markdown("---")
            st.subheader("Detailed Anomaly Data")
            if 'Iso_Anomaly' in anomaly_df.columns and 'Z_Anomaly' in anomaly_df.columns:
                st.dataframe(anomaly_df[(anomaly_df['Iso_Anomaly'] == -1) | (anomaly_df['Z_Anomaly'] == -1)])
            else:
                st.dataframe(anomaly_df)
        else:
            st.warning("Necessary columns not found in anomaly report.")

# ==========================================
# PAGE 4: DEMAND SEGMENTATION
# ==========================================
elif page == "Demand Segmentation":
    st.markdown('<div class="main-header">🎯 Product Demand Segmentation</div>', unsafe_allow_html=True)
    
    if outputs['clusters'] is None:
        st.warning("⚠️ Cluster output not found. Please run the analysis.ipynb notebook first.")
    else:
        cluster_df = outputs['clusters']
        
        if 'PCA1' in cluster_df.columns and 'PCA2' in cluster_df.columns and 'Business_Label' in cluster_df.columns:
            st.subheader("2D Cluster Visualization (PCA)")
            fig_cluster = px.scatter(cluster_df, x='PCA1', y='PCA2', color='Business_Label', hover_data=['Sub-Category'],
                                     title="Sub-Category Demand Clusters", size_max=15, size='Sales_Volume')
            st.plotly_chart(fig_cluster, use_container_width=True)
            
            st.markdown("---")
            st.subheader("Segmentation Table")
            display_cols = [c for c in ['Sub-Category', 'Sales_Volume', 'Growth_Rate', 'Business_Label'] if c in cluster_df.columns]
            st.dataframe(cluster_df[display_cols], use_container_width=True)
            
            st.markdown("---")
            st.subheader("Business Recommendations")
            
            col1, col2 = st.columns(2)
            with col1:
                st.success("**Cash Cows (High Volume)**\n\nMaintain high safety stock and automate reordering.")
                st.info("**High Growth, High Value**\n\nPrioritize premium shipping and aggressively restock.")
            with col2:
                st.warning("**Volatile / Seasonal**\n\nApply just-in-time (JIT) inventory and closely monitor trends.")
                st.error("**Steady & Low Value**\n\nOptimize storage costs, maintain minimal optimal stock.")
        else:
            st.warning("Necessary cluster columns not found.")
