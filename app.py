import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.ticker as mticker
import base64
import os
import datetime
import seaborn as sns
from sqlalchemy import create_engine
from sklearn.metrics import confusion_matrix

# Import from other modules
from database import import_leads_data, import_operations_data, initialize_db_if_empty, migrate_database, process_phone_matching
from data_manager import load_data, apply_filters
from utils import calculate_conversion_rates, calculate_correlations
from derive_scorecard import generate_lead_scorecard, score_lead
from conversion import analyze_phone_matches, analyze_time_to_conversion
from evaluate import (
    calculate_model_metrics, 
    plot_roc_curve, 
    plot_precision_recall_curve, 
    plot_score_distributions,
    get_custom_threshold_metrics
)
from findings import generate_findings
from segmentation import segment_leads, plot_clusters, plot_cluster_conversion_rates, plot_feature_importance_by_cluster
from advanced_analytics import run_all_analytics, plot_conversion_by_category
from mistral_insights import generate_sales_opportunity_analysis, generate_booking_type_recommendations, generate_customer_segment_insights

# Set page config and title
st.set_page_config(
    page_title="Sales Conversion Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        color: #1E88E5;
    }
    .section-header {
        font-size: 1.8rem !important;
        font-weight: 600 !important;
        color: #333;
        padding-top: 1rem;
    }
    .sub-header {
        font-size: 1.4rem !important;
        font-weight: 500 !important;
        color: #666;
    }
    .info-text {
        font-size: 1.1rem !important;
        color: #555;
        margin-bottom: 1rem;
    }
    .highlight {
        background-color: #e8f0fe;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1E88E5;
    }
    .insight-box {
        background-color: #f0f7ff;
        border-radius: 0.3rem;
        padding: 0.5rem;
        border-left: 3px solid #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# Title and Introduction
st.markdown("<div class='main-header'>Sales Conversion Analytics Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='info-text'>Analyze your sales conversion data, identify patterns, and optimize your lead scoring.</div>", unsafe_allow_html=True)

# Ensure database is initialized
initialize_db_if_empty()

# Data source selection
st.sidebar.header("Data Source")
data_source = st.sidebar.radio(
    "Select data source",
    ["Upload CSV Files", "Use Database Data"],
    index=1
)

# Initialize session state for storing processed data
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'lead_df' not in st.session_state:
    st.session_state.lead_df = None
if 'operation_df' not in st.session_state:
    st.session_state.operation_df = None
if 'weights_df' not in st.session_state:
    st.session_state.weights_df = None
if 'thresholds' not in st.session_state:
    st.session_state.thresholds = None

# Data loading section
if data_source == "Upload CSV Files":
    st.sidebar.header("Upload Data")
    leads_file = st.sidebar.file_uploader("Upload Leads CSV", type=["csv"])
    operations_file = st.sidebar.file_uploader("Upload Operations CSV", type=["csv"])
    
    if leads_file is not None:
        # Option to import into database
        if st.sidebar.button("Import Leads to Database"):
            # Save uploaded file temporarily
            temp_path = "temp_leads.csv"
            with open(temp_path, "wb") as f:
                f.write(leads_file.getvalue())
            
            # Import to database
            imported_count = import_leads_data(temp_path)
            st.sidebar.success(f"Successfully imported {imported_count} lead records to database")
    
    if operations_file is not None:
        # Option to import into database
        if st.sidebar.button("Import Operations to Database"):
            # Save uploaded file temporarily
            temp_path = "temp_operations.csv"
            with open(temp_path, "wb") as f:
                f.write(operations_file.getvalue())
            
            # Import to database
            imported_count = import_operations_data(temp_path)
            st.sidebar.success(f"Successfully imported {imported_count} operation records to database")
    
    # Process uploaded files if available using centralized data loader
    if leads_file is not None:
        try:
            # Use the new centralized data loading function
            leads_df, operations_df, processed_df = load_data(
                use_csv=True,
                leads_file=leads_file,
                operations_file=operations_file
            )
            
            # Store in session state
            st.session_state.lead_df = leads_df
            st.session_state.operation_df = operations_df
            st.session_state.processed_df = processed_df
            
            if leads_df is not None and not leads_df.empty:
                st.sidebar.success(f"Processed {len(leads_df)} lead records")
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
else:
    # Simplified loading from database without filters
    st.sidebar.header("Database Data")
    st.sidebar.info("Using all available data from the database")
    
    # Add a load button without filters
    if st.sidebar.button("Load All Data"):
        try:
            # Use the new centralized data loading function
            leads_df, operations_df, processed_df = load_data(use_csv=False)
            
            # Store in session state
            st.session_state.lead_df = leads_df
            st.session_state.operation_df = operations_df
            st.session_state.processed_df = processed_df
            
            if leads_df is not None and not leads_df.empty:
                st.sidebar.success(f"Loaded {len(leads_df)} lead records from database")
        except Exception as e:
            st.error(f"Error loading data from database: {str(e)}")

# Main content area with tabs
if st.session_state.processed_df is not None:
    # Get the processed dataframe and use it directly without filtering
    filtered_df = st.session_state.processed_df.copy()
    raw_df = filtered_df  # For backward compatibility
    
    # Define filters with default values for compatibility
    filters = {
        'date_range': None,
        'status': 'All',
        'states': ['All'],
        'date_col': 'inquiry_date' if 'inquiry_date' in filtered_df.columns else
                    'created' if 'created' in filtered_df.columns else
                    'event_date' if 'event_date' in filtered_df.columns else None
    }
    
    # Display a notice about filters being disabled
    st.info("Filters are currently disabled. Dashboard shows all available data.")
    
    # Create tabs for different analysis views
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "📊 Conversion Analysis", 
        "🔍 Feature Correlation", 
        "🤖 Lead Scoring", 
        "🗃️ Raw Data",
        "📈 Key Findings",
        "🛈 Explanations",
        "🧩 Lead Personas",
        "📊 Advanced Analytics",
        "🧠 AI Insights",
        "🔧 Debug"
    ])
    
    # First ensure all date columns are properly formatted as datetime
    for col in ['inquiry_date', 'created', 'event_date']:
        if col in filtered_df.columns:
            if filtered_df[col].dtype != 'datetime64[ns]':
                try:
                    filtered_df[col] = pd.to_datetime(filtered_df[col], errors='coerce')
                except Exception as e:
                    st.warning(f"Error converting {col} to datetime: {str(e)}")
    
    # Show info about applied filters
    if 'date_filter' in st.session_state and st.session_state.date_filter and len(st.session_state.date_filter) == 2:
        start_date, end_date = st.session_state.date_filter
        st.info(f"Filtered to {len(filtered_df)} leads from {start_date} to {end_date}")
    
    # Region filter information
    if 'region_filter' in st.session_state and st.session_state.region_filter and 'All' not in st.session_state.region_filter:
        st.info(f"Applied region filter: {', '.join(st.session_state.region_filter)}")

    # Tab 1: Conversion Analysis
    with tab1:
        try:
            # Import and use the dedicated conversion tab module
            from conversion_tab import render_conversion_tab
            
            # Call the modular implementation
            render_conversion_tab(filtered_df)
            
        except Exception as e:
            st.error(f"Error in Conversion Analysis tab: {str(e)}")

    # Tab 2: Feature Correlation
    with tab2:
        try:
            # Import and use the dedicated feature correlation tab module
            from feature_correlation_tab import render_feature_correlation_tab
            
            # Call the modular implementation
            render_feature_correlation_tab(filtered_df)
            
        except Exception as e:
            st.error(f"Error in Feature Correlation tab: {str(e)}")
            import traceback
            st.text(traceback.format_exc())

    # Tab 3: Lead Scoring
    with tab3:
        try:
            # Import and use the dedicated lead scoring tab module
            from lead_scoring_tab import render_lead_scoring_tab
            
            # Call the modular implementation
            render_lead_scoring_tab(filtered_df)
            
        except Exception as e:
            st.error(f"Error in Lead Scoring tab: {str(e)}")
            import traceback
            st.text(traceback.format_exc())

    # Tab 4: Raw Data
    with tab4:
        try:
            # Import and use the dedicated raw data tab module
            from raw_data_tab import render_raw_data_tab
            
            # Call the modular implementation
            render_raw_data_tab(filtered_df)
            
        except Exception as e:
            st.error(f"Error in Raw Data tab: {str(e)}")
            import traceback
            st.text(traceback.format_exc())

    # Tab 5: Key Findings
    with tab5:
        try:
            # Import and use the dedicated key findings tab module
            from key_findings_tab import render_key_findings_tab
            
            # Call the modular implementation
            render_key_findings_tab(filtered_df)
            
        except Exception as e:
            st.error(f"Error in Key Findings tab: {str(e)}")
            import traceback
            st.text(traceback.format_exc())
            
    # Tab 6: Explanations
    with tab6:
        try:
            # Import and use the dedicated explanations tab module
            from explanations_tab import render_explanations_tab
            
            # Call the modular implementation
            render_explanations_tab()
            
        except Exception as e:
            st.error(f"Error in Explanations tab: {str(e)}")
            import traceback
            st.text(traceback.format_exc())

# Tab 9: AI Insights
    with tab9:
        try:
            st.markdown("## 🧠 AI Insights")
            
            st.markdown("""
            This tab uses Mistral AI to generate advanced insights and recommendations based on your data.
            These AI-powered analyses can help identify patterns, opportunities, and strategies that might 
            not be immediately obvious from the charts and metrics alone.
            """)
            
            # Check for Mistral API key
            mistral_api_key = os.environ.get("MISTRAL_API_KEY")
            if not mistral_api_key:
                st.warning("⚠️ Mistral API key not found. Please set the MISTRAL_API_KEY environment variable to enable AI insights.")
                
                # Button to set API key in session state
                api_key_input = st.text_input("Enter your Mistral API key:", type="password")
                if st.button("Set API Key") and api_key_input:
                    os.environ["MISTRAL_API_KEY"] = api_key_input
                    st.success("API key set successfully! Refresh the page to see AI insights.")
            else:
                # Display AI analysis options
                st.subheader("Select Analysis Type")
                
                analysis_type = st.radio(
                    "Choose an analysis to generate:",
                    ["Sales Opportunity Analysis", "Booking Type Recommendations", "Customer Segment Insights"],
                    index=0
                )
                
                # Button to generate insights
                if st.button("Generate AI Insights"):
                    with st.spinner("Generating insights with Mistral AI... This may take a moment."):
                        try:
                            if analysis_type == "Sales Opportunity Analysis":
                                insights = generate_sales_opportunity_analysis(filtered_df)
                                st.subheader("🔍 Sales Opportunity Analysis")
                            elif analysis_type == "Booking Type Recommendations":
                                # Get booking type conversion rates
                                conversion_rates = calculate_conversion_rates(filtered_df)
                                booking_type_data = conversion_rates.get('booking_type', pd.DataFrame())
                                insights = generate_booking_type_recommendations(filtered_df, booking_type_data)
                                st.subheader("📝 Booking Type Recommendations")
                            else:  # Customer Segment Insights
                                insights = generate_customer_segment_insights(filtered_df)
                                st.subheader("👥 Customer Segment Insights")
                            
                            # Store insights in session state
                            st.session_state[f'{analysis_type.lower().replace(" ", "_")}_insights'] = insights
                        except Exception as e:
                            st.error(f"Error generating insights: {str(e)}")
                
                # Display insights if available in session state
                for insight_type in ['sales_opportunity_analysis', 'booking_type_recommendations', 'customer_segment_insights']:
                    if insight_type in st.session_state:
                        st.markdown(st.session_state[insight_type])
        
        except Exception as e:
            st.error(f"Error in AI Insights tab: {str(e)}")
            import traceback
            st.text(traceback.format_exc())

# Tab 10: Debug
    with tab10:
        try:
            st.markdown("## 🔧 Dashboard Diagnostics")
            
            # Import debug helpers
            from debug_helpers import create_health_check_table, get_data_summary, safe_display_df_preview
            
            st.markdown("### Data Summary")
            st.markdown(get_data_summary(filtered_df))
            
            st.markdown("### Tab Health Check")
            health_df = create_health_check_table(filtered_df)
            st.table(health_df)
            
            st.markdown("### DataFrame Preview")
            safe_display_df_preview(filtered_df)
            
            # Show session state variables
            st.markdown("### Session State Variables")
            session_state_vars = {k: str(v) for k, v in st.session_state.items()}
            st.json(session_state_vars)
            
            # Show filtered dataframe columns and counts
            st.markdown("### Column Statistics")
            
            # Display categorical columns distribution
            cat_cols = filtered_df.select_dtypes(include=['object', 'category']).columns.tolist()
            if cat_cols:
                selected_cat_col = st.selectbox("Select categorical column to analyze:", cat_cols)
                if selected_cat_col:
                    st.markdown(f"#### Distribution of {selected_cat_col}")
                    value_counts = filtered_df[selected_cat_col].value_counts().reset_index()
                    value_counts.columns = [selected_cat_col, 'Count']
                    st.table(value_counts.head(10))
            
            # Display numeric columns statistics
            num_cols = filtered_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            if num_cols:
                selected_num_col = st.selectbox("Select numeric column to analyze:", num_cols)
                if selected_num_col:
                    st.markdown(f"#### Statistics for {selected_num_col}")
                    stats = filtered_df[selected_num_col].describe()
                    st.dataframe(stats)
        
        except Exception as e:
            st.error(f"Error in Debug tab: {str(e)}")
            import traceback
            st.text(traceback.format_exc())
            
else:
    st.warning("No data loaded. Please select a data source and load data to begin.")

# Add a note about the project
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "Sales Conversion Analytics Dashboard v1.0\n\n"
    "This dashboard provides insights into sales conversion patterns, "
    "lead scoring, and predictive analytics to help optimize your sales process."
)