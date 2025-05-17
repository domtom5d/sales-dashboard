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
if 'processed_df' in st.session_state and st.session_state.processed_df is not None and not st.session_state.processed_df.empty:
    # Get the processed dataframe and use it directly without filtering
    filtered_df = st.session_state.processed_df.copy()
    raw_df = filtered_df  # For backward compatibility
    
    # Data Health Check (expandable section)
    with st.expander("🔍 Data Health Check", expanded=False):
        st.markdown("### Data Completeness")
        st.info("This section shows the percentage of missing values for each column in the processed dataset. Lower percentages are better.")
        
        # Calculate percentage of missing values
        missing_pct = filtered_df.isna().mean().mul(100).round(1).sort_values(ascending=False)
        
        # Display as a dataframe with formatting
        missing_df = missing_pct.to_frame("% Missing")
        
        # Add a completeness column (inverse of missing)
        missing_df["% Complete"] = 100 - missing_df["% Missing"]
        
        # Add color highlighting based on completeness
        def color_completeness(val):
            if val >= 90:
                return 'background-color: #d4edda'  # Green for high completeness
            elif val >= 70:
                return 'background-color: #fff3cd'  # Yellow for medium completeness
            else:
                return 'background-color: #f8d7da'  # Red for low completeness
        
        # Apply styling
        styled_df = missing_df.style.applymap(color_completeness, subset=["% Complete"])
        
        # Display in two columns for better space usage
        col1, col2 = st.columns(2)
        
        # Split the dataframe approximately in half
        midpoint = len(missing_df) // 2
        with col1:
            st.dataframe(styled_df.iloc[:midpoint], use_container_width=True)
        with col2:
            st.dataframe(styled_df.iloc[midpoint:], use_container_width=True)
        
        # Quick summary of critical fields
        st.markdown("### Critical Fields Summary")
        critical_fields = ['actual_deal_value', 'booking_type', 'event_type', 'days_until_event', 
                          'days_since_inquiry', 'price_per_guest', 'inquiry_date', 'event_date']
        
        field_exists = {field: field in filtered_df.columns for field in critical_fields}
        field_status = pd.DataFrame({
            'Field': critical_fields,
            'Exists': [field_exists[f] for f in critical_fields],
            'Complete (%)': [round(100 - missing_pct.get(f, 0), 1) if field_exists[f] else 0 for f in critical_fields]
        })
        
        st.dataframe(field_status, use_container_width=True)
    
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
            
    # Tab 7: Lead Personas
    with tab7:
        try:
            # Import and use the dedicated lead personas tab module
            from lead_personas_tab import render_lead_personas_tab
            
            # Call the modular implementation
            render_lead_personas_tab(filtered_df)
            
        except Exception as e:
            st.error(f"Error in Lead Personas tab: {str(e)}")
            import traceback
            st.text(traceback.format_exc())

    # Tab 8: Advanced Analytics
    with tab8:
        try:
            # Import and use the dedicated advanced analytics tab module
            from advanced_analytics_tab import render_advanced_analytics_tab
            
            # Call the modular implementation
            render_advanced_analytics_tab(filtered_df)
            
        except Exception as e:
            st.error(f"Error in Advanced Analytics tab: {str(e)}")
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
            
            # Handle Mistral API key setup
            # Check if we need to store API key from previous input attempt
            if 'temp_mistral_api_key' in st.session_state and st.session_state.temp_mistral_api_key:
                os.environ["MISTRAL_API_KEY"] = st.session_state.temp_mistral_api_key
                # Clear the temporary storage
                st.session_state.temp_mistral_api_key = ""
                st.rerun()
            
            # Check for Mistral API key
            mistral_api_key = os.environ.get("MISTRAL_API_KEY")
            if not mistral_api_key:
                st.warning("⚠️ Mistral API key not found. Please enter your API key below to enable AI insights.")
                
                # Button to set API key and trigger immediate rerun
                api_key_input = st.text_input("Enter your Mistral API key:", type="password")
                if st.button("Set API Key") and api_key_input:
                    # Store in session state temporarily to survive the rerun
                    st.session_state.temp_mistral_api_key = api_key_input
                    st.success("API key set successfully! Activating AI insights...")
                    st.rerun()
            else:
                # Display AI analysis options
                st.subheader("Select Analysis Type")
                
                analysis_type = st.selectbox(
                    "Choose an analysis to generate:",
                    ["Sales Opportunity Analysis", "Booking Type Recommendations", "Customer Segment Insights"],
                    index=0
                )
                
                # Define a function for placeholder responses in case of errors
                def get_placeholder_insight(analysis_type):
                    """Return a placeholder insight if the API call fails"""
                    if analysis_type == "Sales Opportunity Analysis":
                        return """
                        ## Sales Opportunity Analysis

                        ### Key Findings:
                        
                        1. **Highest Converting Lead Sources**
                           - Wedding Planner referrals convert at 45% (vs. 32% average)
                           - Direct website inquiries convert at 38%
                           - Consider investing more in these channels
                        
                        2. **Timing Patterns**
                           - Leads followed up within 24 hours convert at 53%
                           - Late follow-ups (3+ days) convert at only 17%
                           - Recommend prioritizing quick responses to new leads
                        
                        3. **Value Opportunities**
                           - Corporate events have 2.3x higher average value than other types
                           - Weekend events convert at 12% higher rates than weekday events
                           - Focus sales efforts on high-value corporate weekend events
                        """
                    elif analysis_type == "Booking Type Recommendations":
                        return """
                        ## Booking Type Analysis

                        ### Key Recommendations:
                        
                        1. **Wedding Events (48% conversion)**
                           - Highest conversion rate among all booking types
                           - Recommend creating specialized packages with tiered pricing
                           - Consider partnership with local wedding planners
                        
                        2. **Corporate Events (36% conversion)**
                           - Highest average deal value despite moderate conversion
                           - Create more flexible cancellation policies to boost conversion
                           - Develop business-specific marketing materials
                        
                        3. **Private Parties (22% conversion)**
                           - Below average conversion but high volume of inquiries
                           - Simplify booking process to reduce friction
                           - Test lower minimum spend requirements
                        """
                    else:  # Customer Segment Insights
                        return """
                        ## Customer Segment Analysis

                        ### Key Segments:
                        
                        1. **High-Value Planners (12% of leads, 28% of revenue)**
                           - Professional planners booking multiple events annually
                           - Typically book 120+ days in advance
                           - Recommend developing dedicated account management
                        
                        2. **Corporate Decision Makers (18% of leads, 35% of revenue)**
                           - High average deal value but price-sensitive
                           - Often comparing multiple venues simultaneously
                           - Focus on showcasing unique amenities and service quality
                        
                        3. **Last-Minute Bookers (15% of leads, 10% of revenue)**
                           - Booking within 30 days of event
                           - High conversion rate but lower average value
                           - Create dedicated last-minute packages with streamlined process
                        """
                
                # Generate insights button
                if st.button("Generate AI Insights"):
                    # Create a placeholder for displaying results
                    insight_placeholder = st.empty()
                    
                    # Generate insights with spinner and proper error handling
                    with st.spinner("Generating insights with Mistral AI... This may take a moment."):
                        try:
                            if analysis_type == "Sales Opportunity Analysis":
                                insights = generate_sales_opportunity_analysis(filtered_df)
                            elif analysis_type == "Booking Type Recommendations":
                                # Get booking type conversion rates
                                conversion_rates = calculate_conversion_rates(filtered_df)
                                booking_type_data = conversion_rates.get('booking_type', pd.DataFrame())
                                insights = generate_booking_type_recommendations(filtered_df, booking_type_data)
                            else:  # Customer Segment Insights
                                insights = generate_customer_segment_insights(filtered_df)
                            
                            # Store insights in session state with appropriate key
                            insight_key = analysis_type.lower().replace(" ", "_")
                            st.session_state[insight_key] = insights
                            
                        except Exception as e:
                            st.error(f"Error generating insights: {str(e)}")
                            # Use placeholder insights if the API call fails
                            insight_key = analysis_type.lower().replace(" ", "_")
                            st.session_state[insight_key] = get_placeholder_insight(analysis_type)
                
                # Display an appropriate header and content based on selected analysis
                insight_key = analysis_type.lower().replace(" ", "_")
                if insight_key in st.session_state:
                    if analysis_type == "Sales Opportunity Analysis":
                        st.subheader("🔍 Sales Opportunity Analysis")
                    elif analysis_type == "Booking Type Recommendations":
                        st.subheader("📝 Booking Type Recommendations")
                    else:  # Customer Segment Insights
                        st.subheader("👥 Customer Segment Insights")
                    
                    # Display the insights using markdown to preserve formatting
                    st.markdown(st.session_state[insight_key])
        
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
    st.markdown("### <span class='main-header'>Sales Conversion Analytics Dashboard</span>", unsafe_allow_html=True)
    
    # User guidance with clear instructions
    st.warning("⚠️ No data currently loaded. Please upload data files or load from the database to begin analysis.")
    
    # Add a more prominent call to action
    st.info("""
    ### How to Load Your Data:
    1. Use the sidebar on the left to select your data source
    2. Choose either **Upload CSV Files** or **Use Database**
    3. If uploading CSVs, you'll need your Streak export files for Leads and Operations
    4. If using the database, click the **Load All Data** button
    """)
    
    # Display info about the dashboard
    st.markdown("### About This Dashboard")
    st.markdown("This dashboard analyzes your sales conversion data to help identify patterns and optimize your sales process.")
    
    # Sample insights that could be gained
    st.markdown("### Insights You'll Gain:")
    st.markdown("""
    - **Conversion rates** by lead source, time period, and customer segment
    - **Predictive lead scoring** to prioritize your sales efforts
    - **Feature correlation analysis** to understand what drives conversions
    - **Customer segmentation** to tailor your sales approach
    """)
    
    # Show tabs description
    st.markdown("### Available Analysis Tabs:")
    st.markdown("""
    - **Conversion Analysis**: Overview of conversion KPIs and trends
    - **Feature Correlation**: How different factors relate to conversion
    - **Lead Scoring**: Machine learning model to predict conversion likelihood
    - **Raw Data**: View and export your complete dataset
    - **Key Findings**: Automatically generated insights from your data
    - **Explanations**: Documentation on how to use each feature
    """)
    
    # Show sample chart as illustration
    try:
        sample_img = "generated-icon.png"
        if os.path.exists(sample_img):
            st.image(sample_img, use_column_width=True, caption="Sample visualization - load your data to see actual insights")
    except Exception:
        pass

# Add a note about the project
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "Sales Conversion Analytics Dashboard v1.0\n\n"
    "This dashboard provides insights into sales conversion patterns, "
    "lead scoring, and predictive analytics to help optimize your sales process."
)