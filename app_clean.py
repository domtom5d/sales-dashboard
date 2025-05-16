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
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "📊 Conversion Analysis", 
        "🔍 Feature Correlation", 
        "🤖 Lead Scoring", 
        "🗃️ Raw Data",
        "📈 Key Findings",
        "🛈 Explanations",
        "🧩 Lead Personas",
        "📊 Advanced Analytics",
        "🧠 AI Insights"
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
        st.subheader("Feature Correlation Analysis")
        st.markdown("This analysis helps you understand which factors most strongly correlate with won and lost deals.")
        
        # Select only numeric columns for correlation analysis
        numeric_cols = filtered_df.select_dtypes(include=['int64', 'float64']).columns
        
        if len(numeric_cols) > 0 and 'outcome' in filtered_df.columns:
            # Calculate and display correlation with outcome
            outcome_corr = calculate_correlations(filtered_df, 'outcome', numeric_cols)
            
            st.markdown("### Correlation with Deal Outcome")
            st.markdown("Positive values indicate factors that correlate with winning deals, negative values with losing deals.")
            
            # Create a horizontal bar chart of correlations
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Sort by absolute correlation value
            outcome_corr = outcome_corr.sort_values(by='correlation', key=abs, ascending=False)
            
            # Create color map (blue for positive, red for negative)
            colors = ['#1E88E5' if c >= 0 else '#f44336' for c in outcome_corr['correlation']]
            
            # Plot data
            ax.barh(outcome_corr['feature'], outcome_corr['correlation'], color=colors)
            
            # Customize plot
            ax.set_xlabel('Correlation with Win/Loss')
            ax.set_ylabel('Feature')
            ax.set_title('Feature Correlation with Deal Outcome')
            ax.grid(axis='x', linestyle='--', alpha=0.7)
            
            # Add labels with the actual correlation values
            for i, v in enumerate(outcome_corr['correlation']):
                ax.text(v + (0.01 if v >= 0 else -0.01), 
                        i, 
                        f"{v:.2f}", 
                        va='center', 
                        ha='left' if v >= 0 else 'right',
                        fontweight='bold')
            
            st.pyplot(fig)
            
            # Display feature explanations
            st.markdown("### Feature Explanations")
            for i, row in outcome_corr.iterrows():
                # Skip features with very low correlation
                if abs(row['correlation']) < 0.05:
                    continue
                
                feature = row['feature']
                corr = row['correlation']
                
                if corr > 0:
                    st.markdown(f"**{feature}**: Positive correlation ({corr:.2f}) - Higher values tend to be associated with **won deals**")
                else:
                    st.markdown(f"**{feature}**: Negative correlation ({corr:.2f}) - Higher values tend to be associated with **lost deals**")
        else:
            st.warning("Not enough numeric data available for correlation analysis. Make sure your dataset includes numeric features and outcome labels.")

    # Tab 3: Lead Scoring
    with tab3:
        st.subheader("Lead Scoring Model")
        st.markdown("Use this model to evaluate new leads and focus on the most promising opportunities.")
        
        if 'outcome' in filtered_df.columns:
            # Generate scorecard if not already in session state
            if 'weights_df' not in st.session_state or st.session_state.weights_df is None:
                try:
                    # Generate model
                    weights_df, thresholds, metrics = generate_lead_scorecard(filtered_df)
                    
                    # Store in session state
                    st.session_state.weights_df = weights_df
                    st.session_state.thresholds = thresholds
                    st.session_state.metrics = metrics
                    
                    st.success("Lead scoring model generated successfully!")
                except Exception as e:
                    st.error(f"Error generating lead scoring model: {str(e)}")
                    weights_df = None
                    thresholds = None
                    metrics = None
            else:
                weights_df = st.session_state.weights_df
                thresholds = st.session_state.thresholds
                metrics = st.session_state.metrics
            
            if weights_df is not None:
                # Display model metrics
                st.markdown("### Model Performance")
                if 'metrics' in st.session_state and st.session_state.metrics is not None:
                    metrics = st.session_state.metrics
                    
                    # Display metrics in columns
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("ROC AUC", f"{metrics['roc_auc']:.3f}")
                    
                    with col2:
                        st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                    
                    with col3:
                        st.metric("F1 Score", f"{metrics['f1']:.3f}")
                
                # Display feature weights
                st.markdown("### Feature Weights")
                st.markdown("These weights show how important each feature is for predicting won deals.")
                
                # Sort by absolute weight
                weights_df = weights_df.sort_values(by='weight', key=abs, ascending=False)
                
                # Create a horizontal bar chart of weights
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Create color map (blue for positive, red for negative)
                colors = ['#1E88E5' if w >= 0 else '#f44336' for w in weights_df['weight']]
                
                # Plot data
                ax.barh(weights_df['feature'], weights_df['weight'], color=colors)
                
                # Customize plot
                ax.set_xlabel('Weight')
                ax.set_ylabel('Feature')
                ax.set_title('Feature Importance in Lead Scoring Model')
                ax.grid(axis='x', linestyle='--', alpha=0.7)
                
                # Add labels with the actual weight values
                for i, v in enumerate(weights_df['weight']):
                    ax.text(v + (0.01 if v >= 0 else -0.01), 
                            i, 
                            f"{v:.2f}", 
                            va='center', 
                            ha='left' if v >= 0 else 'right',
                            fontweight='bold')
                
                st.pyplot(fig)
                
                # Score distribution plot
                if 'y_probs' in st.session_state and 'y_true' in st.session_state:
                    st.markdown("### Score Distribution")
                    fig = plot_score_distributions(
                        st.session_state.y_probs, 
                        st.session_state.y_true,
                        thresholds
                    )
                    st.pyplot(fig)
                
                # Lead scoring calculator
                st.markdown("### Lead Scoring Calculator")
                st.markdown("Enter values for a new lead to calculate its score and conversion probability.")
                
                # Create a form for lead scoring
                with st.form("lead_scoring_form"):
                    # Get input fields for the top features
                    input_values = {}
                    
                    # Create two columns for the form
                    form_col1, form_col2 = st.columns(2)
                    
                    # Show form fields for the top features
                    top_features = weights_df.head(10)['feature'].tolist()
                    for i, feature in enumerate(top_features):
                        # Alternate between columns
                        col = form_col1 if i % 2 == 0 else form_col2
                        
                        # Create input field based on feature name
                        if 'days' in feature.lower() or 'guests' in feature.lower() or 'bartenders' in feature.lower():
                            input_values[feature] = col.number_input(f"{feature}", min_value=0, step=1)
                        elif 'price' in feature.lower() or 'value' in feature.lower() or 'revenue' in feature.lower():
                            input_values[feature] = col.number_input(f"{feature}", min_value=0.0, step=100.0)
                        else:
                            input_values[feature] = col.number_input(f"{feature}", min_value=0.0, step=0.1)
                    
                    # Submit button
                    submit_button = st.form_submit_button("Calculate Score")
                
                # Process form submission
                if submit_button:
                    score, probability = score_lead(input_values, weights_df)
                    
                    # Determine the lead category based on thresholds
                    category = "Cool 🧊"
                    if probability >= thresholds['hot']:
                        category = "Hot 🔥"
                    elif probability >= thresholds['warm']:
                        category = "Warm 🟠"
                    
                    # Display results
                    st.markdown("### Lead Score Results")
                    
                    result_col1, result_col2, result_col3 = st.columns(3)
                    
                    with result_col1:
                        st.metric("Raw Score", f"{score:.2f}")
                    
                    with result_col2:
                        st.metric("Win Probability", f"{probability:.1%}")
                    
                    with result_col3:
                        st.metric("Lead Category", category)
                    
                    # Provide explanation
                    st.markdown("### Score Explanation")
                    
                    # Calculate feature contributions
                    contributions = []
                    for feature, weight in zip(weights_df['feature'], weights_df['weight']):
                        if feature in input_values:
                            contribution = weight * input_values[feature]
                            contributions.append({
                                'feature': feature,
                                'value': input_values[feature],
                                'weight': weight,
                                'contribution': contribution
                            })
                    
                    # Convert to DataFrame and sort by absolute contribution
                    contrib_df = pd.DataFrame(contributions)
                    contrib_df = contrib_df.sort_values(by='contribution', key=abs, ascending=False)
                    
                    # Display the top positive and negative factors
                    positive_factors = contrib_df[contrib_df['contribution'] > 0].head(3)
                    negative_factors = contrib_df[contrib_df['contribution'] < 0].head(3)
                    
                    if not positive_factors.empty:
                        st.markdown("**Top Positive Factors:**")
                        for i, row in positive_factors.iterrows():
                            st.markdown(f"- **{row['feature']}**: Value of {row['value']} contributes +{row['contribution']:.2f} to the score")
                    
                    if not negative_factors.empty:
                        st.markdown("**Top Negative Factors:**")
                        for i, row in negative_factors.iterrows():
                            st.markdown(f"- **{row['feature']}**: Value of {row['value']} contributes {row['contribution']:.2f} to the score")
            else:
                st.warning("Unable to generate lead scoring model. Please ensure you have sufficient data with clear outcomes.")
        else:
            st.warning("Outcome column not found in the data. Lead scoring requires a column indicating won/lost status.")

    # Tab 4: Raw Data
    with tab4:
        st.subheader("Raw Data")
        st.markdown("Explore the underlying dataset used for all analyses.")
        
        # Display the dataframe
        st.dataframe(filtered_df)
        
        # Download options
        st.markdown("### Download Data")
        
        # Convert to CSV
        csv = filtered_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="sales_data.csv">Download CSV</a>'
        st.markdown(href, unsafe_allow_html=True)
        
        # Show data info
        with st.expander("Data Information"):
            # Show column descriptions
            st.markdown("#### Column Descriptions")
            col_desc = pd.DataFrame({
                'Column': filtered_df.columns.tolist(),
                'Type': filtered_df.dtypes.astype(str).tolist(),
                'Non-Null Count': filtered_df.count().tolist(),
                'Null Count': filtered_df.isna().sum().tolist(),
                'Non-Null %': (filtered_df.count() / len(filtered_df) * 100).round(2).astype(str) + '%'
            })
            st.dataframe(col_desc)

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