import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
import os
import datetime
import seaborn as sns
from sqlalchemy import create_engine
from sklearn.metrics import confusion_matrix
from database import import_leads_data, import_operations_data, get_lead_data, get_operation_data, get_merged_data, initialize_db_if_empty, migrate_database, process_phone_matching
from utils import process_data, calculate_conversion_rates, calculate_correlations
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
    .info-text {
        font-size: 1.0rem !important;
        color: #555;
    }
    .highlight {
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
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = None

# Initialize and migrate database schema
try:
    # First run the database migration to ensure new columns exist
    if st.sidebar.button("Run Database Migration"):
        with st.spinner("Migrating database schema to add contact fields..."):
            success = migrate_database()
            if success:
                st.sidebar.success("Database migration completed successfully")
            else:
                st.sidebar.error("Database migration failed")
    
    # Initialize database if empty
    try:
        # This is just a check, don't load the models with new columns yet
        is_initialized = initialize_db_if_empty()
    except Exception as e:
        st.sidebar.warning(f"Database needs migration: {e}")
except Exception as e:
    st.sidebar.error(f"Database initialization error: {e}")

# Data loading section
if data_source == "Upload CSV Files":
    st.sidebar.header("Upload Data")
    leads_file = st.sidebar.file_uploader("Upload Leads CSV", type=["csv"])
    operations_file = st.sidebar.file_uploader("Upload Operations CSV", type=["csv"])
    
    if leads_file is not None:
        try:
            df_leads = pd.read_csv(leads_file)
            st.session_state.lead_df = df_leads
            
            # Option to import into database
            if st.sidebar.button("Import Leads to Database"):
                # Save uploaded file temporarily
                temp_path = "temp_leads.csv"
                with open(temp_path, "wb") as f:
                    f.write(leads_file.getvalue())
                
                # Import to database
                imported_count = import_leads_data(temp_path)
                st.sidebar.success(f"Successfully imported {imported_count} lead records to database")
        except Exception as e:
            st.error(f"Error loading leads file: {str(e)}")
    
    if operations_file is not None:
        try:
            df_operations = pd.read_csv(operations_file)
            st.session_state.operation_df = df_operations
            
            # Option to import into database
            if st.sidebar.button("Import Operations to Database"):
                # Save uploaded file temporarily
                temp_path = "temp_operations.csv"
                with open(temp_path, "wb") as f:
                    f.write(operations_file.getvalue())
                
                # Import to database
                imported_count = import_operations_data(temp_path)
                st.sidebar.success(f"Successfully imported {imported_count} operation records to database")
        except Exception as e:
            st.error(f"Error loading operations file: {str(e)}")
    
    if st.session_state.lead_df is not None:
        # Process data if leads are available
        st.session_state.processed_df = process_data(
            st.session_state.lead_df, 
            st.session_state.operation_df
        )
else:
    # Load from database with filters
    st.sidebar.header("Filter Database Data")
    
    # Date range filters
    start_date = st.sidebar.date_input(
        "Start Date",
        datetime.date.today() - datetime.timedelta(days=365)
    )
    end_date = st.sidebar.date_input(
        "End Date",
        datetime.date.today()
    )
    
    # Status filter
    status_options = ["All", "Won", "Lost", "In Progress"]
    status_filter = st.sidebar.multiselect(
        "Status",
        status_options,
        default=["All"]
    )
    
    # Apply filters
    if st.sidebar.button("Load Data"):
        try:
            # Convert status filter to database format
            db_status_filter = None
            if status_filter and "All" not in status_filter:
                db_status_filter = {
                    "status": status_filter
                }
            
            # Load data from database
            leads_df = get_lead_data()
            operations_df = get_operation_data()
            
            if leads_df is not None:
                # Normalize all column names to snake_case for consistency
                if not leads_df.empty:
                    # Show available columns for debugging
                    st.sidebar.expander("Debug: Available Columns").write(leads_df.columns.tolist())
                    
                    # Normalize column names
                    leads_df.columns = (
                        leads_df.columns
                        .astype(str)
                        .str.strip()
                        .str.lower()
                        .str.replace(r'\s+', '_', regex=True)
                    )
                
                if operations_df is not None and not operations_df.empty:
                    # Normalize column names
                    operations_df.columns = (
                        operations_df.columns
                        .astype(str)
                        .str.strip()
                        .str.lower()
                        .str.replace(r'\s+', '_', regex=True)
                    )
                
                st.session_state.lead_df = leads_df
                st.session_state.operation_df = operations_df
                
                # Process the data
                st.session_state.processed_df = process_data(
                    leads_df, 
                    operations_df
                )
                
                st.sidebar.success(f"Loaded {len(leads_df)} lead records from database")
        except Exception as e:
            st.error(f"Error loading data from database: {str(e)}")

# Main content area with tabs
if st.session_state.processed_df is not None:
    # Get the processed dataframe
    filtered_df = st.session_state.processed_df
    
    # Create tabs for different analysis views
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Conversion Analysis", 
        "🔍 Feature Correlation", 
        "🤖 Lead Scoring", 
        "🗃️ Raw Data",
        "📈 Key Findings",        # NEW
        "🛈 Explanations"
    ])
    
    with tab1:
        try:
            # Calculate conversion rates by different categories
            conversion_rates = calculate_conversion_rates(filtered_df)
            
            st.subheader("Conversion Rate Analysis")
            
            # Summary metrics
            overall_conversion = conversion_rates["overall"]["Conversion Rate"][0]
            st.metric("Overall Conversion Rate", f"{overall_conversion:.1%}")
            
            # Calculate and display time to conversion analysis
            st.subheader("⏱️ Time to Conversion Analysis")
            time_to_conversion = analyze_time_to_conversion(filtered_df)
            
            if time_to_conversion.get('error'):
                st.warning(f"Could not analyze time to conversion: {time_to_conversion.get('error')}")
            else:
                # Create metrics row
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                
                with metric_col1:
                    st.metric("Average Days", f"{time_to_conversion.get('average_days', 0):.1f}")
                
                with metric_col2:
                    st.metric("Median Days", f"{time_to_conversion.get('median_days', 0):.1f}")
                
                with metric_col3:
                    st.metric("Minimum Days", f"{time_to_conversion.get('min_days', 0)}")
                
                with metric_col4:
                    st.metric("Maximum Days", f"{time_to_conversion.get('max_days', 0)}")
                
                # Display histogram of time to conversion
                time_col1, time_col2 = st.columns(2)
                
                with time_col1:
                    if 'histogram_data' in time_to_conversion and not time_to_conversion['histogram_data'].empty:
                        st.write("**Distribution of Time to Conversion**")
                        
                        # Create a horizontal bar chart
                        fig, ax = plt.subplots(figsize=(10, 6))
                        bars = sns.barplot(x='Count', y='Time to Conversion', 
                                       data=time_to_conversion['histogram_data'], 
                                       palette='viridis', ax=ax)
                        
                        # Add count labels - in a more type-safe way
                        for i, (_, row) in enumerate(time_to_conversion['histogram_data'].iterrows()):
                            count = row['Count']
                            ax.text(count + 0.5, i, f"{count:.0f}", va='center')
                        
                        ax.set_title('Distribution of Time from Lead to Conversion')
                        plt.tight_layout()
                        st.pyplot(fig)
                
                with time_col2:
                    # Display conversion time by booking type
                    if 'by_booking_type' in time_to_conversion and not time_to_conversion['by_booking_type'].empty:
                        st.write("**Time to Conversion by Booking Type**")
                        # Format columns for better display
                        display_df = time_to_conversion['by_booking_type'].copy()
                        display_df['mean'] = display_df['mean'].round(1)
                        display_df['median'] = display_df['median'].round(1)
                        display_df.columns = ['Booking Type', 'Avg Days', 'Median Days', 'Count']
                        st.dataframe(display_df.sort_values(by='Avg Days'), use_container_width=True)
                    
                    # Display conversion time by event type
                    if 'by_event_type' in time_to_conversion and not time_to_conversion['by_event_type'].empty:
                        st.write("**Time to Conversion by Event Type**")
                        # Format columns for better display
                        display_df = time_to_conversion['by_event_type'].copy()
                        display_df['mean'] = display_df['mean'].round(1)
                        display_df['median'] = display_df['median'].round(1)
                        display_df.columns = ['Event Type', 'Avg Days', 'Median Days', 'Count']
                        st.dataframe(display_df.sort_values(by='Avg Days'), use_container_width=True)
            
            # Create columns for additional metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Plot conversion by booking type using the improved horizontal bar chart
                st.write("#### Conversion by Booking Type")
                
                if "booking_type" in conversion_rates and not conversion_rates["booking_type"].empty:
                    # Get the top 8 booking types by conversion rate
                    booking_df = conversion_rates["booking_type"].copy()
                    
                    # Only include booking types with at least 3 leads
                    booking_df = booking_df[booking_df['total'] >= 3]
                    
                    # Get top 8 or all if less than 8
                    top_booking_types = booking_df.head(8)
                    
                    # Create horizontal bar chart
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.barh(top_booking_types['Booking Type'], 
                                  top_booking_types['Conversion Rate'],
                                  edgecolor='black',
                                  color='skyblue')
                    
                    # Set labels and limits
                    ax.set_xlabel('Conversion Rate')
                    ax.set_xlim(0, min(1, top_booking_types['Conversion Rate'].max() * 1.2))
                    
                    # Add annotations with conversion rate percentage and sample size
                    for i, (rate, total) in enumerate(zip(top_booking_types['Conversion Rate'], 
                                                         top_booking_types['total'])):
                        ax.text(rate + 0.01, i, f"{rate:.0%} (n={int(total)})", va='center')
                    
                    # Add a title
                    plt.title('Top Booking Types by Conversion Rate')
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                else:
                    st.info("No booking type data available.")
            
            with col2:
                # Plot conversion by referral source
                st.write("#### Conversion by Referral Source")
                fig, ax = plt.subplots(figsize=(8, 5))
                conversion_rates["referral_source"].plot(kind="bar", x="Referral Source", y="Conversion Rate", ax=ax)
                ax.set_xlabel("Referral Source")
                ax.set_ylabel("Conversion Rate")
                ax.set_ylim(0, min(1, conversion_rates["referral_source"]["Conversion Rate"].max() * 1.5))
                st.pyplot(fig)
            
            with col3:
                # Plot conversion by days until event
                st.write("#### Conversion by Days Until Event")
                fig, ax = plt.subplots(figsize=(8, 5))
                conversion_rates["days_until_event"].plot(kind="bar", x="Days Until Event Bin", y="Conversion Rate", ax=ax)
                ax.set_xlabel("Days Until Event")
                ax.set_ylabel("Conversion Rate")
                ax.set_ylim(0, min(1, conversion_rates["days_until_event"]["Conversion Rate"].max() * 1.5))
                st.pyplot(fig)
                
        except Exception as e:
            st.error(f"Error in conversion analysis: {str(e)}")
    
    with tab2:
        try:
            # Calculate correlations with outcome
            corr_outcome, corr_matrix = calculate_correlations(filtered_df)
            
            st.subheader("Feature Correlation Analysis")
            
            # Display correlation with outcome
            st.write("#### Correlation with Conversion Outcome")
            corr_outcome = corr_outcome.sort_values("Correlation with Outcome", ascending=False)
            
            # Plot the correlation with outcome
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(corr_outcome.index, corr_outcome["Correlation with Outcome"])
            
            # Color bars based on correlation direction
            for i, bar in enumerate(bars):
                if corr_outcome["Correlation with Outcome"].iloc[i] > 0:
                    bar.set_color("green")
                else:
                    bar.set_color("red")
            
            ax.set_xlabel("Correlation with Conversion")
            ax.axvline(x=0, color="gray", linestyle="-", alpha=0.3)
            st.pyplot(fig)
            
            # Interpretation of results
            st.write("#### Interpretation")
            
            # Get top positive and negative correlations
            if 'Correlation with Outcome' in corr_outcome.columns:
                top_positive = corr_outcome[corr_outcome["Correlation with Outcome"] > 0].head(3)
                top_negative = corr_outcome[corr_outcome["Correlation with Outcome"] < 0].tail(3)
                
                if not top_positive.empty:
                    st.write("**Top Positive Factors:**")
                    for _, row in top_positive.iterrows():
                        feature = row['index'] if 'index' in row else 'Feature'
                        st.write(f"• {feature}: +{row['Correlation with Outcome']:.3f}")
                
                if not top_negative.empty:
                    st.write("**Top Negative Factors:**")
                    for _, row in top_negative.iterrows():
                        feature = row['index'] if 'index' in row else 'Feature'
                        st.write(f"• {feature}: {row['Correlation with Outcome']:.3f}")
            else:
                st.warning("Correlation data structure is not as expected. Unable to display top factors.")
            
            # Plot correlation matrix for top features
            st.write("#### Feature Correlation Matrix")
            
            # Check if we have valid data for correlation matrix
            if not corr_matrix.empty and 'Correlation with Outcome' in corr_outcome.columns:
                try:
                    # Create a list of top features
                    feature_list = []
                    
                    # Add features from top_positive
                    if not top_positive.empty:
                        for _, row in top_positive.iterrows():
                            if 'index' in row:
                                feature_list.append(row['index'])
                    
                    # Add features from top_negative
                    if not top_negative.empty:
                        for _, row in top_negative.iterrows():
                            if 'index' in row:
                                feature_list.append(row['index'])
                    
                    # Add outcome column
                    if 'Outcome' in corr_matrix.columns:
                        feature_list.append('Outcome')
                    
                    # If we have features to display
                    if feature_list:
                        # Create heatmap
                        top_corr_matrix = corr_matrix.loc[feature_list, feature_list]
                        
                        fig, ax = plt.subplots(figsize=(10, 8))
                        im = ax.imshow(top_corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
                        
                        # Add feature labels
                        ax.set_xticks(np.arange(len(feature_list)))
                        ax.set_yticks(np.arange(len(feature_list)))
                        ax.set_xticklabels(feature_list, rotation=45, ha="right")
                        ax.set_yticklabels(feature_list)
                        
                        # Add colorbar
                        plt.colorbar(im)
                        
                        # Add correlation values
                        for i in range(len(feature_list)):
                            for j in range(len(feature_list)):
                                text = ax.text(j, i, f"{top_corr_matrix.iloc[i, j]:.2f}",
                                            ha="center", va="center", color="black" if abs(top_corr_matrix.iloc[i, j]) < 0.7 else "white")
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.info("Not enough features for correlation matrix visualization.")
                except Exception as e:
                    st.warning(f"Could not create correlation matrix: {str(e)}")
            else:
                st.info("Not enough data for correlation analysis.")
        except Exception as e:
            st.error(f"Error in feature correlation analysis: {str(e)}")
    
    with tab3:
        try:
            st.subheader("Lead Scoring Model")
            
            # Button to generate lead scoring model
            if st.button("Generate Lead Scoring Model"):
                # Generate the lead scoring model
                scorecard_df, thresholds, model_metrics = generate_lead_scorecard(use_sample_data=False)
                
                if scorecard_df is not None and thresholds is not None and model_metrics is not None:
                    st.session_state.weights_df = scorecard_df
                    st.session_state.thresholds = thresholds
                    st.session_state.model_metrics = model_metrics
                    
                    # Display the model information
                    st.write("#### Feature Weights")
                    st.write("The model identified these features as significant predictors of conversion:")
                    st.dataframe(scorecard_df)
                
                    # Display the suggested thresholds
                    st.write("#### Score Thresholds")
                    st.write("Based on the analysis, here are the recommended score thresholds for lead classification:")
                    
                    # Create threshold table
                    threshold_data = []
                    max_score = sum(scorecard_df['Points'])
                    
                    for category, threshold in thresholds.items():
                        threshold_data.append({
                            "Category": category,
                            "Minimum Score": threshold,
                            "Score Range": f"{threshold}+ points"
                        })
                    
                    threshold_df = pd.DataFrame(threshold_data)
                    st.dataframe(threshold_df, use_container_width=True)
                    
                    # Display model performance metrics
                    st.write("#### Model Performance")
                    
                    # Create 2 columns for metrics and visualization
                    metrics_col, viz_col = st.columns(2)
                    
                    with metrics_col:
                        # Display ROC AUC and other metrics
                        st.metric("ROC AUC Score", f"{model_metrics['roc_auc']:.3f}")
                        st.metric("Precision-Recall AUC", f"{model_metrics['pr_auc']:.3f}")
                        
                        # Confusion matrix
                        cm = model_metrics['confusion_matrix']
                        st.write("**Confusion Matrix at Optimal Threshold:**")
                        cm_df = pd.DataFrame(
                            cm, 
                            index=["Actual: Lost", "Actual: Won"],
                            columns=["Predicted: Lost", "Predicted: Won"]
                        )
                        st.dataframe(cm_df)
                        
                        # Calculate and display precision, recall, etc.
                        tn, fp, fn, tp = cm.ravel()
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                        accuracy = (tp + tn) / (tp + tn + fp + fn)
                        
                        metrics_data = {
                            "Metric": ["Precision", "Recall", "F1 Score", "Accuracy"],
                            "Value": [f"{precision:.3f}", f"{recall:.3f}", f"{f1:.3f}", f"{accuracy:.3f}"]
                        }
                        st.dataframe(pd.DataFrame(metrics_data))
                    
                    with viz_col:
                        # Plot ROC curve using our evaluation module
                        fig, ax = plt.subplots(figsize=(8, 6))
                        plot_roc_curve(model_metrics, ax)
                        st.pyplot(fig)
                        
                        # Plot score distributions using our evaluation module
                        fig, ax = plt.subplots(figsize=(8, 6))
                        plot_score_distributions(model_metrics, ax)
                        st.pyplot(fig)
                        
                        # Add Precision-Recall curve
                        fig, ax = plt.subplots(figsize=(8, 6))
                        plot_precision_recall_curve(model_metrics, ax)
                        st.pyplot(fig)
                        
                        # Add interactive threshold slider for fine-tuning
                        st.write("#### Interactive Threshold Tuning")
                        st.write("Adjust the threshold to see how it affects the model's predictions:")
                        
                        # Get min and max values from model_metrics
                        y_pred_proba = model_metrics['y_pred_proba']
                        min_score = float(max(0.001, y_pred_proba.min()))
                        max_score = float(min(0.999, y_pred_proba.max()))
                        
                        # Calculate metrics for different thresholds
                        custom_threshold = st.slider(
                            "Select threshold", 
                            min_value=min_score,
                            max_value=max_score,
                            value=float(model_metrics['best_threshold']),
                            step=0.01
                        )
                        
                        # Add business category selection
                        st.write("#### Lock in Business Thresholds")
                        st.write("Set your desired thresholds for each lead category:")
                        
                        hot_threshold = st.number_input(
                            "Hot Lead Threshold", 
                            min_value=0.1, 
                            max_value=0.9, 
                            value=float(model_metrics['best_threshold']), 
                            step=0.01,
                            help="Leads above this score will be categorized as 'Hot'"
                        )
                        
                        warm_threshold = st.number_input(
                            "Warm Lead Threshold", 
                            min_value=0.05, 
                            max_value=hot_threshold - 0.01, 
                            value=max(0.05, float(model_metrics['best_threshold']) / 2), 
                            step=0.01,
                            help="Leads above this score (but below Hot) will be categorized as 'Warm'"
                        )
                        
                        cool_threshold = st.number_input(
                            "Cool Lead Threshold", 
                            min_value=0.01, 
                            max_value=warm_threshold - 0.01, 
                            value=max(0.01, float(model_metrics['best_threshold']) / 4), 
                            step=0.01,
                            help="Leads above this score (but below Warm) will be categorized as 'Cool'"
                        )
                        
                        # Code snippet for implementation
                        if st.checkbox("Show implementation code"):
                            st.code("""
# Zapier Code Step (JavaScript):
const score = parseFloat(inputData.lead_score);
let category = 'Cold';

if (score >= """ + str(hot_threshold) + """) {
  category = 'Hot';
} else if (score >= """ + str(warm_threshold) + """) {
  category = 'Warm';
} else if (score >= """ + str(cool_threshold) + """) {
  category = 'Cool';
}

return {
  category: category,
  score: score,
  is_priority: category === 'Hot'
};
                            """, language="javascript")
                        
                        # Add section for exporting model configuration
                        st.write("#### Operationalize Your Model")
                        
                        # Create tabs for different implementations
                        impl_tab1, impl_tab2, impl_tab3 = st.tabs(["Zapier", "Streak", "Performance Monitoring"])
                        
                        with impl_tab1:
                            st.write("##### Zapier Implementation")
                            st.write("""
                            1. Set up a Zapier trigger for new leads
                            2. Add a 'Code' step using the JavaScript above
                            3. Use 'Paths' to route leads based on category
                            4. Set up different actions for each category (priority Slack alerts, emails, etc.)
                            """)
                        
                        with impl_tab2:
                            st.write("##### Streak Implementation")
                            st.write("""
                            1. Add a custom field in Streak for 'Lead Score' (number) and 'Lead Category' (dropdown)
                            2. Use the Streak API or Google Sheets integration to calculate and update scores
                            3. Create Streak Workflows that trigger based on Lead Category
                            4. Set up differentiated follow-up tasks by category (e.g., "Call Hot Leads within 30 minutes")
                            """)
                            
                            # Template for Google Sheets formula
                            st.code("""
=IF(ISBLANK(A2), "", 
  IF(A2 >= """ + str(hot_threshold) + """, "Hot",
    IF(A2 >= """ + str(warm_threshold) + """, "Warm",
      IF(A2 >= """ + str(cool_threshold) + """, "Cool", "Cold")
    )
  )
)
                            """, language="excel")
                        
                        with impl_tab3:
                            st.write("##### Performance Monitoring")
                            st.write("""
                            To track model performance over time:
                            
                            1. Log every scored lead with its prediction and eventual outcome
                            2. Re-calibrate your model monthly or quarterly as booking patterns evolve
                            3. A/B test your new model against previous approaches
                            """)
                            
                            # Create a downloadable performance tracking template
                            monitoring_df = pd.DataFrame({
                                'lead_id': ['example_1', 'example_2'],
                                'lead_score': [0.75, 0.35],
                                'category': ['Hot', 'Warm'],
                                'date_scored': [datetime.datetime.now(), datetime.datetime.now()],
                                'actual_outcome': ['Won', 'Lost'],
                                'time_to_close_days': [14, 30],
                                'deal_value': [2500, 0]
                            })
                            
                            csv = monitoring_df.to_csv(index=False)
                            b64 = base64.b64encode(csv.encode()).decode()
                            href = f'<a href="data:file/csv;base64,{b64}" download="lead_performance_tracker.csv">Download Performance Tracking Template</a>'
                            st.markdown(href, unsafe_allow_html=True)
                        
                        # Get custom threshold metrics using the evaluation module
                        y_true = model_metrics['y_true']
                        y_pred_proba = model_metrics['y_pred_proba']
                        custom_metrics = get_custom_threshold_metrics(y_true, y_pred_proba, custom_threshold)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Display confusion matrix
                            cm_df = pd.DataFrame(
                                custom_metrics['confusion_matrix'], 
                                index=["Actual: Lost", "Actual: Won"],
                                columns=["Predicted: Lost", "Predicted: Won"]
                            )
                            st.write("**Confusion Matrix at Selected Threshold:**")
                            st.dataframe(cm_df)
                        
                        with col2:
                            # Show metrics
                            metrics_data = {
                                "Metric": ["Precision", "Recall", "F1 Score", "Accuracy", "Specificity"],
                                "Value": [
                                    f"{custom_metrics['precision']:.3f}", 
                                    f"{custom_metrics['recall']:.3f}", 
                                    f"{custom_metrics['f1']:.3f}", 
                                    f"{custom_metrics['accuracy']:.3f}",
                                    f"{custom_metrics['specificity']:.3f}"
                                ]
                            }
                            st.dataframe(pd.DataFrame(metrics_data))
                            
                            # Recommendations based on metrics
                            if custom_metrics['precision'] < 0.3:
                                st.warning("⚠️ Low precision! Consider increasing the threshold.")
                            if custom_metrics['recall'] < 0.3:
                                st.warning("⚠️ Low recall! Consider decreasing the threshold.")
                            
                            # Display actual counts
                            count_data = {
                                "Category": ["True Positives", "False Positives", "True Negatives", "False Negatives"],
                                "Count": [
                                    custom_metrics['tp'],
                                    custom_metrics['fp'],
                                    custom_metrics['tn'],
                                    custom_metrics['fn']
                                ]
                            }
                            st.write("**Prediction Counts:**")
                            st.dataframe(pd.DataFrame(count_data))
                
                # Section 1.5: Contact Matching Analysis
                st.markdown("### 1.5 📱 Contact Matching Analysis")
                
                # Import necessary functions
                from conversion import analyze_phone_matches, analyze_prediction_counts
                
                with st.expander("Lead-to-Booking Phone Matching", expanded=True):
                    st.markdown("""
                    <div class="info-text">
                    This feature matches leads who inquired about an event with the eventual booking records.
                    Tracking the customer journey from inquiry to booking helps understand which leads convert.
                    </div>
                    """, unsafe_allow_html=True)
                    
                    tab1, tab2, tab3 = st.tabs(["Phone Matching", "Area Code Analysis", "Prediction Counts"])
                    
                    with tab1:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if st.button("Run Phone Number Matching"):
                                with st.spinner("Matching inquiries with bookings..."):
                                    try:
                                        matches, total_leads, total_ops = process_phone_matching()
                                        
                                        # Calculate matching rate
                                        if total_leads > 0:
                                            match_rate = (matches / total_leads) * 100
                                        else:
                                            match_rate = 0
                                            
                                        st.success(f"Found {matches} matches out of {total_leads} leads ({match_rate:.1f}%)")
                                        
                                        # Visualize matches
                                        fig, ax = plt.subplots(figsize=(8, 4))
                                        labels = ['Matched', 'Unmatched']
                                        sizes = [matches, total_leads - matches]
                                        colors = ['#1E88E5', '#BBDEFB']
                                        
                                        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
                                        ax.axis('equal')
                                        st.pyplot(fig)
                                    except Exception as e:
                                        st.error(f"Error running phone matching: {e}")
                        
                        with col2:
                            st.markdown("""
                            #### How it works:
                            
                            The system matches leads to bookings using these methods in priority order:
                            
                            1. **Box Key Match**: Direct ID matching
                            2. **Email Match**: Same email address used for inquiry and booking
                            3. **Phone Match**: Same phone number used (after normalization)
                            
                            This helps connect the dots between an initial inquiry and the final booking.
                            """)
                    
                    with tab2:
                        st.subheader("Area Code to State Conversion Analysis")
                        st.write("How often area-code matching predicts wins:")
                        
                        if st.session_state.processed_df is not None:
                            # Analyze phone matches
                            match_conversion, match_counts = analyze_phone_matches(st.session_state.processed_df)
                            
                            if not match_conversion.empty:
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.write("**Conversion Rates by Phone-State Match:**")
                                    st.table(match_conversion)
                                
                                with col2:
                                    st.write("**Count by Phone-State Match:**")
                                    
                                    # Visualize counts
                                    fig, ax = plt.subplots(figsize=(8, 4))
                                    ax.bar(match_counts['Phone-State Match'].astype(str), match_counts['Count'], color=['#1E88E5', '#BBDEFB'])
                                    plt.ylabel('Count')
                                    plt.title('Number of leads by phone-state match')
                                    st.pyplot(fig)
                            else:
                                st.info("No phone match analysis available. Make sure the data includes phone numbers and state information.")
                        else:
                            st.info("Load data first to analyze area code matches.")
                    
                    with tab3:
                        st.subheader("Prediction Counts at Thresholds")
                        st.write("Distribution of leads across Hot/Warm/Cool/Cold categories:")
                        
                        if st.session_state.model_metrics and 'y_pred_proba' in st.session_state.model_metrics:
                            # Get prediction counts
                            y_scores = st.session_state.model_metrics['y_pred_proba']
                            counts_df = analyze_prediction_counts(y_scores, st.session_state.thresholds)
                            
                            if not counts_df.empty:
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.table(counts_df)
                                
                                with col2:
                                    # Plot as bar chart
                                    st.bar_chart(counts_df.set_index('Category'))
                            else:
                                st.info("No prediction data available for analysis.")
                        else:
                            st.info("Train model first to analyze prediction distribution.")
                
                # Section 2: Lead Scoring Calculator
                st.markdown("### 2. Lead Scoring Calculator")
                st.write("Use this tool to score a new lead and determine its likelihood to convert based on your historical data.")
                
                # Store model in session state for easier access
                st.session_state['weights_df'] = scorecard_df
                st.session_state['thresholds'] = thresholds
            
            # Check if model is available
            if 'weights_df' in st.session_state and 'thresholds' in st.session_state:
                # Fetch the learned points & thresholds
                weights = st.session_state['weights_df'].set_index('Feature')['Points']
                thresholds = st.session_state['thresholds']
                max_score = int(weights.sum()) if not weights.empty else 100
                
                # Direct input fields (no form needed)
                st.subheader("Enter Lead Details")
                
                # Use two columns for inputs
                col1, col2 = st.columns(2)
                
                with col1:
                    guests = st.number_input("Number of Guests", min_value=0, value=100)
                    days_until = st.number_input("Days Until Event", min_value=0, value=30)
                    days_since = st.number_input("Days Since Inquiry", min_value=0, value=1)
                
                with col2:
                    bartenders = st.number_input("Bartenders Needed", min_value=0, value=2)
                    is_corp = st.selectbox("Is Corporate Event?", ["No", "Yes"]) == "Yes"
                    referral = st.selectbox("Referral Tier (1-3)", [1, 2, 3])
                    phone_match = st.selectbox("Phone Area Code Matches State?", ["No", "Yes"]) == "Yes"
                
                # Build the feature dict (must match derive_scorecard feature names)
                feature_vals = {
                    'NumberOfGuests': guests,
                    'DaysUntilEvent': days_until,
                    'DaysSinceInquiry': days_since,
                    'BartendersNeeded': bartenders,
                    'IsCorporate': int(is_corp),
                    'ReferralTier': referral,
                    'PhoneMatch': int(phone_match)
                }
                
                # Compute the total score
                score = 0
                feature_contributions = {}
                
                for feature, value in feature_vals.items():
                    if feature in weights.index:
                        # Get the weight for this feature
                        weight = weights.get(feature, 0)
                        
                        # Apply normalization for numeric features
                        if feature == 'NumberOfGuests':
                            norm_value = min(float(value) / 100.0, 1.0)
                            contribution = weight * norm_value
                        elif feature == 'DaysUntilEvent':
                            norm_value = min(float(value) / 365.0, 1.0)
                            contribution = weight * norm_value
                        elif feature == 'DaysSinceInquiry':
                            norm_value = min(float(value) / 30.0, 1.0)
                            contribution = weight * norm_value
                        elif feature == 'BartendersNeeded':
                            norm_value = min(float(value) / 10.0, 1.0)
                            contribution = weight * norm_value
                        elif feature == 'ReferralTier':
                            norm_value = float(value) / 3.0
                            contribution = weight * norm_value
                        else:
                            # Boolean features
                            contribution = weight * float(value)
                        
                        score += contribution
                        feature_contributions[feature] = contribution
                
                # Bucket into categories
                thresholds_list = sorted([(k, v) for k, v in thresholds.items()], key=lambda x: x[1], reverse=True)
                category = "❄️ Cold"
                for cat, threshold in thresholds_list:
                    if score >= threshold:
                        category = f"{'🔥' if cat.lower() == 'hot' else '👍' if cat.lower() == 'warm' else '🙂'} {cat}"
                        break
                
                # Display the results
                st.markdown("### Lead Score Results")
                
                # Create columns for display
                result_col1, result_col2 = st.columns([1, 2])
                
                with result_col1:
                    # Display numeric score and category
                    st.metric("Lead Score", int(score))
                    st.markdown(f"**Category:** {category}")
                    st.markdown(f"**Total Score:** {int(score)} / {max_score} points")
                    
                    # Create a progress bar visualization
                    score_percent = min(100, max(0, (score / max_score) * 100)) if max_score > 0 else 0
                    st.progress(score_percent/100)
                
                with result_col2:
                    # Create a bar chart of feature impacts
                    contrib_df = pd.DataFrame({
                        'Feature': list(feature_contributions.keys()),
                        'Impact': list(feature_contributions.values())
                    }).sort_values(by='Impact', ascending=False)
                    
                    # Use matplotlib for simpler visualization
                    fig, ax = plt.subplots(figsize=(8, 4))
                    bars = ax.barh(contrib_df['Feature'], contrib_df['Impact'])
                    
                    # Color bars based on contribution (positive/negative)
                    for i, bar in enumerate(bars):
                        if contrib_df['Impact'].iloc[i] > 0:
                            bar.set_color('green')
                        else:
                            bar.set_color('red')
                            
                    ax.set_title('Feature Impact on Lead Score')
                    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
                    st.pyplot(fig)
                    
                # Add interpretation and recommendations
                st.markdown("#### Interpretation")
                
                if category.lower().find('hot') >= 0:
                    st.success("This lead has a very high probability of converting based on your historical data. Prioritize immediate follow-up.")
                elif category.lower().find('warm') >= 0:
                    st.info("This lead shows good potential and should be followed up promptly.")
                elif category.lower().find('cool') >= 0:
                    st.warning("This lead has moderate potential. Consider standard follow-up procedures.")
                else:
                    st.error("This lead has lower conversion potential based on your historical patterns.")
                    
                st.markdown("#### Recommendations")
                
                if category.lower().find('hot') >= 0:
                    st.markdown("• 📱 **Call immediately**: High probability of closing")
                    st.markdown("• 💰 **Offer premium package**: Good candidate for upselling")
                    st.markdown("• 🤝 **Schedule site visit**: Ready to make a decision")
                elif category.lower().find('warm') >= 0:
                    st.markdown("• 📱 **Follow up within 24 hours**: Solid potential")
                    st.markdown("• 📊 **Send detailed proposal**: Ready for specific details")
                    st.markdown("• 🔗 **Provide references**: May need social proof")
                elif category.lower().find('cool') >= 0:
                    st.markdown("• 📧 **Email follow-up**: Moderate potential")
                    st.markdown("• ❓ **Address objections**: May have hesitations")
                    st.markdown("• 💡 **Highlight differentiators**: Needs convincing")
                else:  # Cold
                    st.markdown("• 📅 **Schedule for later follow-up**: Low immediate potential")
                    st.markdown("• 📊 **Send general information**: May need education")
                    st.markdown("• 💰 **Consider promotional offer**: May need incentive")
            else:
                st.warning("Please generate a lead scoring model first by clicking the button above.")
                st.info("The model will be built using your historical data to predict which leads are most likely to convert.")

        except Exception as e:
            st.error(f"Error in lead scoring functionality: {str(e)}")
    
    with tab4:
        try:
            # Display the raw data
            st.subheader("Raw Data")
            st.dataframe(filtered_df)
            
            # Add download button for filtered data
            csv = filtered_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="filtered_data.csv">Download CSV File</a>'
            st.markdown(href, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error displaying raw data: {str(e)}")
    
    # Key Findings Tab
    with tab5:
        st.title("📈 Report of Key Findings")
        
        # Check if we have model data in session state
        if 'model_metrics' in st.session_state and 'weights_df' in st.session_state and 'thresholds' in st.session_state:
            try:
                # Get data for findings
                df = filtered_df  # Use the current filtered dataframe
                y_scores = st.session_state.model_metrics.get('y_pred_proba')
                thresholds = st.session_state.thresholds
                
                # Generate dynamic findings
                findings = generate_findings(df, y_scores, thresholds)
                
                # Display the findings
                for finding in findings:
                    st.markdown(f"• {finding}")
                
                st.info("These findings are dynamically generated from your current data and will update as your data changes.")
            except Exception as e:
                st.error(f"Error generating findings: {str(e)}")
                st.info("Try clicking 'Generate Lead Scoring Model' on the Lead Scoring tab first if you haven't already.")
        else:
            st.info("Please select a data source and click 'Generate Lead Scoring Model' on the Lead Scoring tab to see key findings based on your data.")
            
            # Show example findings to demonstrate how the tab will look
            st.subheader("Example Key Findings")
            st.markdown("""
            The Key Findings tab will display insights like these, but specific to your data:
            
            • **Urgency:** Leads closing within 7 days convert at 45%, vs. those >30 days at 10%.
            • **Geography:** Region A leads close at 38%, while Region B at 18%.
            • **Seasonality:** July month has 32% conversion rate, lowest is January at I4%.
            • **Event Type:** Corporate events convert at 28%, Social events at 20%.
            • **Phone‐Match:** Local numbers convert at 16% vs. non‐local at 10%.
            • **Time to Conversion:** Average: 12.5 days, Median: 8.0 days.
            • **Event Type Conversion Speed:** Corporate events convert fastest (8.3 days), while Weddings take longest (16.7 days).
            • **Model AUC:** ROC=0.835, PR=0.574.
            • **Buckets:** 2,458 Hot, 3,721 Warm, 8,942 Cool, 12,311 Cold.
            """)
            st.warning("These are example findings. Generate a model to see findings specific to your business.")
    
    # Explanations Tab
    with tab6:
        st.title("📖 Dashboard Explanations")

        st.header("1. Conversion Summary")
        st.markdown("""
        - **Total Leads**: Number of distinct form submissions processed.  
        - **Won Deals**: Leads you've marked "Definite" or "Tentative."  
        - **Lost Deals**: Leads marked "Lost."  
        - **Conversion Rate** = Won ÷ (Won + Lost).  
          
        _Why it matters:_ Gives you a quick high-level view of how healthy your pipeline is.
        """)

        st.header("2. Conversion by Category")
        st.markdown("""
        Here we break out historical **win‐rates** by:
        - **Event Type** (Corporate vs. Wedding vs. Party)  
        - **Referral Source** (Referral vs. Facebook vs. Google)  
        - **State / Region**  
        - **Guests / Days-to-Event** buckets  

        _Why it matters:_ Pinpoints which lead characteristics produce the strongest closes so you can prioritize similar prospects.
        """)

        st.header("3. Feature Correlation")
        st.markdown("""
        Displays the **absolute Pearson correlations** between each numeric feature and outcome (0=Lost, 1=Won).  
        - Values near **1.0** indicate very strong linear relationships.  
        - Near **0** means little predictive power.

        _Why it matters:_ Helps you decide which features deserve the most weight in your scorecard.
        """)

        st.header("4. Lead Scoring Model")
        st.markdown("""
        We fit a **logistic regression** (and checked with a Random Forest) to derive:
        - **Coefficients** (direction & magnitude of impact on close probability)  
        - **Point values** (normalized, integer weights)  
        - **Score thresholds** (Hot/Warm/Cool/Cold cut-offs)  

        _Why it matters:_ Translates complex statistics into a simple rubric your team can use in real time.
        """)

        st.header("5. Model Performance & Threshold Tuning")
        st.markdown("""
        - **ROC AUC** measures ranking ability (higher is better, 0.5 = random).  
        - **PR AUC** measures precision/recall trade-off under class imbalance.  
        - **Threshold slider** lets you choose the operating point—higher recall if you want more leads, higher precision if you want fewer but stronger leads.  

        _Why it matters:_ Ensures you pick cut-offs that align with your capacity and risk tolerance.
        """)

        st.header("6. Contact-Match Analysis")
        st.markdown("""
        Compares win-rates for leads whose **phone area code** matches the state they specified vs. those that don't.  
        - Local numbers often signal higher trust and close rates.  
        - Mismatches can hint at disposable or non-local leads.

        _Why it matters:_ A quick proxy for lead quality — helps you spot "ghost" or low-fi leads.
        """)

        st.header("7. Time to Conversion Analysis")
        st.markdown("""
        Analyzes how long it typically takes to convert leads from first inquiry to won deal.
        - **Average Days**: Mean time from inquiry to winning the deal
        - **Median Days**: Middle value in the conversion timeline (less affected by outliers)
        - **By Event Type**: Shows which events typically convert faster or slower
        - **By Booking Type**: Breaks down conversion times by booking category
        - **Distribution**: Shows what percentage of deals close within specific time frames

        _Why it matters:_ Helps with forecasting, setting appropriate follow-up schedules, and prioritizing faster-converting event types.
        """)

        st.header("8. Lead Scoring Calculator")
        st.markdown("""
        Enter hypothetical lead details to get:
        - A **numeric score** (sum of data-driven points)  
        - A **category** (Hot, Warm, Cool, Cold)  

        _Why it matters:_ Gives your sales team a consistent, transparent way to prioritize follow-ups on new inbound leads.
        """)

else:
    # Display instructions when no data is loaded
    st.info("Please select a data source to begin analysis.")
    
    # Example layout with placeholder visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Conversion by Category")
        st.write("Select data source to see conversion rates by different categories.")
        
        # Placeholder for empty chart
        fig1, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "No data available", 
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes,
                fontsize=16)
        ax.axis('off')
        st.pyplot(fig1)
    
    with col2:
        st.subheader("Feature Correlation")
        st.write("Select data source to see feature correlations with outcome.")
        
        # Placeholder for empty chart
        fig2, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "No data available", 
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes,
                fontsize=16)
        ax.axis('off')
        st.pyplot(fig2)

# Footer information
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    """
    This dashboard analyzes sales conversion data from Streak exports, 
    helping you identify patterns and optimize your lead scoring process.
    """
)
st.sidebar.markdown("### Help")
st.sidebar.info(
    """
    1. Select a data source (upload or database)
    2. Explore the different analysis tabs
    3. Generate a lead scoring model to predict conversion likelihood
    """
)