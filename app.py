import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
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
from segmentation import segment_leads, plot_clusters, plot_cluster_conversion_rates, plot_feature_importance_by_cluster
from advanced_analytics import run_all_analytics, plot_conversion_by_category

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
    
    # Add a summary of the data source
    st.subheader("Data Source Summary")
    
    # Determine data source type
    if data_source == "Sample Data":
        source_type = "Sample Data"
    elif data_source == "Database":
        source_type = "Database"
    elif data_source == "Upload CSV Files":
        source_type = "CSV Upload"
    else:
        source_type = "Unknown"
    
    # Create dataframe with data source summary
    df = filtered_df
    won_count = df['outcome'].sum()
    lost_count = len(df) - won_count
    conversion_rate = (won_count / len(df)) * 100 if len(df) > 0 else 0
    
    summary_data = {
        'Source': [source_type],
        'Total Leads': [len(df)],
        'Won Deals': [won_count],
        'Lost Deals': [lost_count],
        'Conversion Rate': [f"{conversion_rate:.1f}%"]
    }
    summary_df = pd.DataFrame(summary_data)
    
    # Display the summary
    st.dataframe(summary_df, use_container_width=True)
    
    # Create a date range filter
    st.subheader("Date Range Filter")
    
    # Calculate min and max dates from data
    if 'inquiry_date' in df.columns:
        df['inquiry_date'] = pd.to_datetime(df['inquiry_date'], errors='coerce')
        min_date = df['inquiry_date'].min().date() if not pd.isna(df['inquiry_date'].min()) else datetime.date.today() - datetime.timedelta(days=365)
        max_date = df['inquiry_date'].max().date() if not pd.isna(df['inquiry_date'].max()) else datetime.date.today()
    else:
        min_date = datetime.date.today() - datetime.timedelta(days=365)
        max_date = datetime.date.today()
    
    # Create date range filter interface
    date_col1, date_col2 = st.columns(2)
    
    with date_col1:
        start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
    
    with date_col2:
        end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
    
    # Add sparkline of daily lead volume
    if 'inquiry_date' in df.columns:
        # Create daily lead count
        df_daily = df.copy()
        df_daily['inquiry_date'] = pd.to_datetime(df_daily['inquiry_date']).dt.date
        daily_counts = df_daily.groupby('inquiry_date').size().reset_index(name='count')
        
        # Convert to datetime for better plotting
        daily_counts['inquiry_date'] = pd.to_datetime(daily_counts['inquiry_date'])
        
        # Filter for selected date range
        daily_counts = daily_counts[
            (daily_counts['inquiry_date'].dt.date >= start_date) & 
            (daily_counts['inquiry_date'].dt.date <= end_date)
        ]
        
        # Create sparkline if we have data
        if not daily_counts.empty:
            st.subheader("Daily Lead Volume")
            
            # Create sparkline chart
            fig, ax = plt.subplots(figsize=(8, 2))
            ax.plot(daily_counts['inquiry_date'], daily_counts['count'], color='#0068c9', linewidth=2)
            
            # Fill the area under the line
            ax.fill_between(daily_counts['inquiry_date'], daily_counts['count'], alpha=0.2, color='#0068c9')
            
            # Remove spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.get_yaxis().set_ticks([])
            
            # Format the dates on the x-axis
            ax.xaxis.set_major_formatter(DateFormatter('%b %d'))
            plt.xticks(rotation=45)
            
            # Add annotations for min and max values
            min_idx = daily_counts['count'].idxmin()
            max_idx = daily_counts['count'].idxmax()
            
            if min_idx is not None and max_idx is not None:
                min_date = daily_counts.iloc[min_idx]['inquiry_date']
                min_count = daily_counts.iloc[min_idx]['count']
                max_date = daily_counts.iloc[max_idx]['inquiry_date']
                max_count = daily_counts.iloc[max_idx]['count']
                
                # Add min marker
                ax.scatter(min_date, min_count, color='#FF474C', s=50, zorder=5)
                ax.annotate(f"{min_count:,}", 
                           (min_date, min_count),
                           xytext=(0, -15),
                           textcoords='offset points',
                           ha='center',
                           fontsize=9,
                           color='#FF474C')
                
                # Add max marker
                ax.scatter(max_date, max_count, color='#00CC96', s=50, zorder=5)
                ax.annotate(f"{max_count:,}", 
                           (max_date, max_count),
                           xytext=(0, 10),
                           textcoords='offset points',
                           ha='center',
                           fontsize=9,
                           color='#00CC96')
            
            # Add statistics
            total_leads = daily_counts['count'].sum()
            avg_daily = daily_counts['count'].mean()
            
            st.pyplot(fig)
            
            # Add metrics in small columns
            lead_col1, lead_col2, lead_col3 = st.columns(3)
            with lead_col1:
                st.metric("Total Leads", f"{total_leads:,}")
            with lead_col2:
                st.metric("Avg. Daily Leads", f"{avg_daily:.1f}")
            with lead_col3:
                st.metric("Days in Period", f"{len(daily_counts)}")
    
    # Apply date filter to the dataframe
    if 'inquiry_date' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df['inquiry_date'].dt.date >= start_date) & 
            (filtered_df['inquiry_date'].dt.date <= end_date)
        ]
        st.info(f"Filtered to {len(filtered_df)} leads from {start_date} to {end_date}")
    
    # Create tabs for different analysis views
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Conversion Analysis", 
        "🔍 Feature Correlation", 
        "🤖 Lead Scoring", 
        "🗃️ Raw Data",
        "📈 Key Findings",
        "🛈 Explanations",
        "🧩 Lead Personas",
        "📊 Advanced Analytics"
    ])
    
    with tab1:
        try:
            # Calculate conversion rates by different categories
            conversion_rates = calculate_conversion_rates(filtered_df)
            
            st.subheader("Conversion Rate Analysis")
            
            # Summary metrics
            overall_conversion = conversion_rates["overall"]["Conversion Rate"][0]
            
            # Calculate week-over-week change in conversion rate
            if 'inquiry_date' in filtered_df.columns:
                # Get current data
                current_df = filtered_df.copy()
                
                # Calculate previous week's date range
                current_start = pd.to_datetime(start_date)
                current_end = pd.to_datetime(end_date)
                date_range = (current_end - current_start).days
                
                # Previous period
                prev_end = current_start - pd.Timedelta(days=1)
                prev_start = prev_end - pd.Timedelta(days=date_range)
                
                # Filter for previous period
                prev_df = df[(df['inquiry_date'] >= prev_start) & 
                             (df['inquiry_date'] <= prev_end)]
                
                # Calculate previous conversion rate
                if len(prev_df) > 0:
                    prev_conversion = prev_df['outcome'].mean()
                    wow_change = overall_conversion - prev_conversion
                    
                    # Create metric with delta
                    st.metric(
                        "Overall Conversion Rate", 
                        f"{overall_conversion:.1%}", 
                        f"{wow_change:.1%}",
                        delta_color="normal"
                    )
                else:
                    # Just show current conversion if no previous data
                    st.metric("Overall Conversion Rate", f"{overall_conversion:.1%}")
            else:
                # Fallback if no date data
                st.metric("Overall Conversion Rate", f"{overall_conversion:.1%}")
            
            # Calculate and display time to conversion analysis
            st.subheader("⏱️ Time to Conversion Analysis")
            time_to_conversion = analyze_time_to_conversion(filtered_df)
            
            if time_to_conversion.get('error'):
                st.warning(f"Could not analyze time to conversion: {time_to_conversion.get('error')}")
            else:
                # Create metrics row
                metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
                
                with metric_col1:
                    st.metric("Average Days", f"{time_to_conversion.get('average_days', 0):.1f}")
                
                with metric_col2:
                    st.metric("Median Days", f"{time_to_conversion.get('median_days', 0):.1f}")
                
                with metric_col3:
                    st.metric("90th Percentile", f"{time_to_conversion.get('percentile_90', 0):.1f}")
                
                with metric_col4:
                    st.metric("Minimum Days", f"{time_to_conversion.get('min_days', 0)}")
                
                with metric_col5:
                    st.metric("Maximum Days", f"{time_to_conversion.get('max_days', 0)}")
                
                # Check for negative time anomalies
                if 'has_negative_days' in time_to_conversion and time_to_conversion['has_negative_days']:
                    st.warning(f"⚠️ **Data Quality Issue**: {time_to_conversion['negative_days_count']} records ({time_to_conversion['negative_days_percent']:.1f}%) show negative time-to-conversion. This indicates potential date entry errors in the source data.")
                
                # Display histogram of time to conversion
                time_col1, time_col2 = st.columns(2)
                
                with time_col1:
                    if 'days_data' in time_to_conversion and len(time_to_conversion['days_data']) > 0:
                        st.write("**Distribution of Time to Conversion**")
                        
                        # Create a violin plot for better density insights
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        # Filter out extreme outliers to make visualization cleaner
                        days_data = np.array(time_to_conversion['days_data'])
                        # Apply a reasonable cap if we have data
                        if len(days_data) > 0:
                            # Cap at 95th percentile + 1.5 IQR to remove extreme outliers
                            q1, q3 = np.percentile(days_data, [25, 75])
                            iqr = q3 - q1
                            upper_limit = q3 + 1.5 * iqr
                            upper_limit = max(30, min(upper_limit, 180))  # Ensure reasonable bounds
                            
                            # Bin data for display
                            filtered_data = days_data[days_data <= upper_limit]
                            
                            # Create violin plot
                            sns.violinplot(y=filtered_data, ax=ax, orient='h', color='#6495ED')
                            
                            # Add metrics as vertical lines
                            if 'median_days' in time_to_conversion:
                                median = time_to_conversion.get('median_days', 0)
                                if median <= upper_limit:
                                    ax.axhline(y=median, color='red', linestyle='--', alpha=0.7,
                                            label=f'Median: {median:.1f} days')
                            
                            if 'average_days' in time_to_conversion:
                                mean = time_to_conversion.get('average_days', 0)
                                if mean <= upper_limit:
                                    ax.axhline(y=mean, color='green', linestyle='--', alpha=0.7,
                                            label=f'Mean: {mean:.1f} days')
                            
                            # Add legend
                            ax.legend()
                            
                            # Set labels
                            ax.set_xlabel('Density')
                            ax.set_ylabel('Days to Conversion')
                            ax.set_title('Distribution of Time from Lead to Conversion')
                            
                            # Show statistics in a text box
                            props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
                            textstr = (f"n = {len(days_data)}\n"
                                    f"Mean = {time_to_conversion.get('average_days', 0):.1f}\n"
                                    f"Median = {time_to_conversion.get('median_days', 0):.1f}\n"
                                    f"90th % = {time_to_conversion.get('percentile_90', 0):.1f}")
                            
                            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                                verticalalignment='top', bbox=props)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                        else:
                            st.info("Not enough data for visualization")
                
                with time_col2:
                    # Display conversion time by booking type
                    if 'by_booking_type' in time_to_conversion and not time_to_conversion['by_booking_type'].empty:
                        st.write("**Time to Conversion by Booking Type**")
                        # Format columns for better display
                        display_df = time_to_conversion['by_booking_type'].copy()
                        display_df['mean'] = display_df['mean'].round(1)
                        display_df['median'] = display_df['median'].round(1)
                        
                        # Group low-volume types into 'Other'
                        MIN_COUNT = 3  # Minimum count to show as individual category
                        TOP_N = 5  # Show only top N categories
                        
                        # Copy the dataframe
                        grouped_df = display_df.copy()
                        
                        # Sort by count descending to get top types by volume
                        grouped_df = grouped_df.sort_values('count', ascending=False)
                        
                        # If we have more than TOP_N booking types, create an 'Other' category
                        if len(grouped_df) > TOP_N:
                            # Select top categories
                            top_categories = grouped_df.iloc[:TOP_N].copy()
                            
                            # Group everything else into 'Other'
                            other_categories = grouped_df.iloc[TOP_N:].copy()
                            
                            # Only create 'Other' if we have something to group
                            if len(other_categories) > 0:
                                # Calculate weighted mean, median, and sum of counts for 'Other'
                                other_count = other_categories['count'].sum()
                                other_mean = (other_categories['mean'] * other_categories['count']).sum() / other_count
                                other_median = np.median(
                                    np.repeat(
                                        other_categories['median'].values, 
                                        other_categories['count'].astype(int).values
                                    )
                                )
                                
                                # Calculate weighted standard deviation if available
                                if 'std' in other_categories.columns:
                                    other_std = np.sqrt(
                                        (other_categories['count'] * other_categories['std']**2).sum() / other_count
                                    )
                                else:
                                    other_std = 0
                                
                                # Create a row for 'Other'
                                other_row = pd.DataFrame({
                                    'booking_type': ['Other'],
                                    'mean': [other_mean],
                                    'median': [other_median],
                                    'count': [other_count]
                                })
                                
                                if 'std' in grouped_df.columns:
                                    other_row['std'] = other_std
                                
                                # Combine top categories with 'Other'
                                grouped_df = pd.concat([top_categories, other_row])
                            else:
                                grouped_df = top_categories
                        
                        # Format final dataframe for display
                        grouped_df.columns = ['Booking Type', 'Avg Days', 'Median Days'] + \
                                         (['Std Dev'] if 'std' in grouped_df.columns else []) + \
                                         ['Count']
                        
                        # Display the table
                        st.dataframe(grouped_df.sort_values(by='Avg Days'), use_container_width=True)
                        
                        # Create a chart visualization
                        if len(grouped_df) > 1:
                            fig, ax = plt.subplots(figsize=(10, max(3, min(8, len(grouped_df)))))
                            
                            # Plot in ascending order of average days
                            sorted_df = grouped_df.sort_values(by='Avg Days')
                            
                            # Generate the bar chart
                            bars = ax.barh(sorted_df['Booking Type'], sorted_df['Avg Days'], color='skyblue')
                            
                            # Add count annotations
                            for i, bar in enumerate(bars):
                                count = sorted_df.iloc[i]['Count']
                                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                                        f"n={count}", va='center', fontsize=8)
                            
                            ax.set_xlabel('Average Days to Conversion')
                            ax.set_title('Conversion Time by Booking Type')
                            plt.tight_layout()
                            st.pyplot(fig)
                    
                    # Display conversion time by standardized event type categories
                    if 'by_event_type' in time_to_conversion and not time_to_conversion['by_event_type'].empty:
                        st.write("**Time to Conversion by Event Type Category**")
                        # Format columns for better display
                        display_df = time_to_conversion['by_event_type'].copy()
                        display_df['mean'] = display_df['mean'].round(1)
                        display_df['median'] = display_df['median'].round(1)
                        
                        # Handle column names based on whether std is included
                        column_names = ['Event Type', 'Avg Days', 'Median Days']
                        if 'std' in display_df.columns:
                            display_df['std'] = display_df['std'].round(1)
                            column_names.append('Std Dev')
                        column_names.append('Count')
                        
                        display_df.columns = column_names
                        
                        # Add total row if there's more than one row
                        if len(display_df) > 1:
                            # Create dictionary for total row
                            total_dict = {
                                'Event Type': 'Overall Average',
                                'Avg Days': display_df['Avg Days'].mean().round(1),
                                'Median Days': display_df['Median Days'].median().round(1),
                                'Count': display_df['Count'].sum()
                            }
                            
                            # Add std dev to total if it exists
                            if 'Std Dev' in display_df.columns:
                                # Calculate weighted std dev
                                weights = display_df['Count'] / display_df['Count'].sum()
                                total_dict['Std Dev'] = (weights * display_df['Std Dev']).sum().round(1)
                            
                            # Create and append the total row
                            total_row = pd.DataFrame([total_dict])
                            display_df = pd.concat([display_df, total_row])
                        
                        # Sort by ascending average days
                        st.dataframe(display_df.sort_values(by='Avg Days'), use_container_width=True)
                        
                        # Create a bar chart with error bars for visual comparison of categories
                        if len(display_df) > 2:  # Only create chart if we have enough categories
                            chart_df = display_df[display_df['Event Type'] != 'Overall Average'].copy()
                            fig, ax = plt.subplots(figsize=(10, max(5, min(8, len(chart_df)))))
                            
                            # Plot in descending order of conversion time
                            sorted_df = chart_df.sort_values(by='Avg Days', ascending=False)
                            
                            # Define the y positions
                            y_pos = np.arange(len(sorted_df))
                            
                            # Check if we have std deviation data for error bars
                            if 'Std Dev' in sorted_df.columns:
                                # Create horizontal bar chart with error bars
                                bars = ax.barh(sorted_df['Event Type'], sorted_df['Avg Days'], 
                                              xerr=sorted_df['Std Dev'],
                                              color='skyblue', alpha=0.7, capsize=5,
                                              error_kw={'ecolor': 'darkblue', 'alpha': 0.6, 'capthick': 2})
                                
                                # Add a note about error bars
                                ax.text(0.05, 0.98, "Error bars: ±1 standard deviation", 
                                        transform=ax.transAxes, fontsize=8, verticalalignment='top',
                                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                            else:
                                # Simple bars without error bars
                                bars = ax.barh(sorted_df['Event Type'], sorted_df['Avg Days'], color='skyblue')
                            
                            # Add data labels with counts
                            for i, bar in enumerate(bars):
                                count = sorted_df.iloc[i]['Count']
                                # Position the text at the end of the bar plus any error bar
                                if 'Std Dev' in sorted_df.columns:
                                    x_pos = sorted_df.iloc[i]['Avg Days'] + sorted_df.iloc[i]['Std Dev'] + 0.5
                                else:
                                    x_pos = bar.get_width() + 0.5
                                
                                ax.text(x_pos, bar.get_y() + bar.get_height()/2, 
                                        f"n={count}", va='center', fontsize=8)
                            
                            ax.set_xlabel('Average Days to Conversion')
                            ax.set_title('Average Conversion Time by Event Type')
                            plt.tight_layout()
                            st.pyplot(fig)
                    
                    # Show detailed event types if available
                    if 'by_event_type_detailed' in time_to_conversion and not time_to_conversion['by_event_type_detailed'].empty:
                        with st.expander("View Detailed Event Types"):
                            st.write("**All Event Types (Uncategorized)**")
                            # Format columns for better display
                            display_df = time_to_conversion['by_event_type_detailed'].copy()
                            display_df['mean'] = display_df['mean'].round(1)
                            display_df['median'] = display_df['median'].round(1)
                            
                            # Handle column names based on whether std is included
                            column_names = ['Event Type', 'Avg Days', 'Median Days']
                            if 'std' in display_df.columns:
                                display_df['std'] = display_df['std'].round(1)
                                column_names.append('Std Dev')
                            column_names.append('Count')
                            
                            display_df.columns = column_names
                            
                            # Sort by count descending and only show types with at least 2 occurrences
                            filtered_df = display_df[display_df['Count'] >= 2].sort_values(by='Count', ascending=False)
                            if not filtered_df.empty:
                                st.dataframe(filtered_df, use_container_width=True)
                            else:
                                st.info("No detailed event types with sufficient data.")
                            
                            st.caption("Note: The main categories above group similar event types together for more meaningful analysis.")
            
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
        
    # Lead Personas Tab (New)
    with tab7:
        st.title("🧩 Lead Personas")
        
        # Information about what this tab does
        st.markdown("""
        This tab uses unsupervised machine learning to discover natural "lead personas" in your data. 
        These personas can help you understand different types of leads and their conversion patterns.
        
        ### What are Lead Personas?
        Lead personas are distinct groups of leads that share similar characteristics. By identifying these natural
        groupings, you can:
        
        - Discover unique lead segments you might not have been aware of
        - Apply targeted scoring models to each persona
        - Tailor your sales approach to each persona type
        - Identify high-converting personas to prioritize
        """)
        
        if st.session_state.processed_df is None:
            st.info("Please select a data source on the sidebar and process the data to use the Lead Personas feature.")
        else:
            col1, col2 = st.columns([2, 1])
            
            with col2:
                # Clustering parameters
                st.subheader("Clustering Settings")
                algorithm = st.selectbox(
                    "Clustering Algorithm", 
                    ["kmeans", "dbscan", "gmm"],
                    help="K-Means is recommended for most cases. DBSCAN is good for finding irregularly shaped clusters. GMM works well for overlapping clusters."
                )
                
                # Only show number of clusters option for k-means and GMM
                if algorithm in ["kmeans", "gmm"]:
                    n_clusters = st.slider(
                        "Number of Clusters", 
                        min_value=2, 
                        max_value=10, 
                        value=4,
                        help="Set to 0 to automatically determine the optimal number of clusters."
                    )
                    if n_clusters == 0:
                        n_clusters = None
                else:
                    n_clusters = None
                    
                # Analyze button
                if st.button("Generate Lead Personas"):
                    with st.spinner("Analyzing lead data and discovering natural segments..."):
                        # Run segmentation
                        segmentation_results = segment_leads(
                            st.session_state.processed_df, 
                            n_clusters=n_clusters,
                            algorithm=algorithm
                        )
                        
                        # Store results in session state
                        st.session_state.segmentation_results = segmentation_results
                        
                        # Show success message
                        if 'error' in segmentation_results:
                            st.error(segmentation_results['error'])
                        else:
                            st.success(f"Successfully identified {segmentation_results['n_clusters']} lead personas!")
            
            with col1:
                # Main content area - show results if available
                if 'segmentation_results' in st.session_state and st.session_state.segmentation_results:
                    results = st.session_state.segmentation_results
                    
                    # Check for error
                    if 'error' in results:
                        st.error(f"Error: {results['error']}")
                    else:
                        # Display results
                        st.subheader(f"Lead Persona Analysis ({results['n_clusters']} Segments)")
                        
                        # Show visualization tabs
                        vis_tab1, vis_tab2, vis_tab3 = st.tabs([
                            "Conversion by Persona", 
                            "Visual Segmentation", 
                            "Feature Importance"
                        ])
                        
                        with vis_tab1:
                            # Show conversion rates by cluster
                            st.write("### Conversion Rates by Lead Persona")
                            
                            conversion_fig = plot_cluster_conversion_rates(
                                results['conversion_by_cluster'],
                                results['cluster_names']
                            )
                            st.pyplot(conversion_fig)
                            
                            # Display the dataframe with conversion rates
                            st.write("#### Detailed Conversion Data")
                            display_df = results['conversion_by_cluster'].copy()
                            
                            # Add cluster names if available
                            if 'cluster_names' in results:
                                display_df['Persona'] = display_df['Cluster'].map(results['cluster_names'])
                                cols = ['Persona', 'Cluster', 'Conversion Rate', 'Count']
                            else:
                                cols = ['Cluster', 'Conversion Rate', 'Count']
                                
                            # Format conversion rate as percentage
                            display_df['Conversion Rate'] = display_df['Conversion Rate'].apply(
                                lambda x: f"{x:.1%}"
                            )
                            
                            st.dataframe(display_df[cols], use_container_width=True)
                        
                        with vis_tab2:
                            # Show 2D visualization of clusters
                            st.write("### Visual Cluster Segmentation")
                            st.write("This plot shows how leads are grouped into different personas (using dimensionality reduction to visualize in 2D).")
                            
                            cluster_fig = plot_clusters(
                                results['reduced_data'], 
                                results['clusters'],
                                f"Lead Personas ({algorithm.upper()})"
                            )
                            st.pyplot(cluster_fig)
                        
                        with vis_tab3:
                            # Show feature importance by cluster
                            st.write("### Feature Importance by Persona")
                            st.write("This heatmap shows what features differentiate each persona. Darker colors indicate more distinctive values.")
                            
                            feature_fig = plot_feature_importance_by_cluster(
                                results['cluster_profiles'],
                                results['feature_names']
                            )
                            st.pyplot(feature_fig)
                            
                        # Persona characteristics section
                        st.write("### Lead Persona Characteristics")
                        
                        # For each cluster, show key characteristics
                        for cluster in sorted(results['cluster_profiles'].index):
                            cluster_name = results['cluster_names'].get(cluster, f"Persona {cluster}")
                            
                            # Get the profile for this cluster
                            profile = results['cluster_profiles'].loc[cluster]
                            
                            # Get conversion rate for this cluster
                            conv_rate = results['conversion_by_cluster'][
                                results['conversion_by_cluster']['Cluster'] == cluster
                            ]['Conversion Rate'].values[0]
                            
                            # Create expandable section for this persona
                            with st.expander(f"{cluster_name} (Conv. Rate: {conv_rate:.1%})"):
                                # Create two columns within the expander
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.write("**Key Metrics:**")
                                    metrics = []
                                    
                                    # Add numerical features
                                    for feature in ['days_until_event', 'number_of_guests', 'bartenders_needed']:
                                        if feature in profile:
                                            feature_name = feature.replace('_', ' ').title()
                                            metrics.append(f"- {feature_name}: {profile[feature]:.1f}")
                                    
                                    # Display metrics
                                    st.markdown("\n".join(metrics))
                                
                                with col2:
                                    st.write("**Common Characteristics:**")
                                    chars = []
                                    
                                    # Look for categorical features (which would be one-hot encoded)
                                    for col in profile.index:
                                        if '_' in col and profile[col] > 0.3:  # Only include if >30% of cluster has this value
                                            feature, value = col.split('_', 1)
                                            chars.append(f"- {feature.title()}: {value}")
                                    
                                    # Display characteristics or a message if none found
                                    if chars:
                                        st.markdown("\n".join(chars))
                                    else:
                                        st.write("No distinctive categorical characteristics found.")
                else:
                    # Show placeholder content
                    st.info("Click 'Generate Lead Personas' to analyze your data and discover natural lead segments.")
                    
                    # Show example content
                    example_col1, example_col2 = st.columns([1, 1])
                    with example_col1:
                        st.markdown("""
                        #### Example Personas You Might Find:
                        
                        **High-Value Urgent Events (26% conversion)**
                        - Large guest count (150+)
                        - Short lead time (<30 days)
                        - Corporate booking type
                        
                        **Small Social Gatherings (8% conversion)**
                        - Small guest count (<50)
                        - Medium lead time (30-90 days)
                        - Social event types
                        """)
                    
                    with example_col2:
                        st.markdown("""
                        #### How to Use Personas:
                        
                        1. **Identify top-converting segments** to prioritize in your sales process
                        2. **Create persona-specific lead scoring** models for more accurate predictions
                        3. **Customize your follow-up approach** based on persona characteristics
                        4. **Train sales team** on the unique needs of each persona
                        """)
                        
    # Advanced Analytics Tab
    with tab8:
        st.title("📊 Advanced Analytics")
        
        # Information about this tab
        st.markdown("""
        This tab provides deeper insights into conversion patterns across various dimensions of your business.
        
        ### What's Included
        - **Referral Source Analysis**: Find your highest-converting referral channels
        - **Marketing Source Analysis**: Measure which marketing efforts pay off
        - **Booking Type Performance**: Identify your most profitable service types
        - **Price Per Guest Analysis**: Determine optimal pricing strategies
        - **Event Seasonality**: Discover monthly and day-of-week patterns
        - **Staff Ratio Impact**: Find the optimal staffing level for conversions
        """)
        
        if st.session_state.processed_df is None:
            st.info("Please select a data source on the sidebar and process the data to use the Advanced Analytics features.")
        else:
            # Run analytics
            with st.spinner("Running advanced analytics..."):
                analytics_results = run_all_analytics(st.session_state.processed_df)
            
            # Create sections for each analytics type
            st.header("Source & Channel Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                # Referral Sources
                st.subheader("Top Referral Sources")
                if analytics_results['referral_sources'] is not None:
                    ref_df = analytics_results['referral_sources']
                    st.dataframe(ref_df.iloc[:5][['referral_source', 'Total', 'Won', 'Conversion %']], use_container_width=True)
                    
                    # Plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    plot_conversion_by_category(
                        st.session_state.processed_df,
                        'referral_source',
                        'Conversion Rate by Referral Source',
                        ax=ax,
                        top_n=8
                    )
                    st.pyplot(fig)
                else:
                    st.info("No referral source data available. Make sure your data includes a 'referral_source' column.")
            
            with col2:
                # Marketing Sources
                st.subheader("Top Marketing Sources")
                if analytics_results['marketing_sources'] is not None:
                    mkt_df = analytics_results['marketing_sources']
                    st.dataframe(mkt_df.iloc[:5][['marketing_source', 'Total', 'Won', 'Conversion %']], use_container_width=True)
                    
                    # Plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    plot_conversion_by_category(
                        st.session_state.processed_df,
                        'marketing_source',
                        'Conversion Rate by Marketing Source',
                        ax=ax,
                        top_n=8
                    )
                    st.pyplot(fig)
                else:
                    st.info("No marketing source data available. Make sure your data includes a 'marketing_source' column.")
            
            # Booking Types
            st.header("Event & Booking Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                # Booking Types
                st.subheader("Booking Type Performance")
                if analytics_results['booking_types'] is not None:
                    bt_df = analytics_results['booking_types']
                    st.dataframe(bt_df[['booking_type_clean', 'Total', 'Won', 'Conversion %']], use_container_width=True)
                    
                    # Plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    plot_conversion_by_category(
                        st.session_state.processed_df,
                        'booking_type',
                        'Conversion Rate by Booking Type',
                        ax=ax,
                        top_n=8
                    )
                    st.pyplot(fig)
                else:
                    st.info("No booking type data available. Make sure your data includes a 'booking_type' column.")
            
            with col2:
                # Price Per Guest Analysis
                st.subheader("Price Per Guest Impact")
                if analytics_results['price_per_guest'] is not None:
                    ppg_df = analytics_results['price_per_guest']
                    st.dataframe(ppg_df[['ppg_bin', 'Total', 'Won', 'Conversion %']], use_container_width=True)
                    
                    # Create bar chart
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.bar(ppg_df['ppg_bin'], ppg_df['Conversion'].values, color='skyblue')
                    
                    # Add data labels
                    for i, bar in enumerate(bars):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                f"{height:.1%}\n(n={ppg_df.iloc[i]['Total']})",
                                ha='center', va='bottom')
                    
                    # Customize plot
                    ax.set_title('Conversion Rate by Price Per Guest', fontsize=14)
                    ax.set_ylabel('Conversion Rate', fontsize=12)
                    ax.set_ylim(0, max(ppg_df['Conversion'].values) * 1.3)
                    ax.grid(axis='y', linestyle='--', alpha=0.7)
                    
                    st.pyplot(fig)
                else:
                    st.info("No price per guest data available. Make sure your data includes 'actual_deal_value' and 'number_of_guests' columns.")
            
            # Time and Seasonality Section
            st.header("Time & Seasonality Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                # Days Since Inquiry Analysis
                st.subheader("Days Since Inquiry Impact")
                if analytics_results['days_since_inquiry'] is not None:
                    dsi_df = analytics_results['days_since_inquiry']
                    st.dataframe(dsi_df[['dsi_bin', 'Total', 'Won', 'Conversion %']], use_container_width=True)
                    
                    # Create bar chart
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.bar(dsi_df['dsi_bin'], dsi_df['Conversion'].values, color='skyblue')
                    
                    # Add data labels
                    for i, bar in enumerate(bars):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                f"{height:.1%}\n(n={dsi_df.iloc[i]['Total']})",
                                ha='center', va='bottom')
                    
                    # Customize plot
                    ax.set_title('Conversion Rate by Days Since Inquiry', fontsize=14)
                    ax.set_ylabel('Conversion Rate', fontsize=12)
                    ax.set_ylim(0, max(dsi_df['Conversion'].values) * 1.3)
                    ax.grid(axis='y', linestyle='--', alpha=0.7)
                    
                    st.pyplot(fig)
                else:
                    st.info("No days since inquiry data available. Make sure your data includes a 'days_since_inquiry' column.")
            
            with col2:
                # Event Month Analysis
                st.subheader("Event Month Seasonality")
                if analytics_results['event_month'] is not None:
                    month_df = analytics_results['event_month']
                    st.dataframe(month_df[['event_month', 'Total', 'Won', 'Conversion %']], use_container_width=True)
                    
                    # Create bar chart
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.bar(month_df['event_month'], month_df['Conversion'].values, color='skyblue')
                    
                    # Add data labels
                    for i, bar in enumerate(bars):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                f"{height:.1%}",
                                ha='center', va='bottom')
                    
                    # Customize plot
                    ax.set_title('Conversion Rate by Event Month', fontsize=14)
                    ax.set_ylabel('Conversion Rate', fontsize=12)
                    ax.set_ylim(0, max(month_df['Conversion'].values) * 1.3)
                    ax.grid(axis='y', linestyle='--', alpha=0.7)
                    plt.xticks(rotation=45)
                    
                    st.pyplot(fig)
                else:
                    st.info("No event month data available. Make sure your data includes an 'event_date' column.")
            
            # Staff Ratio & Day of Week
            col1, col2 = st.columns(2)
            
            with col1:
                # Staff Ratio Analysis
                st.subheader("Staff-to-Guest Ratio Impact")
                if analytics_results['staff_ratio'] is not None:
                    sr_df = analytics_results['staff_ratio']
                    st.dataframe(sr_df[['staff_ratio_bin', 'Total', 'Won', 'Conversion %']], use_container_width=True)
                    
                    # Create bar chart
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.bar(sr_df['staff_ratio_bin'], sr_df['Conversion'].values, color='skyblue')
                    
                    # Add data labels
                    for i, bar in enumerate(bars):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                f"{height:.1%}\n(n={sr_df.iloc[i]['Total']})",
                                ha='center', va='bottom')
                    
                    # Customize plot
                    ax.set_title('Conversion Rate by Staff-to-Guest Ratio', fontsize=14)
                    ax.set_ylabel('Conversion Rate', fontsize=12)
                    ax.set_ylim(0, max(sr_df['Conversion'].values) * 1.3)
                    ax.grid(axis='y', linestyle='--', alpha=0.7)
                    
                    st.pyplot(fig)
                else:
                    st.info("No staff ratio data available. Make sure your data includes 'bartenders_needed' and 'number_of_guests' columns.")
            
            with col2:
                # Inquiry Weekday Analysis
                st.subheader("Inquiry Day of Week")
                if analytics_results['inquiry_weekday'] is not None:
                    wkday_df = analytics_results['inquiry_weekday']
                    st.dataframe(wkday_df[['inquiry_weekday', 'Total', 'Won', 'Conversion %']], use_container_width=True)
                    
                    # Create bar chart
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.bar(wkday_df['inquiry_weekday'], wkday_df['Conversion'].values, color='skyblue')
                    
                    # Add data labels
                    for i, bar in enumerate(bars):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                f"{height:.1%}\n(n={wkday_df.iloc[i]['Total']})",
                                ha='center', va='bottom')
                    
                    # Customize plot
                    ax.set_title('Conversion Rate by Inquiry Day of Week', fontsize=14)
                    ax.set_ylabel('Conversion Rate', fontsize=12)
                    ax.set_ylim(0, max(wkday_df['Conversion'].values) * 1.3)
                    ax.grid(axis='y', linestyle='--', alpha=0.7)
                    
                    st.pyplot(fig)
                else:
                    st.info("No inquiry weekday data available. Make sure your data includes an 'inquiry_date' column.")
            
            # Add explanation text
            st.markdown("""
            ### How to Use These Insights
            
            1. **Focus resources on high-converting channels**: Prioritize your effort on referral and marketing sources with the highest conversion rates.
            
            2. **Optimize pricing strategy**: Use the price per guest analysis to determine the most effective pricing tiers.
            
            3. **Plan for seasonality**: Adjust staffing and marketing based on month-to-month patterns.
            
            4. **Streamline your process**: If same-day inquiries convert significantly better, consider improving your immediate response protocols.
            
            5. **Optimize staffing ratio**: Find the sweet spot for staff-to-guest ratio that maximizes both customer satisfaction and conversion rate.
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