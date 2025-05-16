import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os
import io
import plotly.express as px
import plotly.graph_objects as go
from conversion_analysis import run_conversion_analysis, plot_booking_types, plot_referral_marketing_sources
from conversion import analyze_phone_matches, analyze_time_to_conversion
from utils import process_data, calculate_conversion_rates, calculate_correlations, load_and_normalize_data
from database import get_lead_data, get_operation_data, import_leads_data, import_operations_data, process_phone_matching
from derive_scorecard import generate_lead_scorecard, score_lead
from mistral_insights import generate_insights

# Set page config
st.set_page_config(page_title="Sales Conversion Analytics", page_icon="📊", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        color: #1E88E5;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0;
    }
    .sub-header {
        color: #424242;
        font-size: 1.2rem;
        margin-top: 0;
    }
    .header-container {
        margin-bottom: 2rem;
    }
    .insight-box {
        background-color: #f0f7ff;
        border-left: 4px solid #1E88E5;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .metric-title {
        font-size: 1rem;
        color: #424242;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .metric-delta {
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Main app header and description
st.markdown("""
<div class="header-container">
    <h1 class="main-header">Sales Conversion Analytics</h1>
    <p class="sub-header">Interactive dashboard for analyzing lead conversion metrics and optimizing sales processes</p>
</div>
""", unsafe_allow_html=True)

# Data loading section in sidebar
st.sidebar.header("Data Source")

data_source = st.sidebar.radio(
    "Choose data source",
    ["Database Data", "Upload CSV"],
    index=0
)

# Load data
if data_source == "Database Data":
    if st.sidebar.button("Load All Data"):
        filtered_df = load_and_normalize_data(use_database=True)
        raw_df = filtered_df.copy()
        st.success("Data loaded from database!")
else:
    leads_file = st.sidebar.file_uploader("Upload Leads CSV", type=['csv'])
    
    if leads_file is not None:
        try:
            # Save uploaded file temporarily
            temp_path = "temp_leads.csv"
            with open(temp_path, "wb") as f:
                f.write(leads_file.getbuffer())
            
            # Import the data
            imported_count = import_leads_data(temp_path)
            st.sidebar.success(f"Successfully imported {imported_count} lead records!")
        except Exception as e:
            st.sidebar.error(f"Error importing leads data: {str(e)}")
    
    operations_file = st.sidebar.file_uploader("Upload Operations CSV", type=['csv'])
    
    if operations_file is not None:
        try:
            # Save uploaded file temporarily
            temp_path = "temp_operations.csv"
            with open(temp_path, "wb") as f:
                f.write(operations_file.getbuffer())
            
            # Import the data
            imported_count = import_operations_data(temp_path)
            st.sidebar.success(f"Successfully imported {imported_count} operation records!")
        except Exception as e:
            st.sidebar.error(f"Error importing operations data: {str(e)}")

# Initialize data if not already loaded
if 'filtered_df' not in locals():
    try:
        # Try to load from database if available
        leads_df = get_lead_data()
        if leads_df is not None and not leads_df.empty:
            operations_df = get_operation_data()
            
            # Process the data
            filtered_df = load_and_normalize_data(use_database=True)
            raw_df = filtered_df.copy()
        else:
            # Fallback to default empty dataframe
            filtered_df = pd.DataFrame()
            raw_df = pd.DataFrame()
    except Exception as e:
        st.sidebar.error(f"Error loading data: {str(e)}")
        filtered_df = pd.DataFrame()
        raw_df = pd.DataFrame()

# Define tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Conversion Analysis", 
    "Lead Scoring", 
    "Contact Matching", 
    "Insights", 
    "Admin"
])

# Setup default filters for backward compatibility
if 'date_filter' not in st.session_state:
    # Set default date range to last 90 days if data is available
    if 'inquiry_date' in filtered_df.columns:
        try:
            max_date = filtered_df['inquiry_date'].max()
            if pd.notna(max_date):
                default_start = max_date - datetime.timedelta(days=90)
                st.session_state.date_filter = (default_start.date(), max_date.date())
            else:
                st.session_state.date_filter = None
        except:
            st.session_state.date_filter = None
    else:
        st.session_state.date_filter = None

if 'status_filter' not in st.session_state:
    st.session_state.status_filter = 'All'

if 'region_filter' not in st.session_state:
    st.session_state.region_filter = ['All']

# For debugging
if filtered_df.empty:
    st.info("No data loaded. Please use the sidebar to load data.")
else:
    # Show info about applied filters
    if 'date_filter' in st.session_state and st.session_state.date_filter and len(st.session_state.date_filter) == 2:
        start_date, end_date = st.session_state.date_filter
        st.info(f"Loaded {len(filtered_df)} leads from {start_date} to {end_date}")
    
    # Region filter information
    if 'region_filter' in st.session_state and st.session_state.region_filter and 'All' not in st.session_state.region_filter:
        st.info(f"Applied region filter: {', '.join(st.session_state.region_filter)}")

    with tab1:
        # --- Conversion Analysis Tab (with filters disabled) ---
        st.markdown(
            "## Conversion Analysis  \n"
            "*Showing all data without filtering – filters have been disabled for consistent data view.*"
        )

        # Use the unfiltered dataframe
        df = filtered_df
        
        if df.empty:
            st.warning("No data available. Please load data using the sidebar options.")
        else:
            # 1. KPI Overview
            st.subheader("KPI Overview")
            
            # Calculate KPIs
            total = len(df)
            won = int(df['outcome'].sum()) if 'outcome' in df.columns else 0
            lost = total - won
            rate = won/total if total > 0 else 0
            
            # Display KPI cards
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Leads", f"{total:,}")
            c2.metric("Won Deals", f"{won:,}")
            c3.metric("Lost Deals", f"{lost:,}")
            c4.metric("Conversion Rate", f"{rate:.1%}")
            
            # 2. Trend Analysis
            st.subheader("Trend Analysis")
            
            # Identify date column
            date_col = None
            for col_name in ['inquiry_date', 'created', 'created_at', 'date']:
                if col_name in df.columns:
                    date_col = col_name
                    break
            
            if date_col:
                # Create trend charts
                trend_col1, trend_col2 = st.columns(2)
                
                with trend_col1:
                    st.write("Conversion Rate Over Time")
                    
                    # Prepare time series data
                    df['created_dt'] = pd.to_datetime(df[date_col], errors='coerce')
                    weekly = (
                        df.dropna(subset=['created_dt','outcome'])
                          .set_index('created_dt')
                          .resample('W')['outcome']
                          .agg(['size','sum'])
                    )
                    
                    if not weekly.empty:
                        weekly['rate'] = weekly['sum'] / weekly['size']
                        
                        # Plot conversion rate trend
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.plot(weekly.index, weekly['rate'], marker='o', linewidth=2)
                        ax.set_ylabel('Conversion Rate')
                        ax.set_title('Weekly Conversion Rate')
                        ax.grid(True, alpha=0.3)
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Show week-over-week change
                        if len(weekly) >= 2:
                            last_week_rate = weekly['rate'].iloc[-1]
                            prev_week_rate = weekly['rate'].iloc[-2]
                            wow_change = (last_week_rate - prev_week_rate) / prev_week_rate if prev_week_rate > 0 else 0
                            delta_color = "normal" if wow_change == 0 else ("inverse" if wow_change < 0 else "normal")
                            st.metric("Week-over-Week Change", f"{wow_change:.1%}", delta=f"{wow_change:.1%}", delta_color=delta_color)
                    else:
                        st.info("Insufficient time series data to generate trend charts.")
                
                with trend_col2:
                    st.write("Lead Volume Over Time")
                    
                    if not weekly.empty:
                        # Plot lead volume trend
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.bar(weekly.index, weekly['size'], color='#1E88E5')
                        ax.set_ylabel('Lead Count')
                        ax.set_title('Weekly Lead Volume')
                        ax.grid(True, alpha=0.3)
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Show week-over-week volume change
                        if len(weekly) >= 2:
                            last_week_vol = weekly['size'].iloc[-1]
                            prev_week_vol = weekly['size'].iloc[-2]
                            wow_vol_change = (last_week_vol - prev_week_vol) / prev_week_vol if prev_week_vol > 0 else 0
                            delta_color = "normal" if wow_vol_change >= 0 else "inverse"
                            st.metric("Volume Change", f"{wow_vol_change:.1%}", delta=f"{wow_vol_change:.1%}", delta_color=delta_color)
            else:
                st.info("No date field available for trend analysis. Please ensure your data has date information.")
            
            # 3. Category Breakouts
            st.subheader("Category Breakouts")
            
            # Helper function for category analysis
            def plot_category(col, title, min_count=10):
                st.write(f"#### {title}")
                if col not in df or df[col].dropna().empty:
                    st.info(f"No {title} data available")
                    return
                
                # Prepare data
                cats = df[col].fillna("Missing").astype(str)
                counts = cats.value_counts()
                keep = counts[counts>=min_count].index
                cats = cats.where(cats.isin(keep), "Other")
                
                # Calculate conversion rates
                summary = cats.to_frame("cat").join(df['outcome']).groupby("cat") \
                             .agg(total=("outcome","size"), won=("outcome","sum"))
                summary['rate'] = summary['won']/summary['total']
                top = summary.sort_values('rate', ascending=False)
                
                if not top.empty:
                    # Plot horizontal bar chart
                    fig, ax = plt.subplots(figsize=(10, max(3, min(len(top), 10) * 0.4)))
                    bars = ax.barh(top.index, top['rate'] * 100, color='teal')
                    
                    # Add percentage and count labels
                    for i, (idx, row) in enumerate(top.iterrows()):
                        ax.text(row['rate'] * 100 + 1, i, f"{row['rate']:.1%} (n={int(row['total'])})", va='center')
                    
                    ax.set_xlabel("Conversion Rate (%)")
                    ax.set_xlim(0, min(100, ax.get_xlim()[1] * 1.2))  # Add some padding but cap at 100%
                    st.pyplot(fig)
                    
                    # Add best/worst callout
                    if len(top) > 1:
                        best = top.iloc[0]
                        worst = top.iloc[-1]
                        if best['rate'] > worst['rate']:
                            diff = best['rate'] - worst['rate']
                            st.info(f"💡 **{best.name}** has a {diff:.1%} higher conversion rate than **{worst.name}**")
                else:
                    st.info(f"Not enough data points ({min_count}+ needed) for {title}")
            
            # Create two columns for category plots
            cat_col1, cat_col2 = st.columns(2)
            
            with cat_col1:
                plot_category('booking_type', "Conversion by Booking Type")
                plot_category('event_type', "Conversion by Event Type")
            
            with cat_col2:
                plot_category('referral_source', "Conversion by Referral Source")
                plot_category('marketing_source', "Conversion by Marketing Source")
            
            # 4. Timing Factors
            st.subheader("Timing Factors")
            
            # Helper function for numeric bin plots
            def plot_numeric_bin(col, title, bins, labels):
                st.write(f"#### {title}")
                if col not in df:
                    st.info(f"No {title} data available")
                    return
                
                # Prepare data
                nums = pd.to_numeric(df[col], errors='coerce')
                valid = df.loc[nums.notna()].copy()
                
                if valid.empty:
                    st.info(f"No valid numeric data for {title}")
                    return
                
                valid['bin'] = pd.cut(nums.dropna(), bins=bins, labels=labels)
                summary = valid.groupby('bin')['outcome'] \
                               .agg(total='size', won='sum')
                
                if summary.empty:
                    st.info(f"No valid {title} data to plot.")
                    return
                
                summary['rate'] = summary['won']/summary['total']
                
                # Plot bar chart
                fig, ax = plt.subplots(figsize=(10, 5))
                bars = ax.bar(summary.index.astype(str), summary['rate'] * 100, color='#3498db')
                
                # Add percentage and count labels
                for i, (idx, row) in enumerate(summary.iterrows()):
                    ax.text(i, row['rate'] * 100 + 1, f"{row['rate']:.1%}\n(n={int(row['total'])})", ha='center')
                
                ax.set_ylabel("Conversion Rate (%)")
                ax.set_ylim(0, min(100, ax.get_ylim()[1] * 1.2))  # Add some padding but cap at 100%
                plt.xticks(rotation=0)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Add best/worst callout
                if len(summary) > 1:
                    best_bin = summary['rate'].idxmax()
                    worst_bin = summary['rate'].idxmin()
                    if summary.loc[best_bin, 'rate'] > summary.loc[worst_bin, 'rate']:
                        diff = summary.loc[best_bin, 'rate'] - summary.loc[worst_bin, 'rate']
                        st.info(f"💡 **{best_bin}** leads convert at {summary.loc[best_bin, 'rate']:.1%}, which is {diff:.1%} higher than **{worst_bin}** leads at {summary.loc[worst_bin, 'rate']:.1%}")
            
            # Create two columns for timing factors
            time_col1, time_col2 = st.columns(2)
            
            with time_col1:
                # Days Until Event
                plot_numeric_bin(
                    'days_until_event', "Conversion by Days Until Event",
                    bins=[0, 7, 30, 90, np.inf], labels=['≤7d', '8–30d', '31–90d', '90+d']
                )
                
                # Weekday Analysis
                st.write("#### Conversion by Submission Weekday")
                if date_col and 'created_dt' in df:
                    wd = df.dropna(subset=['created_dt']).copy()
                    wd['weekday'] = wd['created_dt'].dt.day_name()
                    
                    weekday_summary = wd.groupby('weekday').agg(
                        total=('outcome', 'size'),
                        won=('outcome', 'sum')
                    )
                    
                    weekday_summary['rate'] = weekday_summary['won'] / weekday_summary['total']
                    
                    # Reindex to ensure days are in correct order
                    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    weekday_summary = weekday_summary.reindex(weekday_order)
                    
                    # Plot weekday bar chart
                    fig, ax = plt.subplots(figsize=(10, 5))
                    bars = ax.bar(weekday_summary.index, weekday_summary['rate'] * 100, color='#9b59b6')
                    
                    # Add percentage and count labels
                    for i, (idx, row) in enumerate(weekday_summary.iterrows()):
                        if pd.notna(row['rate']):
                            ax.text(i, row['rate'] * 100 + 1, f"{row['rate']:.1%}\n(n={int(row['total'])})", ha='center')
                    
                    ax.set_ylabel("Conversion Rate (%)")
                    ax.set_ylim(0, min(100, ax.get_ylim()[1] * 1.2))
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Find best/worst weekday
                    if not weekday_summary['rate'].isna().all() and len(weekday_summary) > 1:
                        best_day = weekday_summary['rate'].idxmax()
                        worst_day = weekday_summary['rate'].idxmin()
                        if weekday_summary.loc[best_day, 'rate'] > weekday_summary.loc[worst_day, 'rate']:
                            diff = weekday_summary.loc[best_day, 'rate'] - weekday_summary.loc[worst_day, 'rate']
                            st.info(f"💡 **{best_day}** submissions convert at {weekday_summary.loc[best_day, 'rate']:.1%}, which is {diff:.1%} higher than **{worst_day}** at {weekday_summary.loc[worst_day, 'rate']:.1%}")
                else:
                    st.info("No date information available for weekday analysis.")
            
            with time_col2:
                # Days Since Inquiry
                plot_numeric_bin(
                    'days_since_inquiry', "Conversion by Days Since Inquiry",
                    bins=[0, 1, 3, 7, 30, np.inf], labels=['0d', '1–3d', '4–7d', '8–30d', '30+d']
                )
                
                # Hour Analysis (if available)
                st.write("#### Conversion by Submission Hour")
                if date_col and 'created_dt' in df:
                    hd = df.dropna(subset=['created_dt']).copy()
                    hd['hour'] = hd['created_dt'].dt.hour
                    
                    hour_summary = hd.groupby('hour').agg(
                        total=('outcome', 'size'),
                        won=('outcome', 'sum')
                    )
                    
                    hour_summary['rate'] = hour_summary['won'] / hour_summary['total']
                    
                    if not hour_summary.empty and not hour_summary['rate'].isna().all():
                        # Plot hour line chart
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.plot(hour_summary.index, hour_summary['rate'] * 100, marker='o', linewidth=2, color='#e74c3c')
                        
                        # Add count annotations
                        for idx, row in hour_summary.iterrows():
                            if pd.notna(row['rate']) and row['total'] > 5:  # Only annotate hours with sufficient data
                                ax.text(idx, row['rate'] * 100 + 2, f"n={int(row['total'])}", ha='center', fontsize=8)
                        
                        ax.set_xlabel("Hour of Day (24h)")
                        ax.set_ylabel("Conversion Rate (%)")
                        ax.set_xticks(range(0, 24, 2))
                        ax.set_xlim(-0.5, 23.5)
                        ax.set_ylim(0, min(100, ax.get_ylim()[1] * 1.2))
                        ax.grid(True, alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Find peak hours
                        peak_hours = hour_summary.nlargest(3, 'rate')
                        low_hours = hour_summary.nsmallest(3, 'rate')
                        
                        if not peak_hours.empty and not low_hours.empty:
                            peak_hour = peak_hours.index[0]
                            low_hour = low_hours.index[0]
                            peak_rate = peak_hours.iloc[0]['rate']
                            low_rate = low_hours.iloc[0]['rate']
                            
                            if peak_rate > low_rate:
                                diff = peak_rate - low_rate
                                st.info(f"💡 Leads submitted at **{peak_hour:02d}:00** convert at {peak_rate:.1%}, which is {diff:.1%} higher than **{low_hour:02d}:00** at {low_rate:.1%}")
                    else:
                        st.info("Insufficient hourly data for analysis.")
                else:
                    st.info("No timestamp information available for hour analysis.")
            
            # 5. Price & Size Effects
            st.subheader("Price & Size Effects")
            
            # Create two columns for price and size factors
            size_col1, size_col2 = st.columns(2)
            
            with size_col1:
                # Number of Guests
                plot_numeric_bin(
                    'number_of_guests', "Conversion by Number of Guests",
                    bins=[0, 50, 100, 200, 500, np.inf], 
                    labels=['1–50', '51–100', '101–200', '201–500', '500+']
                )
            
            with size_col2:
                # Staff-to-Guest Ratio
                st.write("#### Conversion by Staff-to-Guest Ratio")
                if {'bartenders_needed', 'number_of_guests'}.issubset(df.columns):
                    try:
                        # Calculate ratio
                        df_ratio = df.copy()
                        df_ratio['staff_ratio'] = df_ratio['bartenders_needed'] / df_ratio['number_of_guests'].replace(0, np.nan)
                        
                        # Create bins
                        df_ratio['ratio_bin'] = pd.cut(
                            df_ratio['staff_ratio'], 
                            bins=[0, 0.01, 0.02, 0.05, np.inf],
                            labels=['<1%', '1–2%', '2–5%', '5%+']
                        )
                        
                        # Calculate conversion rates
                        ratio_summary = df_ratio.groupby('ratio_bin').agg(
                            total=('outcome', 'size'),
                            won=('outcome', 'sum')
                        )
                        
                        ratio_summary['rate'] = ratio_summary['won'] / ratio_summary['total']
                        
                        if not ratio_summary.empty and not ratio_summary['rate'].isna().all():
                            # Plot staff ratio bar chart
                            fig, ax = plt.subplots(figsize=(10, 5))
                            bars = ax.bar(ratio_summary.index.astype(str), ratio_summary['rate'] * 100, color='#f39c12')
                            
                            # Add percentage and count labels
                            for i, (idx, row) in enumerate(ratio_summary.iterrows()):
                                if pd.notna(row['rate']):
                                    ax.text(i, row['rate'] * 100 + 1, f"{row['rate']:.1%}\n(n={int(row['total'])})", ha='center')
                            
                            ax.set_ylabel("Conversion Rate (%)")
                            ax.set_ylim(0, min(100, ax.get_ylim()[1] * 1.2))
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # Find best ratio
                            best_ratio = ratio_summary['rate'].idxmax()
                            worst_ratio = ratio_summary['rate'].idxmin()
                            
                            if pd.notna(best_ratio) and pd.notna(worst_ratio):
                                best_rate = ratio_summary.loc[best_ratio, 'rate']
                                worst_rate = ratio_summary.loc[worst_ratio, 'rate']
                                
                                if best_rate > worst_rate:
                                    diff = best_rate - worst_rate
                                    st.info(f"💡 A staff-to-guest ratio of **{best_ratio}** converts at {best_rate:.1%}, which is {diff:.1%} higher than **{worst_ratio}** at {worst_rate:.1%}")
                        else:
                            st.info("Insufficient staff ratio data for analysis.")
                    except Exception as e:
                        st.info(f"Could not calculate staff-to-guest ratio: {str(e)}")
                else:
                    st.info("Staff and guest count information not available for ratio analysis.")
            
            # 6. Geographic Analysis
            st.subheader("Geographic Analysis")
            
            geo_col1, geo_col2 = st.columns(2)
            
            with geo_col1:
                # State/Region Analysis
                plot_category('state', "Conversion by State/Region", min_count=5)
            
            with geo_col2:
                # Phone Area Code Match
                st.write("#### Conversion by Phone Match")
                
                if 'phone_number' in df.columns and 'state' in df.columns:
                    try:
                        # Use the phone matching analysis function
                        phone_match_data = analyze_phone_matches(df)
                        
                        if phone_match_data and isinstance(phone_match_data, tuple) and len(phone_match_data) >= 1:
                            match_conv = phone_match_data[0]
                            
                            if not match_conv.empty:
                                # Format the table for display
                                display_conv = match_conv.copy()
                                
                                if 'conversion_rate' in display_conv.columns:
                                    display_conv['conversion_rate'] = display_conv['conversion_rate'].apply(lambda x: f"{x:.1%}")
                                
                                st.dataframe(display_conv)
                                
                                # Create visualization if possible
                                if 'conversion_rate' in match_conv.columns:
                                    fig, ax = plt.subplots(figsize=(10, 5))
                                    
                                    # Convert to numeric for plotting
                                    match_conv['plot_rate'] = pd.to_numeric(match_conv['conversion_rate'], errors='coerce')
                                    
                                    bars = ax.bar(match_conv.index, match_conv['plot_rate'] * 100, color='#2ecc71')
                                    
                                    # Add percentage labels
                                    for i, (idx, row) in enumerate(match_conv.iterrows()):
                                        if pd.notna(row['plot_rate']):
                                            ax.text(i, row['plot_rate'] * 100 + 1, f"{row['plot_rate']:.1%}", ha='center')
                                    
                                    ax.set_ylabel('Conversion Rate (%)')
                                    ax.set_ylim(0, min(100, ax.get_ylim()[1] * 1.2))
                                    plt.tight_layout()
                                    st.pyplot(fig)
                            else:
                                st.info("No phone matching data available.")
                        else:
                            st.info("Phone matching analysis did not return expected results.")
                    except Exception as e:
                        st.info(f"Could not perform phone matching analysis: {str(e)}")
                else:
                    st.info("Phone number and state information not available for matching analysis.")
            
            # 7. Data Quality & Anomalies
            st.subheader("Data Quality & Anomalies")
            
            # Calculate missing percentages
            miss = df.isna().mean().mul(100).round(1).sort_values(ascending=False)
            missing_df = pd.DataFrame({'% Missing': miss})
            
            # Create two columns
            quality_col1, quality_col2 = st.columns(2)
            
            with quality_col1:
                st.write("#### Missing Data by Field")
                st.dataframe(missing_df)
            
            with quality_col2:
                st.write("#### Data Anomalies")
                
                anomalies = []
                
                # Check for negative days until event
                if 'days_until_event' in df.columns:
                    neg_days = pd.to_numeric(df['days_until_event'], errors='coerce') < 0
                    neg_count = int(neg_days.sum())
                    
                    if neg_count > 0:
                        anomalies.append(f"• {neg_count} leads have negative Days Until Event.")
                
                # Check for extreme guest counts
                if 'number_of_guests' in df.columns:
                    large_guests = pd.to_numeric(df['number_of_guests'], errors='coerce') > 1000
                    large_count = int(large_guests.sum())
                    
                    if large_count > 0:
                        anomalies.append(f"• {large_count} leads have more than 1,000 guests.")
                
                # Check for far future events
                if 'days_until_event' in df.columns:
                    far_days = pd.to_numeric(df['days_until_event'], errors='coerce') > 365
                    far_count = int(far_days.sum())
                    
                    if far_count > 0:
                        anomalies.append(f"• {far_count} leads have events more than 1 year in the future.")
                
                if anomalies:
                    for anomaly in anomalies:
                        st.warning(anomaly)
                else:
                    st.success("No significant data anomalies detected.")

    with tab2:
        # --- Lead Scoring Tab ---
        st.markdown("## Lead Scoring")
        st.markdown("Score leads based on key factors to prioritize follow-up efforts.")
        
        # Generate lead score model
        if st.button("Generate Lead Scoring Model"):
            try:
                with st.spinner("Generating lead scoring model..."):
                    weights_df, thresholds, metrics = generate_lead_scorecard()
                    
                    # Display model metrics
                    st.success(f"Lead scoring model generated! ROC AUC: {metrics.get('roc_auc', 0):.3f}")
                    
                    # Store in session state
                    st.session_state.scorecard = weights_df
                    st.session_state.thresholds = thresholds
                    st.session_state.metrics = metrics
            except Exception as e:
                st.error(f"Error generating lead scorecard: {str(e)}")
        
        # Check if model exists
        if 'scorecard' in st.session_state:
            weights_df = st.session_state.scorecard
            thresholds = st.session_state.thresholds
            metrics = st.session_state.metrics
            
            # Display tabs for different lead scoring features
            score_tabs = st.tabs(["Score Calculator", "Model Performance", "Feature Importance"])
            
            # Calculator tab
            with score_tabs[0]:
                st.subheader("Lead Score Calculator")
                
                # Create a form for input
                with st.form("lead_calculator"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Dynamic booking type options
                        booking_types = filtered_df['booking_type'].dropna().unique().tolist() if 'booking_type' in filtered_df.columns else []
                        booking_type = st.selectbox("Booking Type", options=[''] + sorted(booking_types))
                        
                        guests = st.number_input("Number of Guests", min_value=0, value=100)
                        days_until = st.number_input("Days Until Event", min_value=0, value=30)
                        
                    with col2:
                        # Dynamic state options
                        states = filtered_df['state'].dropna().unique().tolist() if 'state' in filtered_df.columns else []
                        state = st.selectbox("State", options=[''] + sorted(states))
                        
                        bartenders = st.number_input("Bartenders Needed", min_value=0, value=2)
                        deal_value = st.number_input("Deal Value ($)", min_value=0, value=1000)
                    
                    submit = st.form_submit_button("Calculate Score")
                
                if submit:
                    # Prepare lead data
                    lead_data = {
                        'booking_type': booking_type,
                        'number_of_guests': guests,
                        'days_until_event': days_until,
                        'state': state,
                        'bartenders_needed': bartenders,
                        'actual_deal_value': deal_value
                    }
                    
                    # Calculate score
                    try:
                        score, category = score_lead(lead_data, weights_df)
                        
                        # Display results
                        st.subheader("Score Results")
                        
                        # Create score display with color
                        color = "#e74c3c" if category == "Hot" else "#f39c12" if category == "Warm" else "#3498db"
                        score_html = f"""
                        <div style="display: flex; align-items: center; margin-bottom: 20px;">
                            <div style="background-color: {color}; color: white; padding: 15px 25px; border-radius: 5px; text-align: center; margin-right: 20px;">
                                <div style="font-size: 24px; font-weight: bold;">{score:.1f}</div>
                                <div>points</div>
                            </div>
                            <div style="background-color: {color}; color: white; padding: 15px 25px; border-radius: 5px; text-align: center;">
                                <div style="font-size: 24px; font-weight: bold;">{category}</div>
                                <div>lead</div>
                            </div>
                        </div>
                        """
                        st.markdown(score_html, unsafe_allow_html=True)
                        
                        # Show conversion probability
                        if 'best_threshold' in metrics:
                            prob = score / 10  # Simplified probability calculation
                            st.metric("Estimated Conversion Probability", f"{prob:.1%}")
                        
                        # Show top factors
                        st.subheader("Top Contributing Factors")
                        
                        # Calculate impact of each feature
                        factors = []
                        for feature, weight in zip(weights_df['feature'], weights_df['weight']):
                            if feature in lead_data and pd.notna(lead_data[feature]):
                                value = lead_data[feature]
                                impact = weight
                                factors.append((feature, value, weight, impact))
                        
                        # Sort by absolute impact
                        factors.sort(key=lambda x: abs(x[3]), reverse=True)
                        
                        # Display positive and negative factors
                        pos_factors = [f for f in factors if f[3] > 0]
                        neg_factors = [f for f in factors if f[3] < 0]
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Positive Factors**")
                            if pos_factors:
                                for feature, value, weight, impact in pos_factors[:3]:
                                    st.markdown(f"• **{feature}**: +{impact:.1f} points")
                            else:
                                st.markdown("No significant positive factors.")
                        
                        with col2:
                            st.markdown("**Negative Factors**")
                            if neg_factors:
                                for feature, value, weight, impact in neg_factors[:3]:
                                    st.markdown(f"• **{feature}**: {impact:.1f} points")
                            else:
                                st.markdown("No significant negative factors.")
                    
                    except Exception as e:
                        st.error(f"Error calculating lead score: {str(e)}")
            
            # Model Performance tab
            with score_tabs[1]:
                st.subheader("Model Performance")
                
                if metrics:
                    # Display key metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        roc_auc = metrics.get('roc_auc', 0)
                        st.metric("ROC AUC", f"{roc_auc:.3f}")
                    
                    with col2:
                        threshold = metrics.get('best_threshold', 0)
                        st.metric("Optimal Threshold", f"{threshold:.3f}")
                    
                    with col3:
                        # Calculate accuracy if possible
                        if 'confusion_matrix' in metrics:
                            cm = metrics['confusion_matrix']
                            accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum() if cm.sum() > 0 else 0
                            st.metric("Model Accuracy", f"{accuracy:.1%}")
                    
                    # ROC Curve
                    st.subheader("ROC Curve")
                    
                    if all(k in metrics for k in ['fpr', 'tpr']):
                        # Plot ROC curve
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.plot(metrics['fpr'], metrics['tpr'], 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
                        ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
                        ax.set_xlabel('False Positive Rate')
                        ax.set_ylabel('True Positive Rate')
                        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
                        ax.legend(loc='lower right')
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                    
                    # Score Distributions
                    st.subheader("Score Distributions")
                    
                    if all(k in metrics for k in ['won_scores', 'lost_scores']):
                        # Plot score distributions
                        fig, ax = plt.subplots(figsize=(8, 6))
                        
                        # Get the scores
                        won_scores = metrics['won_scores']
                        lost_scores = metrics['lost_scores']
                        
                        # Plot histograms
                        bins = np.linspace(0, 10, 20)
                        ax.hist(won_scores, bins=bins, alpha=0.7, label='Won', color='green')
                        ax.hist(lost_scores, bins=bins, alpha=0.7, label='Lost', color='red')
                        
                        ax.set_xlabel('Score')
                        ax.set_ylabel('Count')
                        ax.set_title('Score Distribution by Outcome')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                    
                    # Confusion Matrix
                    st.subheader("Confusion Matrix")
                    
                    if 'confusion_matrix' in metrics:
                        cm = metrics['confusion_matrix']
                        
                        # Plot confusion matrix
                        fig, ax = plt.subplots(figsize=(6, 6))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                        ax.set_xlabel('Predicted')
                        ax.set_ylabel('Actual')
                        ax.set_title('Confusion Matrix')
                        ax.set_xticklabels(['Lost', 'Won'])
                        ax.set_yticklabels(['Lost', 'Won'])
                        st.pyplot(fig)
                else:
                    st.info("No model metrics available. Please regenerate the lead scoring model.")
            
            # Feature Importance tab
            with score_tabs[2]:
                st.subheader("Feature Importance")
                
                if not weights_df.empty:
                    # Sort features by absolute weight
                    importance_df = weights_df.copy()
                    importance_df = importance_df.sort_values('weight', key=abs, reverse=True)
                    
                    # Plot feature importance
                    fig, ax = plt.subplots(figsize=(10, max(6, min(len(importance_df), 20) * 0.4)))
                    
                    # Create bars with color based on sign
                    bars = ax.barh(importance_df['feature'], importance_df['weight'])
                    for i, bar in enumerate(bars):
                        bar.set_color('green' if bar.get_width() >= 0 else 'red')
                    
                    ax.set_xlabel('Weight (Impact on Score)')
                    ax.set_title('Feature Importance')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    
                    # Feature importance table
                    st.subheader("Feature Weights")
                    
                    # Format the table for display
                    display_weights = importance_df.copy()
                    display_weights.columns = ['Feature', 'Weight']
                    st.dataframe(display_weights)
                    
                    # Key insights
                    st.subheader("Key Insights")
                    
                    # Find top positive and negative features
                    top_positive = importance_df[importance_df['weight'] > 0].head(3)
                    top_negative = importance_df[importance_df['weight'] < 0].head(3)
                    
                    if not top_positive.empty:
                        st.markdown("**Top Positive Factors:**")
                        for i, (_, row) in enumerate(top_positive.iterrows()):
                            st.markdown(f"• **{row['feature']}**: +{row['weight']:.2f} points")
                    
                    if not top_negative.empty:
                        st.markdown("**Top Negative Factors:**")
                        for i, (_, row) in enumerate(top_negative.iterrows()):
                            st.markdown(f"• **{row['feature']}**: {row['weight']:.2f} points")
                else:
                    st.info("No feature weights available. Please regenerate the lead scoring model.")
        else:
            st.info("No lead scoring model available. Click 'Generate Lead Scoring Model' to create one.")

    with tab3:
        # --- Contact Matching Analysis Tab ---
        st.markdown("## Contact Matching Analysis")
        st.markdown("Analyze phone matching patterns and identify leads with matching contact information.")
        
        if st.button("Process Phone Matching"):
            try:
                with st.spinner("Processing phone matching..."):
                    matched_count, total_leads, total_operations = process_phone_matching()
                    st.session_state.phone_matching_done = True
                    match_pct = matched_count / total_leads if total_leads > 0 else 0
                    
                    st.success(f"Phone matching complete! Matched {matched_count} of {total_leads} leads ({match_pct:.1%}).")
            except Exception as e:
                st.error(f"Error processing phone matching: {str(e)}")
        
        # Get phone matching data
        try:
            phone_match_data = analyze_phone_matches(filtered_df)
            
            if phone_match_data and len(phone_match_data) >= 2:
                match_conv, match_counts = phone_match_data
                
                # Display results in two columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Conversion Rate by Phone Match")
                    
                    if not match_conv.empty:
                        # Create formatted display dataframe
                        display_conv = match_conv.copy()
                        
                        if 'conversion_rate' in display_conv.columns:
                            display_conv['conversion_rate'] = display_conv['conversion_rate'].apply(lambda x: f"{x:.1%}")
                        
                        st.dataframe(display_conv)
                        
                        # Visualize conversion rates
                        if 'conversion_rate' in match_conv.columns:
                            fig, ax = plt.subplots(figsize=(8, 5))
                            
                            # Convert to numeric for plotting
                            match_conv['plot_rate'] = pd.to_numeric(match_conv['conversion_rate'], errors='coerce')
                            
                            ax.bar(match_conv.index, match_conv['plot_rate'] * 100, color='teal')
                            ax.set_ylabel('Conversion Rate (%)')
                            ax.set_ylim(0, min(100, ax.get_ylim()[1] * 1.2))
                            plt.tight_layout()
                            st.pyplot(fig)
                    else:
                        st.info("No conversion data available by phone match status.")
                
                with col2:
                    st.subheader("Lead Count by Phone Match")
                    
                    if not match_counts.empty:
                        st.dataframe(match_counts)
                        
                        # Visualize counts
                        if 'count' in match_counts.columns:
                            fig, ax = plt.subplots(figsize=(8, 5))
                            ax.bar(match_counts.index, match_counts['count'], color='#3498db')
                            ax.set_ylabel('Count')
                            plt.tight_layout()
                            st.pyplot(fig)
                    else:
                        st.info("No count data available by phone match status.")
                
                # Phone area code analysis
                st.subheader("Phone Area Code Analysis")
                
                # Time to conversion analysis
                st.subheader("Time to Conversion Analysis")
                
                try:
                    time_analysis = analyze_time_to_conversion(filtered_df)
                    
                    if time_analysis:
                        # Display KPIs
                        t1, t2, t3 = st.columns(3)
                        
                        with t1:
                            avg_days = time_analysis.get('average_days', 0)
                            st.metric("Average Days to Convert", f"{avg_days:.1f}")
                        
                        with t2:
                            median_days = time_analysis.get('median_days', 0)
                            st.metric("Median Days to Convert", f"{median_days:.1f}")
                        
                        with t3:
                            min_days = time_analysis.get('min_days', 0)
                            max_days = time_analysis.get('max_days', 0)
                            st.metric("Range", f"{min_days}–{max_days} days")
                        
                        # Conversion time distribution
                        if 'histogram_data' in time_analysis:
                            st.subheader("Distribution of Time to Conversion")
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sns.histplot(time_analysis['histogram_data'], bins=20, kde=True, ax=ax)
                            ax.set_xlabel('Days to Conversion')
                            ax.set_ylabel('Count')
                            plt.tight_layout()
                            st.pyplot(fig)
                        
                        # Conversion time by booking type
                        if 'by_booking_type' in time_analysis:
                            st.subheader("Average Time to Conversion by Booking Type")
                            
                            # Get the data
                            by_type = time_analysis['by_booking_type']
                            
                            if not by_type.empty:
                                # Plot bar chart
                                fig, ax = plt.subplots(figsize=(10, 6))
                                by_type.plot(kind='bar', ax=ax)
                                ax.set_xlabel('Booking Type')
                                ax.set_ylabel('Average Days to Conversion')
                                plt.xticks(rotation=45)
                                plt.tight_layout()
                                st.pyplot(fig)
                    else:
                        st.info("No time to conversion data available.")
                except Exception as e:
                    st.error(f"Error analyzing time to conversion: {str(e)}")
            else:
                st.info("No phone matching data available. Use the 'Process Phone Matching' button to generate data.")
        except Exception as e:
            st.error(f"Error analyzing phone matches: {str(e)}")

    with tab4:
        # --- Insights Tab ---
        st.markdown("## AI-Powered Insights")
        st.markdown("Discover key findings and actionable insights from your conversion data.")
        
        # Generate insights button
        if st.button("Generate AI Insights"):
            try:
                with st.spinner("Generating AI insights..."):
                    insights = generate_insights(filtered_df)
                    st.session_state.insights = insights
            except Exception as e:
                st.error(f"Error generating insights: {str(e)}")
                st.session_state.insights = None
        
        # Display insights if available
        if 'insights' in st.session_state and st.session_state.insights:
            insights = st.session_state.insights
            
            for i, insight in enumerate(insights):
                with st.expander(f"Insight {i+1}: {insight.get('title', 'Untitled Insight')}", expanded=i==0):
                    # Display content
                    st.markdown(insight.get('content', ''))
                    
                    # Display recommendations if available
                    if 'recommendations' in insight and insight['recommendations']:
                        st.subheader("Recommendations")
                        for rec in insight['recommendations']:
                            st.markdown(f"• {rec}")
                    
                    # Display chart if available
                    if 'chart' in insight:
                        try:
                            chart_data = insight['chart']
                            chart_type = insight.get('chart_type', 'bar')
                            
                            if chart_type == 'bar':
                                fig, ax = plt.subplots()
                                ax.bar(chart_data.get('x', []), chart_data.get('y', []))
                                ax.set_xlabel(chart_data.get('x_label', ''))
                                ax.set_ylabel(chart_data.get('y_label', ''))
                                ax.set_title(chart_data.get('title', ''))
                                plt.tight_layout()
                                st.pyplot(fig)
                            elif chart_type == 'line':
                                fig, ax = plt.subplots()
                                ax.plot(chart_data.get('x', []), chart_data.get('y', []))
                                ax.set_xlabel(chart_data.get('x_label', ''))
                                ax.set_ylabel(chart_data.get('y_label', ''))
                                ax.set_title(chart_data.get('title', ''))
                                plt.tight_layout()
                                st.pyplot(fig)
                            elif chart_type == 'pie':
                                fig, ax = plt.subplots()
                                ax.pie(chart_data.get('y', []), labels=chart_data.get('x', []), autopct='%1.1f%%')
                                ax.set_title(chart_data.get('title', ''))
                                plt.tight_layout()
                                st.pyplot(fig)
                        except Exception as e:
                            st.warning(f"Could not display chart: {str(e)}")
        else:
            # Show placeholder information
            st.info("Click 'Generate AI Insights' to get personalized insights based on your data.")
            
            with st.expander("Sample Insight: Best Performing Lead Sources", expanded=True):
                st.markdown("""
                ### Wedding Planners and Corporate Events Drive Highest Conversion
                
                Wedding Planner referrals convert at 56%, while Corporate Event leads convert at 48% - both significantly higher than the average conversion rate of 32%.
                
                These high-value lead sources account for 28% of your won deals despite representing only 15% of total lead volume.
                
                #### Recommendations:
                - Prioritize follow-up for Wedding Planner and Corporate leads
                - Explore partnership opportunities with wedding planners
                - Consider specialized materials for corporate event inquiries
                """)

    with tab5:
        # --- Admin Tab ---
        st.markdown("## Administration")
        st.markdown("Manage your data and dashboard settings.")
        
        admin_tabs = st.tabs(["Data Import", "Database", "Settings"])
        
        # Data Import tab
        with admin_tabs[0]:
            st.subheader("Data Import")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Import Leads Data")
                leads_upload = st.file_uploader("Upload Leads CSV", type=['csv'], key="admin_leads")
                
                if leads_upload is not None:
                    try:
                        # Save uploaded file temporarily
                        temp_path = "temp_leads.csv"
                        with open(temp_path, "wb") as f:
                            f.write(leads_upload.getbuffer())
                        
                        if st.button("Import Leads"):
                            with st.spinner("Importing leads data..."):
                                imported_count = import_leads_data(temp_path)
                                st.success(f"Successfully imported {imported_count} lead records!")
                    except Exception as e:
                        st.error(f"Error importing leads data: {str(e)}")
            
            with col2:
                st.markdown("### Import Operations Data")
                ops_upload = st.file_uploader("Upload Operations CSV", type=['csv'], key="admin_ops")
                
                if ops_upload is not None:
                    try:
                        # Save uploaded file temporarily
                        temp_path = "temp_operations.csv"
                        with open(temp_path, "wb") as f:
                            f.write(ops_upload.getbuffer())
                        
                        if st.button("Import Operations"):
                            with st.spinner("Importing operations data..."):
                                imported_count = import_operations_data(temp_path)
                                st.success(f"Successfully imported {imported_count} operation records!")
                    except Exception as e:
                        st.error(f"Error importing operations data: {str(e)}")
            
            if st.button("Reload Dashboard Data"):
                st.rerun()
        
        # Database tab
        with admin_tabs[1]:
            st.subheader("Database Administration")
            
            # Load current data
            try:
                leads_df = get_lead_data()
                operations_df = get_operation_data()
                
                # Display summary
                col1, col2 = st.columns(2)
                
                with col1:
                    leads_count = len(leads_df) if leads_df is not None else 0
                    st.metric("Total Leads", f"{leads_count:,}")
                
                with col2:
                    ops_count = len(operations_df) if operations_df is not None else 0
                    st.metric("Total Operations", f"{ops_count:,}")
                
                # Data preview
                with st.expander("Leads Data Preview"):
                    if leads_df is not None and not leads_df.empty:
                        st.dataframe(leads_df.head(10))
                    else:
                        st.info("No leads data available.")
                
                with st.expander("Operations Data Preview"):
                    if operations_df is not None and not operations_df.empty:
                        st.dataframe(operations_df.head(10))
                    else:
                        st.info("No operations data available.")
            except Exception as e:
                st.error(f"Error loading database data: {str(e)}")
        
        # Settings tab
        with admin_tabs[2]:
            st.subheader("Dashboard Settings")
            
            # Theme settings
            st.markdown("### Theme Settings")
            primary_color = st.color_picker("Primary Color", "#1E88E5")
            
            # Display settings
            st.markdown("### Display Settings")
            default_tab = st.selectbox(
                "Default Tab",
                options=["Conversion Analysis", "Lead Scoring", "Contact Matching", "Insights", "Admin"]
            )
            
            # Save settings
            if st.button("Save Settings"):
                # Store in session state
                st.session_state.settings = {
                    'primary_color': primary_color,
                    'default_tab': default_tab
                }
                
                st.success("Settings saved successfully!")
                
                # Custom CSS for applied color
                custom_css = f"""
                <style>
                    .main-header {{
                        color: {primary_color};
                    }}
                </style>
                """
                st.markdown(custom_css, unsafe_allow_html=True)