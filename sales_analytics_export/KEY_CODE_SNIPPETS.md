# Key Code Snippets - Sales Conversion Analytics Dashboard

## Main Dashboard Layout (from app.py)

```python
# Main app header and description
st.markdown("""
<div class="header-container">
    <h1 class="main-header">Sales Conversion Analytics</h1>
    <p class="sub-header">Interactive dashboard for analyzing lead conversion metrics and optimizing sales processes</p>
</div>
""", unsafe_allow_html=True)

# Define tabs for different dashboard sections
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Conversion Analysis", 
    "Lead Scoring", 
    "Contact Matching", 
    "Insights", 
    "Admin"
])

# Conversion Analysis Tab (First Landing Page)
with tab1:
    # Load data
    filtered_df = load_and_normalize_data()
    raw_df = filtered_df.copy()
    
    # Setup default filters (filters no longer active - displaying all data)
    filters = {
        'date_range': None,
        'status': 'All',
        'states': ['All'],
        'date_col': 'inquiry_date' if 'inquiry_date' in filtered_df.columns else 'Inquiry Date'
    }
    
    # Display KPI summary
    st.subheader("Key Performance Indicators")
    try:
        # Calculate KPIs
        total_leads = len(filtered_df)
        won = filtered_df[filtered_df['outcome'] == 1]
        conversion_rate = len(won) / total_leads if total_leads > 0 else 0
        
        # Display KPI metrics
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Leads", f"{total_leads}")
        with c2:
            st.metric("Won Deals", f"{len(won)}")
        with c3:
            st.metric("Conversion Rate", f"{conversion_rate:.1%}")
    except Exception as e:
        st.error(f"Error calculating KPIs: {e}")
    
    # Simplified notice instead of filter UI
    st.subheader("Data Overview")
    st.markdown("Showing all data without filtering. The dashboard displays your complete dataset.")
    
    # Trend Analysis Section
    st.subheader("Conversion Trends")
    
    # ... further visualization code
```

## Data Loading Function (from utils.py)

```python
def load_and_normalize_data(use_database=True):
    """
    Load and normalize data from database or sample CSVs
    
    Args:
        use_database (bool): Whether to load from database or sample CSVs
        
    Returns:
        DataFrame: Normalized and processed dataframe
    """
    if use_database:
        # Initialize database if needed
        initialize_db_if_empty()
        
        # Get merged data from database
        df = get_merged_data()
    else:
        # Load sample data
        data_dir = "data"
        leads_path = os.path.join(data_dir, "leads_sample.csv")
        operations_path = os.path.join(data_dir, "operations_sample.csv")
        
        df_leads = pd.read_csv(leads_path)
        df_operations = pd.read_csv(operations_path)
        
        # Normalize and merge data
        df = normalize_data(df_leads, df_operations)
    
    # Process and clean the data
    if df is not None and not df.empty:
        # Standardize column names
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]
        
        # Convert dates to datetime
        date_cols = ['inquiry_date', 'event_date']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Ensure outcome column exists
        if 'outcome' not in df.columns:
            if 'won' in df.columns:
                df['outcome'] = df['won'].astype(int)
            else:
                df['outcome'] = 0
    
    return df
```

## Conversion Analysis Functions (from conversion_analysis.py)

```python
def plot_time_trends(df):
    """Create trend charts for conversion rate and lead volume over time"""
    if 'inquiry_date' in df.columns:
        try:
            # Group by month and calculate rates
            df_trend = df.copy()
            df_trend['month'] = df_trend['inquiry_date'].dt.to_period('M')
            monthly = df_trend.groupby('month').agg(
                volume=('box_key', 'count'),
                won=('outcome', 'sum')
            )
            monthly['rate'] = monthly['won'] / monthly['volume']
            
            # Create figure and plot
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(monthly.index.astype(str), monthly['rate'], marker='o', linewidth=2)
            ax.set_ylabel('Conversion Rate')
            ax.set_title('Monthly Conversion Rate Trend')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error plotting time trends: {e}")

def plot_top_categories(df, col, title, min_count=10):
    """
    Plot conversion rates for top categories in a column
    
    Args:
        df (DataFrame): Dataframe with outcome data
        col (str): Column name to group by
        title (str): Title for the chart
        min_count (int): Minimum count to include a category
    """
    if col in df.columns:
        try:
            # Group by category and calculate conversion rates
            conversion_rates = df.groupby(col).agg(
                total=('box_key', 'count'),
                won=('outcome', 'sum')
            ).reset_index()
            
            conversion_rates['rate'] = conversion_rates['won'] / conversion_rates['total']
            conversion_rates = conversion_rates[conversion_rates['total'] >= min_count]
            
            # Format for display
            conversion_rates = conversion_rates.sort_values('rate', ascending=False)
            
            # Plot conversion rates
            if not conversion_rates.empty:
                fig, ax = plt.subplots(figsize=(10, 4))
                conversion_rates['rate_pct'] = conversion_rates['rate'] * 100
                
                # Create horizontal bar chart
                bars = ax.barh(conversion_rates[col], conversion_rates['rate_pct'])
                
                # Add percentage labels
                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                            f"{width:.1f}%", va='center')
                
                ax.set_xlabel('Conversion Rate (%)')
                ax.set_title(title)
                st.pyplot(fig)
                
                # Show best and worst performers
                best = conversion_rates.iloc[0]
                worst = conversion_rates.iloc[-1]
                if best['rate'] > worst['rate']:
                    best_name = best[col]
                    worst_name = worst[col]
                    diff = best['rate'] - worst['rate']
                    st.info(f"💡 **{best_name}** has a {diff:.1%} higher conversion rate than **{worst_name}**")
            else:
                st.info(f"Not enough data to analyze {col}")
                
        except Exception as e:
            st.error(f"Error plotting {col}: {e}")
    else:
        st.info(f"{col} not found in data")
```