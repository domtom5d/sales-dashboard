import streamlit as st
import pandas as pd
import numpy as np
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from utils import process_data, calculate_conversion_rates, calculate_correlations
import database as db
from scipy import stats

# Set matplotlib style for more modern-looking plots
plt.style.use('ggplot')
sns.set(style="whitegrid")

# Set page configuration
st.set_page_config(
    page_title="Streak Export Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# Dashboard Title
st.title("📊 Sales Conversion Analytics Dashboard")
st.markdown("Analyze conversion rates and patterns from your Streak exports.")

# Initialize upload variables before the sidebar
uploaded_leads = None
uploaded_operations = None

# Sidebar for file uploads and filters
with st.sidebar:
    st.header("Data Source")
    
    data_source = st.radio(
        "Select data source",
        ["Sample Data", "Upload Your Own Data"],
        key="data_source"
    )
    
    if data_source == "Upload Your Own Data":
        uploaded_leads = st.file_uploader("Upload Leads Data (CSV)", type="csv", key="leads_uploader")
        uploaded_operations = st.file_uploader("Upload Operations Data (CSV)", type="csv", key="operations_uploader")
    
    st.markdown("---")
    st.header("Filters")
    
    # These filters will be populated after data is loaded

# Initialize variables for data and filters
df = None
filtered_df = None
selected_event_type = 'All'
selected_referral_source = 'All'
selected_marketing_source = 'All'
selected_state = 'All'
selected_guest_bin = 'All'
selected_days_bin = 'All'

# Function to load and process data
def load_data():
    global df
    
    try:
        if data_source == "Sample Data":
            # Initialize database with sample data if needed
            db.initialize_db_if_empty()
            
            # Fetch data from database
            leads_df = db.get_lead_data()
            operations_df = db.get_operation_data()
            
            if leads_df is None or leads_df.empty:
                st.error("No data found in the database. Please try uploading your own data.")
                return
            
            # Process data using our enhanced utils function that includes all the advanced features
            df = process_data(leads_df, operations_df)
            
        elif data_source == "Upload Your Own Data":
            # Check if files are uploaded
            if uploaded_leads is not None:
                # Save file to temporary location
                with open('temp_leads.csv', 'wb') as f:
                    f.write(uploaded_leads.getvalue())
                
                # Import data to database
                db.import_leads_data('temp_leads.csv')
                
                # Initialize operations_df
                operations_df = None
                
                # If operations data is uploaded
                if uploaded_operations is not None:
                    with open('temp_operations.csv', 'wb') as f:
                        f.write(uploaded_operations.getvalue())
                    
                    # Import operations data
                    db.import_operations_data('temp_operations.csv')
                    
                    # Get operations data from database
                    operations_df = db.get_operation_data()
                
                # Fetch data from database
                leads_df = db.get_lead_data()
                
                # Process data using enhanced utils function with all advanced features
                if not leads_df.empty:
                    df = process_data(leads_df, operations_df)
                else:
                    st.error("No data found in the database after upload.")
                    return
            else:
                st.info("Please upload your data files.")
                return
        
        return df is not None and not df.empty
    except Exception as e:
        st.error(f"Error loading or processing data: {str(e)}")
        st.exception(e)
        return False

# Load data
data_loaded = load_data()

# If data is loaded successfully, populate filters and display content
if data_loaded and df is not None:
    # Populate filters
    with st.sidebar:
        # Event Type filter
        if 'Event Type' in df.columns:
            event_types = ['All'] + sorted(df['Event Type'].dropna().unique().tolist())
            selected_event_type = st.selectbox("Event Type", event_types)
        
        # Referral Source filter
        if 'Referral Source' in df.columns:
            referral_sources = ['All'] + sorted(df['Referral Source'].dropna().unique().tolist())
            selected_referral_source = st.selectbox("Referral Source", referral_sources)
        
        # Marketing Source filter
        if 'Marketing Source' in df.columns:
            marketing_sources = ['All'] + sorted(df['Marketing Source'].dropna().unique().tolist())
            selected_marketing_source = st.selectbox("Marketing Source", marketing_sources)
        
        # State filter
        if 'State' in df.columns:
            states = ['All'] + sorted(df['State'].dropna().unique().tolist())
            selected_state = st.selectbox("State", states)
        
        # Guests Bin filter
        if 'Guests Bin' in df.columns:
            guest_bins = ['All'] + sorted(df['Guests Bin'].dropna().unique().tolist())
            selected_guest_bin = st.selectbox("Number of Guests", guest_bins)
        
        # Days Until Event Bin filter
        if 'DaysUntilBin' in df.columns:
            days_bins = ['All'] + sorted(df['DaysUntilBin'].dropna().unique().tolist())
            selected_days_bin = st.selectbox("Days Until Event", days_bins)
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_event_type != 'All' and 'Event Type' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Event Type'] == selected_event_type]
    
    if selected_referral_source != 'All' and 'Referral Source' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Referral Source'] == selected_referral_source]
    
    if selected_marketing_source != 'All' and 'Marketing Source' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Marketing Source'] == selected_marketing_source]
    
    if selected_state != 'All' and 'State' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['State'] == selected_state]
    
    if selected_guest_bin != 'All' and 'Guests Bin' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Guests Bin'] == selected_guest_bin]
    
    if selected_days_bin != 'All' and 'DaysUntilBin' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['DaysUntilBin'] == selected_days_bin]
    
    # Summary statistics
    st.header("Conversion Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_leads = len(filtered_df)
        st.metric("Total Leads", total_leads)
    
    with col2:
        if 'Won' in filtered_df.columns:
            won_deals = filtered_df['Won'].sum()
            st.metric("Won Deals", won_deals)
    
    with col3:
        if 'Lost' in filtered_df.columns:
            lost_deals = filtered_df['Lost'].sum()
            st.metric("Lost Deals", lost_deals)
    
    with col4:
        if 'Outcome' in filtered_df.columns:
            conversion_rate = (filtered_df['Outcome'].mean() * 100).round(1)
            st.metric("Conversion Rate", f"{conversion_rate}%")
    
    # Tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Conversion by Category", "Feature Analysis", "Raw Data"])
    
    with tab1:
        try:
            # Calculate conversion rates by category
            conversion_rates = calculate_conversion_rates(filtered_df)
            
            if conversion_rates:
                # Create a layout with 2 columns
                cols = st.columns(2)
                
                # Plot conversion rates by category
                for i, (category, conv_df) in enumerate(conversion_rates.items()):
                    with cols[i % 2]:  # Alternate between columns
                        # Sort by conversion rate
                        conv_df = conv_df.sort_values('Conversion Rate', ascending=False)
                        
                        # Create chart
                        fig, ax = plt.subplots(figsize=(10, 6))
                        bars = ax.barh(conv_df.iloc[:, 0], conv_df['Conversion Rate'], color='steelblue')
                        
                        # Add percentage labels
                        for bar in bars:
                            width = bar.get_width()
                            ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, f'{width:.1%}', 
                                   va='center')
                        
                        # Chart formatting
                        ax.set_title(f"Conversion Rate by {category}")
                        ax.set_xlabel("Conversion Rate")
                        ax.set_xlim(0, min(1, conv_df['Conversion Rate'].max() * 1.2))  # Cap at 100%
                        plt.tight_layout()
                        
                        # Display chart
                        st.pyplot(fig)
            else:
                st.info("No conversion rate data available for the current selection.")
        except Exception as e:
            st.error(f"Error generating conversion rate charts: {str(e)}")
    
    with tab2:
        # Feature correlations with outcome
        st.subheader("Feature Correlation with Outcome")
        
        try:
            # Calculate correlations
            corr_df, full_corr_matrix = calculate_correlations(filtered_df)
            
            # Check if we have correlation data
            if isinstance(corr_df, pd.DataFrame) and not corr_df.empty:
                # Create bar chart of correlations with matplotlib
                fig, ax = plt.subplots(figsize=(10, 6))
                if 'index' in corr_df.columns and 'Correlation with Outcome' in corr_df.columns:
                    # Remove 'Outcome' from correlations (it'll always be 1.0)
                    corr_df = corr_df[corr_df['index'] != 'Outcome']
                    # Sort by correlation strength
                    corr_df = corr_df.sort_values('Correlation with Outcome', ascending=False)
                    
                    bars = ax.barh(corr_df['index'], corr_df['Correlation with Outcome'], color='steelblue')
                    ax.set_xlabel('Correlation Strength')
                    ax.set_ylabel('Feature')
                    ax.set_title('Feature Correlation with Outcome')
                    # Add values to bars
                    for bar in bars:
                        width = bar.get_width()
                        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.2f}', 
                                ha='left', va='center')
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Correlation heatmap using seaborn (much prettier than plotly for this)
                if isinstance(full_corr_matrix, pd.DataFrame) and not full_corr_matrix.empty:
                    st.subheader("Feature Correlation Heatmap")
                    
                    # Filter to only numeric columns we care about
                    numeric_cols = [
                        'Price Per Guest', 'Event Duration Hours', 'Number Of Guests',
                        'Days Until Event', 'Days Since Inquiry', 'Bartenders Needed',
                        'Is Corporate', 'Referral Tier', 'Actual Deal Value', 'Outcome'
                    ]
                    
                    # Keep only columns that exist in the dataframe
                    heatmap_cols = [c for c in numeric_cols if c in full_corr_matrix.columns]
                    
                    if len(heatmap_cols) > 1:  # Need at least 2 columns for a correlation
                        # Create correlation matrix for selected columns
                        corr_matrix = full_corr_matrix[heatmap_cols].corr()
                        
                        # Create heatmap
                        fig, ax = plt.subplots(figsize=(12, 10))
                        sns.heatmap(
                            corr_matrix,
                            annot=True,      # Show correlation values
                            fmt='.2f',       # Format to 2 decimal places
                            cmap='coolwarm', # Blue to red colormap
                            square=True,     # Make cells square
                            cbar_kws={'shrink': .75},
                            ax=ax
                        )
                        ax.set_title('Feature Correlation Matrix')
                        plt.tight_layout()  # Fixes layout issues
                        st.pyplot(fig)
                
                # Scatter plots for key numeric features
                st.subheader("Feature Relationship to Outcome")
                
                # Prioritize the advanced features but include originals too
                key_features = [
                    'Price Per Guest', 'Event Duration Hours', 
                    'Days Until Event', 'Number Of Guests',
                    'Bartenders Needed', 'Is Corporate', 'Referral Tier'
                ]
                
                # Filter to only features present in the dataset
                available_features = [f for f in key_features if f in filtered_df.columns]
                
                # Display in a grid - 2 columns
                cols = st.columns(2)
                
                for i, feature in enumerate(available_features):
                    try:
                        # Create a new figure for each plot
                        fig, ax = plt.subplots(figsize=(8, 5))
                        
                        # Extract data and handle missing values
                        temp_df = filtered_df[[feature, 'Outcome']].copy()
                        temp_df[feature] = pd.to_numeric(temp_df[feature], errors='coerce')
                        temp_df = temp_df.dropna()
                        
                        if not temp_df.empty:
                            # Add jitter to the outcome for better visualization (0 and 1 would overlap)
                            jittered_outcome = temp_df['Outcome'] + np.random.normal(0, 0.05, len(temp_df))
                            
                            # Create scatter plot
                            ax.scatter(
                                temp_df[feature], 
                                jittered_outcome,
                                alpha=0.6,        # Make points semi-transparent
                                color='steelblue',
                                edgecolor='white',
                                s=70              # Point size
                            )
                            
                            # Add trend line
                            try:
                                slope, intercept, r_value, p_value, std_err = stats.linregress(
                                    temp_df[feature], temp_df['Outcome']
                                )
                                x = np.array([min(temp_df[feature]), max(temp_df[feature])])
                                y = slope * x + intercept
                                ax.plot(x, y, color='red', linestyle='--', linewidth=2)
                                # Add R-squared to plot
                                ax.text(
                                    0.05, 0.95, f'R² = {r_value**2:.3f}', 
                                    transform=ax.transAxes, 
                                    fontsize=12, 
                                    verticalalignment='top'
                                )
                            except Exception as e:
                                st.error(f"Error generating trendline: {e}")
                            
                            # Formatting
                            ax.set_xlabel(feature)
                            ax.set_ylabel('Outcome (Won=1, Lost=0)')
                            ax.set_title(f'{feature} vs. Outcome')
                            ax.grid(True, linestyle='--', alpha=0.7)
                            
                            # Add a horizontal line at y=0.5 to show decision boundary
                            ax.axhline(y=0.5, color='gray', linestyle='-', alpha=0.3)
                            
                            plt.tight_layout()
                            
                            # Display in alternating columns
                            with cols[i % 2]:
                                st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Error generating scatter plot for {feature}: {str(e)}")
            else:
                st.info("No correlation data available for the current selection.")
        except Exception as e:
            st.error(f"Error generating correlation charts: {str(e)}")
    
    with tab3:
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

# Add footer
st.markdown("---")
st.markdown("Streak Export Analysis Dashboard | Built with Streamlit")