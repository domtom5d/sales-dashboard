import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import base64
from utils import process_data, calculate_conversion_rates, calculate_correlations
import database as db

# Set page configuration
st.set_page_config(
    page_title="Sales Conversion Dashboard",
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
            df = db.get_lead_data()
            
            if df is None or df.empty:
                st.error("No data found in the database. Please try uploading your own data.")
                return
            
            # Process data if not empty
            if not df.empty:
                # Add Guests Bin and DaysUntilBin if not present
                if 'Guests Bin' not in df.columns and 'number_of_guests' in df.columns:
                    # Define bins for guests
                    bins_guests = [0, 50, 100, 200, np.inf]
                    labels_guests = ['0–50', '51–100', '101–200', '200+']
                    # Filter out None values before binning
                    df['number_of_guests'] = pd.to_numeric(df['number_of_guests'], errors='coerce')
                    df['Guests Bin'] = pd.cut(df['number_of_guests'].fillna(-1), bins=bins_guests, labels=labels_guests)
                
                if 'DaysUntilBin' not in df.columns and 'days_until_event' in df.columns:
                    # Define bins for days until event
                    bins_days = [0, 7, 30, 90, np.inf]
                    labels_days = ['0–7 days', '8–30 days', '31–90 days', '91+ days']
                    # Filter out None values before binning
                    df['days_until_event'] = pd.to_numeric(df['days_until_event'], errors='coerce')
                    df['DaysUntilBin'] = pd.cut(df['days_until_event'].fillna(-1), bins=bins_days, labels=labels_days)
                
                # Make column names consistent with the original data
                df.rename(columns={
                    'bartenders_needed': 'Bartenders Needed',
                    'number_of_guests': 'Number Of Guests',
                    'days_until_event': 'Days Until Event',
                    'days_since_inquiry': 'Days Since Inquiry',
                    'marketing_source': 'Marketing Source',
                    'referral_source': 'Referral Source',
                    'state': 'State',
                    'won': 'Won',
                    'lost': 'Lost',
                    'event_type': 'Event Type'
                }, inplace=True)
            
        elif data_source == "Upload Your Own Data":
            # Check if files are uploaded
            if uploaded_leads is not None:
                # Save file to temporary location
                with open('temp_leads.csv', 'wb') as f:
                    f.write(uploaded_leads.getvalue())
                
                # Import data to database
                db.import_leads_data('temp_leads.csv')
                
                # If operations data is uploaded
                if uploaded_operations is not None:
                    with open('temp_operations.csv', 'wb') as f:
                        f.write(uploaded_operations.getvalue())
                    
                    # Import operations data
                    db.import_operations_data('temp_operations.csv')
                
                # Fetch data from database
                df = db.get_lead_data()
                
                # Process data if not empty
                if not df.empty:
                    # Add Guests Bin and DaysUntilBin if not present
                    if 'Guests Bin' not in df.columns and 'number_of_guests' in df.columns:
                        # Define bins for guests
                        bins_guests = [0, 50, 100, 200, np.inf]
                        labels_guests = ['0–50', '51–100', '101–200', '200+']
                        
                        # Handle None values - replace with NaN first
                        guests = df['number_of_guests'].copy()
                        guests = pd.to_numeric(guests, errors='coerce')  # Convert to numeric, invalid values become NaN
                        
                        # Only bin non-NaN values
                        df['Guests Bin'] = pd.cut(guests, bins=bins_guests, labels=labels_guests)
                    
                    if 'DaysUntilBin' not in df.columns and 'days_until_event' in df.columns:
                        # Define bins for days until event
                        bins_days = [0, 7, 30, 90, np.inf]
                        labels_days = ['0–7 days', '8–30 days', '31–90 days', '91+ days']
                        
                        # Handle None values - replace with NaN first
                        days_until = df['days_until_event'].copy()
                        days_until = pd.to_numeric(days_until, errors='coerce')  # Convert to numeric, invalid values become NaN
                        
                        # Only bin non-NaN values
                        df['DaysUntilBin'] = pd.cut(days_until, bins=bins_days, labels=labels_days)
                    
                    # Make column names consistent with the original data
                    df.rename(columns={
                        'bartenders_needed': 'Bartenders Needed',
                        'number_of_guests': 'Number Of Guests',
                        'days_until_event': 'Days Until Event',
                        'days_since_inquiry': 'Days Since Inquiry',
                        'marketing_source': 'Marketing Source',
                        'referral_source': 'Referral Source',
                        'state': 'State',
                        'won': 'Won',
                        'lost': 'Lost',
                        'event_type': 'Event Type'
                    }, inplace=True)
        
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
        
        # Days Until Event filter
        if 'DaysUntilBin' in df.columns:
            days_bins = ['All'] + sorted(df['DaysUntilBin'].dropna().unique().tolist())
            selected_days_bin = st.selectbox("Days Until Event", days_bins)
    
    # Apply filters to dataframe
    filtered_df = df.copy()
    
    if 'Event Type' in df.columns and selected_event_type != 'All':
        filtered_df = filtered_df[filtered_df['Event Type'] == selected_event_type]
    
    if 'Referral Source' in df.columns and selected_referral_source != 'All':
        filtered_df = filtered_df[filtered_df['Referral Source'] == selected_referral_source]
    
    if 'Marketing Source' in df.columns and selected_marketing_source != 'All':
        filtered_df = filtered_df[filtered_df['Marketing Source'] == selected_marketing_source]
    
    if 'State' in df.columns and selected_state != 'All':
        filtered_df = filtered_df[filtered_df['State'] == selected_state]
    
    if 'Guests Bin' in df.columns and selected_guest_bin != 'All':
        filtered_df = filtered_df[filtered_df['Guests Bin'] == selected_guest_bin]
    
    if 'DaysUntilBin' in df.columns and selected_days_bin != 'All':
        filtered_df = filtered_df[filtered_df['DaysUntilBin'] == selected_days_bin]
    
    # Calculate metrics
    total_leads = filtered_df.shape[0]
    
    # Handle potential None values
    if 'Won' in filtered_df.columns:
        # Convert boolean strings to actual booleans if needed
        if filtered_df['Won'].dtype == 'object':
            filtered_df['Won'] = filtered_df['Won'].map({'True': True, 'False': False})
            filtered_df['Won'] = filtered_df['Won'].fillna(False)
        
        won_leads = filtered_df['Won'].sum()
    else:
        won_leads = 0
    
    if 'Lost' in filtered_df.columns:
        # Convert boolean strings to actual booleans if needed
        if filtered_df['Lost'].dtype == 'object':
            filtered_df['Lost'] = filtered_df['Lost'].map({'True': True, 'False': False})
            filtered_df['Lost'] = filtered_df['Lost'].fillna(False)
            
        lost_leads = filtered_df['Lost'].sum()
    else:
        lost_leads = 0
    
    conversion_rate = won_leads / total_leads if total_leads > 0 else 0
    
    # Summary metrics section
    st.header("Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Leads", f"{total_leads:,}")
    
    with col2:
        st.metric("Won Deals", f"{won_leads:,}")
    
    with col3:
        st.metric("Lost Deals", f"{lost_leads:,}")
    
    with col4:
        st.metric("Conversion Rate", f"{conversion_rate:.1%}")
    
    st.markdown("---")
    
    # Conversion rates by different factors
    st.header("Conversion Rates Analysis")
    
    tab1, tab2, tab3 = st.tabs(["By Category", "By Numeric Features", "Raw Data"])
    
    with tab1:
        # Calculate conversion rates for different categories
        try:
            # Check if we have required columns
            if 'Outcome' not in filtered_df.columns:
                # Add Outcome column if it doesn't exist
                if 'Won' in filtered_df.columns and 'Lost' in filtered_df.columns:
                    filtered_df['Outcome'] = filtered_df['Won'].astype(int)
            
            conv_rates = calculate_conversion_rates(filtered_df)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Conversion by Event Type
                if conv_rates and 'Event Type' in conv_rates:
                    st.subheader("Conversion by Event Type")
                    conv_event_type = conv_rates['Event Type'].sort_values(by='Conversion Rate', ascending=False)
                    
                    # Create bar chart using Plotly
                    fig = px.bar(
                        conv_event_type,
                        x='Event Type',
                        y='Conversion Rate',
                        color='Conversion Rate',
                        color_continuous_scale='blues',
                        text_auto=False
                    )
                    fig.update_layout(xaxis_title="Event Type", yaxis_title="Conversion Rate")
                    fig.update_traces(textposition='outside')
                    st.plotly_chart(fig, key="event_type_chart", use_container_width=True)
                
                # Conversion by Marketing Source
                if conv_rates and 'Marketing Source' in conv_rates:
                    st.subheader("Conversion by Marketing Source")
                    conv_marketing = conv_rates['Marketing Source'].sort_values(by='Conversion Rate', ascending=False)
                    
                    fig = px.bar(
                        conv_marketing,
                        x='Marketing Source',
                        y='Conversion Rate',
                        color='Conversion Rate',
                        color_continuous_scale='blues',
                        text_auto=False
                    )
                    fig.update_layout(xaxis_title="Marketing Source", yaxis_title="Conversion Rate")
                    fig.update_traces(textposition='outside')
                    st.plotly_chart(fig, key="marketing_chart", use_container_width=True)
                
                # Conversion by Number of Guests
                if conv_rates and 'Guests Bin' in conv_rates:
                    st.subheader("Conversion by Number of Guests")
                    conv_guests = conv_rates['Guests Bin']
                    
                    # Ensure the bins are displayed in the correct order
                    bin_order = ['0–50', '51–100', '101–200', '200+']
                    conv_guests['Guests Bin'] = pd.Categorical(
                        conv_guests['Guests Bin'],
                        categories=bin_order,
                        ordered=True
                    )
                    conv_guests = conv_guests.sort_values('Guests Bin')
                    
                    fig = px.bar(
                        conv_guests,
                        x='Guests Bin',
                        y='Conversion Rate',
                        color='Conversion Rate',
                        color_continuous_scale='blues',
                        text_auto=False
                    )
                    fig.update_layout(xaxis_title="Number of Guests", yaxis_title="Conversion Rate")
                    fig.update_traces(textposition='outside')
                    st.plotly_chart(fig, key="guests_chart", use_container_width=True)
            
            with col2:
                # Conversion by Referral Source
                if conv_rates and 'Referral Source' in conv_rates:
                    st.subheader("Conversion by Referral Source")
                    conv_referral = conv_rates['Referral Source'].sort_values(by='Conversion Rate', ascending=False)
                    
                    fig = px.bar(
                        conv_referral,
                        x='Referral Source',
                        y='Conversion Rate',
                        color='Conversion Rate',
                        color_continuous_scale='blues',
                        text_auto=False
                    )
                    fig.update_layout(xaxis_title="Referral Source", yaxis_title="Conversion Rate")
                    fig.update_traces(textposition='outside')
                    st.plotly_chart(fig, key="referral_chart", use_container_width=True)
                
                # Conversion by State
                if conv_rates and 'State' in conv_rates:
                    st.subheader("Conversion by State")
                    conv_state = conv_rates['State'].sort_values(by='Conversion Rate', ascending=False)
                    
                    fig = px.bar(
                        conv_state,
                        x='State',
                        y='Conversion Rate',
                        color='Conversion Rate',
                        color_continuous_scale='blues',
                        text_auto=False
                    )
                    fig.update_layout(xaxis_title="State", yaxis_title="Conversion Rate")
                    fig.update_traces(textposition='outside')
                    st.plotly_chart(fig, key="state_chart", use_container_width=True)
                
                # Conversion by Days Until Event
                if conv_rates and 'DaysUntilBin' in conv_rates:
                    st.subheader("Conversion by Days Until Event")
                    conv_days = conv_rates['DaysUntilBin']
                    
                    # Ensure the bins are displayed in the correct order
                    bin_order = ['0–7 days', '8–30 days', '31–90 days', '91+ days']
                    conv_days['DaysUntilBin'] = pd.Categorical(
                        conv_days['DaysUntilBin'],
                        categories=bin_order,
                        ordered=True
                    )
                    conv_days = conv_days.sort_values('DaysUntilBin')
                    
                    fig = px.bar(
                        conv_days,
                        x='DaysUntilBin',
                        y='Conversion Rate',
                        color='Conversion Rate',
                        color_continuous_scale='blues',
                        text_auto=False
                    )
                    fig.update_layout(xaxis_title="Days Until Event", yaxis_title="Conversion Rate")
                    fig.update_traces(textposition='outside')
                    st.plotly_chart(fig, key="days_chart", use_container_width=True)
        except Exception as e:
            st.error(f"Error generating conversion rate charts: {str(e)}")
    
    with tab2:
        # Feature correlations with outcome
        st.subheader("Feature Correlation with Outcome")
        
        try:
            # Calculate correlations
            corr_df = calculate_correlations(filtered_df)
            
            if not corr_df.empty:
                # Remove the 'Outcome' row if it exists (it will always have correlation 1.0)
                corr_df = corr_df[corr_df['index'] != 'Outcome']
                
                # Create bar chart of correlations
                fig = px.bar(
                    corr_df,
                    x='index',
                    y='Correlation with Outcome',
                    color='Correlation with Outcome',
                    color_continuous_scale='blues',
                    text_auto=False
                )
                fig.update_layout(xaxis_title="Feature", yaxis_title="Correlation Strength")
                fig.update_traces(textposition='outside')
                st.plotly_chart(fig, key="correlation_chart", use_container_width=True)
                
                # Scatter plots for each numeric feature
                st.subheader("Relationship Between Numeric Features and Conversion")
                
                numeric_features = [col for col in ['Days Since Inquiry', 'Days Until Event', 
                                                'Number Of Guests', 'Bartenders Needed']
                                if col in filtered_df.columns]
                
                for i, feature in enumerate(numeric_features):
                    try:
                        temp_df = filtered_df[[feature, 'Outcome']].copy()
                        # Convert to numeric and handle missing values
                        temp_df[feature] = pd.to_numeric(temp_df[feature], errors='coerce')
                        temp_df = temp_df.dropna()
                        
                        if not temp_df.empty:
                            fig = px.scatter(
                                temp_df,
                                x=feature,
                                y='Outcome',
                                color='Outcome',
                                color_continuous_scale='blues',
                                trendline='ols',
                                labels={'Outcome': 'Won (1) / Lost (0)'}
                            )
                            fig.update_layout(title=f"{feature} vs. Outcome")
                            st.plotly_chart(fig, key=f"scatter_{i}", use_container_width=True)
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
        fig1 = go.Figure()
        fig1.add_annotation(
            text="No data available",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=20)
        )
        fig1.update_layout(height=300)
        st.plotly_chart(fig1, key="placeholder_chart1", use_container_width=True)
    
    with col2:
        st.subheader("Feature Correlation")
        st.write("Select data source to see feature correlations with outcome.")
        
        # Placeholder for empty chart
        fig2 = go.Figure()
        fig2.add_annotation(
            text="No data available",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=20)
        )
        fig2.update_layout(height=300)
        st.plotly_chart(fig2, key="placeholder_chart2", use_container_width=True)

# Add footer
st.markdown("---")
st.markdown("Streak Export Analysis Dashboard | Built with Streamlit")