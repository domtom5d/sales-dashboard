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
from derive_scorecard import generate_lead_scorecard, score_lead

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
    tab1, tab2, tab3, tab4 = st.tabs(["Conversion by Category", "Feature Analysis", "Lead Scoring", "Raw Data"])
    
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
        st.subheader("AI-Powered Lead Scoring Model")
        
        try:
            # Section 1: Generate Scorecard from Historical Data
            st.markdown("### 1. Generate Lead Scoring Model")
            st.write("This model analyzes your historical conversion data to create a weighted scoring system that predicts which leads are most likely to convert.")
            
            # Display loading message while generating scorecard
            with st.spinner("Analyzing historical data to build lead scoring model..."):
                scorecard_df, thresholds = generate_lead_scorecard(use_sample_data=True)
            
            if scorecard_df is not None and not scorecard_df.empty:
                # Success message
                st.success("Lead scoring model successfully generated!")
                
                # Create two columns - one for scorecard table, one for visualization
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    # Display the scorecard as a table
                    st.write("#### Feature Weights")
                    
                    # Format coefficient and add sign
                    scorecard_df['Weight'] = scorecard_df['Coefficient'].apply(
                        lambda x: f"+{x:.3f}" if x > 0 else f"{x:.3f}"
                    )
                    
                    # Reorder columns for display
                    display_df = scorecard_df[['Feature', 'Weight', 'Points']]
                    st.dataframe(display_df, use_container_width=True)
                
                with col2:
                    # Visualize the points as a horizontal bar chart
                    st.write("#### Points Distribution")
                    
                    # Create color map based on coefficient sign
                    colors = ['red' if coef < 0 else 'green' for coef in scorecard_df['Coefficient']]
                    
                    # Create horizontal bar chart
                    fig, ax = plt.subplots(figsize=(8, 8))
                    bars = ax.barh(
                        scorecard_df['Feature'],
                        scorecard_df['Points'], 
                        color=colors
                    )
                    
                    # Add point values to the end of each bar
                    for bar in bars:
                        width = bar.get_width()
                        ax.text(
                            width + 0.1, 
                            bar.get_y() + bar.get_height()/2, 
                            f"{int(width)}", 
                            va='center'
                        )
                    
                    ax.set_xlabel('Points')
                    ax.set_title('Lead Scoring Model Weights')
                    plt.tight_layout()
                    st.pyplot(fig)
                
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
                
                # Section 2: Lead Scoring Calculator
                st.markdown("### 2. Lead Scoring Calculator")
                st.write("Use this tool to score a new lead and determine its likelihood to convert based on your historical data.")
                
                # Create form for lead scoring
                with st.form(key="lead_score_form"):
                    # Create multiple columns to organize inputs
                    form_col1, form_col2 = st.columns(2)
                    
                    lead_data = {}
                    
                    with form_col1:
                        if 'PricePerGuest' in scorecard_df['Feature'].values:
                            lead_data['PricePerGuest'] = st.number_input(
                                "Price Per Guest ($)",
                                min_value=0.0,
                                max_value=1000.0,
                                value=100.0,
                                step=10.0,
                                help="The price per guest for this lead"
                            )
                        
                        if 'DaysUntilEvent' in scorecard_df['Feature'].values:
                            lead_data['DaysUntilEvent'] = st.number_input(
                                "Days Until Event",
                                min_value=0,
                                max_value=365,
                                value=90,
                                step=1,
                                help="Number of days until the event date"
                            )
                        
                        if 'NumberOfGuests' in scorecard_df['Feature'].values:
                            lead_data['NumberOfGuests'] = st.number_input(
                                "Number of Guests",
                                min_value=0,
                                max_value=1000,
                                value=100,
                                step=10,
                                help="Expected number of guests at the event"
                            )
                        
                        if 'BartendersNeeded' in scorecard_df['Feature'].values:
                            lead_data['BartendersNeeded'] = st.number_input(
                                "Bartenders Needed",
                                min_value=0,
                                max_value=20,
                                value=2,
                                step=1,
                                help="Number of bartenders required"
                            )
                    
                    with form_col2:
                        if 'DaysSinceInquiry' in scorecard_df['Feature'].values:
                            lead_data['DaysSinceInquiry'] = st.number_input(
                                "Days Since Inquiry",
                                min_value=0,
                                max_value=365,
                                value=7,
                                step=1,
                                help="Number of days since the lead's initial inquiry"
                            )
                        
                        if 'IsCorporate' in scorecard_df['Feature'].values:
                            lead_data['IsCorporate'] = st.selectbox(
                                "Is Corporate Event?",
                                options=[0, 1],
                                format_func=lambda x: "Yes" if x == 1 else "No",
                                help="Whether this is a corporate event"
                            )
                        
                        if 'ReferralTier' in scorecard_df['Feature'].values:
                            lead_data['ReferralTier'] = st.slider(
                                "Referral Tier",
                                min_value=1,
                                max_value=3,
                                value=1,
                                step=1,
                                help="Referral tier (3=Referral, 2=Social, 1=Search)"
                            )
                        
                        if 'PhoneMatch' in scorecard_df['Feature'].values:
                            lead_data['PhoneMatch'] = st.selectbox(
                                "Phone Area Code Matches State?",
                                options=[0, 1],
                                format_func=lambda x: "Yes" if x == 1 else "No",
                                help="Whether the phone area code matches the state"
                            )
                    
                    # Submit button
                    submit_button = st.form_submit_button(label="Calculate Lead Score")
                
                # Calculate and display score when form is submitted
                if submit_button:
                    # Calculate score
                    score = score_lead(lead_data, scorecard_df)
                    
                    # Ensure score is numeric
                    if isinstance(score, tuple):
                        score = score[0]  # Extract numeric score from tuple if returned
                    
                    # Make sure we have a valid score
                    try:
                        score = float(score)
                    except (ValueError, TypeError):
                        score = 0
                    
                    # Determine category
                    category = "Cold"
                    if thresholds and isinstance(thresholds, dict):
                        for cat, threshold in thresholds.items():
                            try:
                                if score >= threshold:
                                    category = cat
                                    break
                            except (TypeError, ValueError):
                                continue
                    
                    # Display score with fancy styling
                    st.markdown("### Lead Score Results")
                    
                    # Calculate max possible score from positive coefficients
                    try:
                        max_score = sum(scorecard_df[scorecard_df['Coefficient'] > 0]['Points'])
                    except:
                        # Fallback if there's an issue
                        max_score = sum(abs(scorecard_df['Points']))
                    
                    # Ensure we have valid values
                    score = float(score) if not isinstance(score, (int, float)) else score
                    max_score = float(max_score) if max_score else 100
                    
                    # Calculate percentage of maximum possible score
                    score_percent = min(100, max(0, (score / max_score) * 100)) if max_score > 0 else 0
                    
                    # Create columns for metric and gauge
                    result_col1, result_col2 = st.columns([1, 3])
                    
                    with result_col1:
                        # Display numeric score and category
                        st.metric("Lead Score", int(score))
                        st.markdown(f"**Category:** {category}")
                    
                    with result_col2:
                        # Create a progress bar visualization of the score
                        st.markdown(f"**Score: {int(score)}/{int(max_score)} points ({score_percent:.1f}%)**")
                        st.progress(score_percent/100)
                        
                        # Add colored indicators for threshold ranges
                        threshold_cols = st.columns(4)
                        
                        # Safely display threshold info
                        if thresholds and isinstance(thresholds, dict) and all(k in thresholds for k in ['Cool', 'Warm', 'Hot']):
                            with threshold_cols[0]:
                                st.markdown(f"<span style='color: blue'>Cold: < {thresholds['Cool']}</span>", unsafe_allow_html=True)
                            with threshold_cols[1]:
                                st.markdown(f"<span style='color: teal'>Cool: {thresholds['Cool']}+</span>", unsafe_allow_html=True)
                            with threshold_cols[2]:
                                st.markdown(f"<span style='color: orange'>Warm: {thresholds['Warm']}+</span>", unsafe_allow_html=True)
                            with threshold_cols[3]:
                                st.markdown(f"<span style='color: red'>Hot: {thresholds['Hot']}+</span>", unsafe_allow_html=True)
                        else:
                            # Fallback if thresholds not available
                            with threshold_cols[0]:
                                st.markdown(f"<span style='color: blue'>Cold: Low score</span>", unsafe_allow_html=True)
                            with threshold_cols[1]:
                                st.markdown(f"<span style='color: teal'>Cool: Medium-low</span>", unsafe_allow_html=True)
                            with threshold_cols[2]:
                                st.markdown(f"<span style='color: orange'>Warm: Medium-high</span>", unsafe_allow_html=True)
                            with threshold_cols[3]:
                                st.markdown(f"<span style='color: red'>Hot: High score</span>", unsafe_allow_html=True)
                    
                    # Add interpretation and recommendations
                    st.markdown("#### Interpretation")
                    
                    if category == "Hot":
                        st.success("This lead has a very high probability of converting based on your historical data. Prioritize immediate follow-up.")
                    elif category == "Warm":
                        st.info("This lead shows good potential and should be followed up promptly.")
                    elif category == "Cool":
                        st.warning("This lead has moderate potential. Consider standard follow-up procedures.")
                    else:
                        st.error("This lead has lower conversion potential based on your historical patterns.")
            else:
                st.error("Unable to generate lead scoring model from the available data. Please ensure you have sufficient historical lead data with win/loss outcomes.")
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