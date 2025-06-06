import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
import os
import datetime
from sqlalchemy import create_engine
from database import import_leads_data, import_operations_data, get_lead_data, get_operation_data, get_merged_data, initialize_db_if_empty
from utils import process_data, calculate_conversion_rates, calculate_correlations
from derive_scorecard import generate_lead_scorecard, score_lead

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
    tab1, tab2, tab3, tab4 = st.tabs([
        "Conversion Analysis", 
        "Feature Correlation", 
        "Lead Scoring", 
        "Raw Data"
    ])
    
    with tab1:
        try:
            # Calculate conversion rates by different categories
            conversion_rates = calculate_conversion_rates(filtered_df)
            
            st.subheader("Conversion Rate Analysis")
            
            # Summary metrics
            overall_conversion = conversion_rates["overall"]["Conversion Rate"][0]
            st.metric("Overall Conversion Rate", f"{overall_conversion:.1%}")
            
            # Create columns for metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Plot conversion by booking type
                st.write("#### Conversion by Booking Type")
                fig, ax = plt.subplots(figsize=(8, 5))
                conversion_rates["booking_type"].plot(kind="bar", x="Booking Type", y="Conversion Rate", ax=ax)
                ax.set_xlabel("Booking Type")
                ax.set_ylabel("Conversion Rate")
                ax.set_ylim(0, min(1, conversion_rates["booking_type"]["Conversion Rate"].max() * 1.5))
                st.pyplot(fig)
            
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
            top_positive = corr_outcome[corr_outcome["Correlation with Outcome"] > 0].iloc[0:3]
            top_negative = corr_outcome[corr_outcome["Correlation with Outcome"] < 0].iloc[-3:]
            
            if not top_positive.empty:
                st.write("**Top Positive Factors:**")
                for idx, row in top_positive.iterrows():
                    st.write(f"• {idx}: +{row['Correlation with Outcome']:.3f}")
            
            if not top_negative.empty:
                st.write("**Top Negative Factors:**")
                for idx, row in top_negative.iterrows():
                    st.write(f"• {idx}: {row['Correlation with Outcome']:.3f}")
            
            # Plot correlation matrix for top features
            st.write("#### Feature Correlation Matrix")
            top_features = pd.concat([top_positive, top_negative]).index.tolist()
            top_features.append("Outcome")
            top_corr_matrix = corr_matrix.loc[top_features, top_features]
            
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(top_corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
            
            # Add feature labels
            ax.set_xticks(np.arange(len(top_features)))
            ax.set_yticks(np.arange(len(top_features)))
            ax.set_xticklabels(top_features, rotation=45, ha="right")
            ax.set_yticklabels(top_features)
            
            # Add colorbar
            plt.colorbar(im)
            
            # Add correlation values
            for i in range(len(top_features)):
                for j in range(len(top_features)):
                    text = ax.text(j, i, f"{top_corr_matrix.iloc[i, j]:.2f}",
                                  ha="center", va="center", color="black" if abs(top_corr_matrix.iloc[i, j]) < 0.7 else "white")
            
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error in feature correlation analysis: {str(e)}")
    
    with tab3:
        try:
            st.subheader("Lead Scoring Model")
            
            # Button to generate lead scoring model
            if st.button("Generate Lead Scoring Model"):
                # Generate the lead scoring model
                scorecard_df, thresholds = generate_lead_scorecard(use_sample_data=False)
                
                if scorecard_df is not None and thresholds is not None:
                    st.session_state.weights_df = scorecard_df
                    st.session_state.thresholds = thresholds
                    
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