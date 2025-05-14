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
        """)

        st.header("2. Feature Correlation")
        st.markdown("""
        - Shows how strongly each feature predicts conversion.
        - Positive values (blue) indicate features that correlate with more conversions.
        - Negative values (red) indicate features that correlate with fewer conversions.
        - Correlation ranges from -1 (strong negative) to +1 (strong positive).
        """)

        st.header("3. Lead Scoring Model")
        st.markdown("""
        - **ROC Curve**: Measures the model's ability to distinguish between won and lost deals. 
          - AUC of 0.5 = random guessing
          - AUC above 0.7 = good model
          - AUC above 0.8 = excellent model
        - **Precision-Recall Curve**: Shows the tradeoff between correctly identifying won deals vs. correctly finding all won deals.
        - **Score Distribution**: Shows how scores are distributed for won vs. lost deals.
        - **Hot/Warm/Cool/Cold**: Custom thresholds to categorize leads based on score.
        """)

        st.header("4. Time to Conversion")
        st.markdown("""
        - Measures how quickly leads convert after initial inquiry.
        - Broken down by event type to identify which kinds of events have faster booking decisions.
        - Statistics include average, median, minimum, maximum, and 90th percentile days to conversion.
        """)

        st.header("5. Phone Number Analysis")
        st.markdown("""
        - Checks if leads with local area codes convert differently than those with non-local area codes.
        - Also analyzes match rates between customers who submitted multiple forms.
        """)
        
    # Lead Personas Tab
    with tab7:
        st.title("🧩 Lead Personas")
        
        # Information about what this tab does
        st.markdown("""
        This tab uses unsupervised machine learning to discover natural "lead personas" in your data. 
        These personas can help you understand different types of leads and their conversion patterns.
        
        ### What are Lead Personas?
        Lead personas are distinct groups of leads that share similar characteristics. By identifying these natural
        groupings in your data, you can:
        
        - Discover which types of leads convert best
        - Tailor your marketing and sales approaches to different lead types
        - Understand what distinguishes high-converting from low-converting leads
        """)
        
        # Controls for segmentation
        st.subheader("Segmentation Controls")
        col1, col2 = st.columns(2)
        
        with col1:
            n_clusters = st.slider("Number of Personas", min_value=2, max_value=10, value=4, 
                                  help="How many distinct lead personas to identify")
        
        with col2:
            algorithm = st.selectbox("Clustering Algorithm", 
                                    ["K-Means", "DBSCAN", "Gaussian Mixture"],
                                    help="Different algorithms find different types of patterns")
        
        # Run clustering if we have data
        if st.button("Discover Lead Personas"):
            if "processed_df" in st.session_state and st.session_state.processed_df is not None:
                try:
                    # Get the data
                    df = filtered_df.copy()
                    
                    # Convert categorical variables to dummy variables
                    features, clusters, pca_result, cluster_stats = segment_leads(
                        df, n_clusters=n_clusters, algorithm=algorithm
                    )
                    
                    # Store results in session state
                    st.session_state.segmentation_results = {
                        "features": features,
                        "clusters": clusters,
                        "pca_result": pca_result,
                        "cluster_stats": cluster_stats,
                        "n_clusters": n_clusters,
                        "algorithm": algorithm
                    }
                    
                    # Show results
                    st.success(f"Successfully identified {n_clusters} lead personas!")
                    
                    # Display cluster visualization
                    st.subheader("Lead Persona Visualization")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig1 = plot_clusters(pca_result, clusters, n_clusters)
                        st.pyplot(fig1)
                        st.caption("Each point represents a lead, colored by persona. Similar leads are closer together.")
                    
                    with col2:
                        fig2 = plot_cluster_conversion_rates(df, clusters, n_clusters)
                        st.pyplot(fig2)
                        st.caption("Conversion rates for each lead persona, along with the proportion of leads in each.")
                    
                    # Display feature importance
                    st.subheader("What Makes Each Persona Unique")
                    fig3 = plot_feature_importance_by_cluster(features, clusters, n_clusters, top_n=5)
                    st.pyplot(fig3)
                    st.caption("The top distinguishing characteristics of each lead persona.")
                    
                    # Display cluster statistics
                    st.subheader("Persona Statistics")
                    st.dataframe(cluster_stats)
                    
                except Exception as e:
                    st.error(f"Error during lead segmentation: {str(e)}")
            else:
                st.error("Please load or select data first.")
        
        # Display previous results if available
        elif "segmentation_results" in st.session_state:
            results = st.session_state.segmentation_results
            st.success(f"Showing previously identified {results['n_clusters']} lead personas using {results['algorithm']}.")
            
            # Display cluster visualization
            st.subheader("Lead Persona Visualization")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = plot_clusters(results["pca_result"], results["clusters"], results["n_clusters"])
                st.pyplot(fig1)
                st.caption("Each point represents a lead, colored by persona. Similar leads are closer together.")
            
            with col2:
                fig2 = plot_cluster_conversion_rates(filtered_df, results["clusters"], results["n_clusters"])
                st.pyplot(fig2)
                st.caption("Conversion rates for each lead persona, along with the proportion of leads in each.")
            
            # Display feature importance
            st.subheader("What Makes Each Persona Unique")
            fig3 = plot_feature_importance_by_cluster(results["features"], results["clusters"], results["n_clusters"], top_n=5)
            st.pyplot(fig3)
            st.caption("The top distinguishing characteristics of each lead persona.")
            
            # Display cluster statistics
            st.subheader("Persona Statistics")
            st.dataframe(results["cluster_stats"])
    
    # Advanced Analytics Tab
    with tab8:
        st.title("📊 Advanced Analytics")
        
        # Information about this tab
        st.markdown("""
        This tab provides deeper insights into conversion patterns across various dimensions of your business.
        
        ### What's Included
        - **Referral Source Analysis**: Find your highest-converting referral channels
        - **Marketing Source Analysis**: Measure which marketing efforts pay off
        - **Booking Type Analysis**: See which event types convert best
        - **Price Per Guest Analysis**: Understand how pricing affects conversions
        - **Seasonality Analysis**: Discover monthly and day-of-week patterns
        - **Staff Ratio Analysis**: Optimize your staffing recommendations
        """)
        
        # Run analytics if button is clicked
        if st.button("Run Advanced Analytics"):
            if "processed_df" in st.session_state and st.session_state.processed_df is not None:
                try:
                    # Run all analytics
                    df = filtered_df.copy()
                    analytics_results = run_all_analytics(df)
                    
                    # Store in session state
                    st.session_state.analytics_results = analytics_results
                    
                    # Show results
                    st.success("Advanced analytics completed successfully!")
                    
                    # Display results
                    st.header("Results")
                    
                    # Create tabs for different analysis types
                    analysis_tabs = st.tabs([
                        "Referral Sources", 
                        "Marketing Sources", 
                        "Booking Types",
                        "Price Per Guest",
                        "Event Month",
                        "Inquiry Day",
                        "Staff Ratio"
                    ])
                    
                    # Referral Sources
                    with analysis_tabs[0]:
                        st.subheader("Referral Source Conversion Analysis")
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            if "referral_source_analysis" in analytics_results and not analytics_results["referral_source_analysis"].empty:
                                fig, ax = plt.subplots(figsize=(10, 6))
                                plot_conversion_by_category(
                                    df, 
                                    "referral_source", 
                                    "Conversion Rate by Referral Source",
                                    ax=ax,
                                    sort_by="conversion",
                                    top_n=10
                                )
                                st.pyplot(fig)
                            else:
                                st.info("No referral source data available.")
                        
                        with col2:
                            if "referral_source_analysis" in analytics_results and not analytics_results["referral_source_analysis"].empty:
                                st.dataframe(analytics_results["referral_source_analysis"])
                            else:
                                st.info("No referral source data available.")
                    
                    # Marketing Sources
                    with analysis_tabs[1]:
                        st.subheader("Marketing Source Conversion Analysis")
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            if "marketing_source_analysis" in analytics_results and not analytics_results["marketing_source_analysis"].empty:
                                fig, ax = plt.subplots(figsize=(10, 6))
                                plot_conversion_by_category(
                                    df, 
                                    "marketing_source", 
                                    "Conversion Rate by Marketing Source",
                                    ax=ax,
                                    sort_by="conversion",
                                    top_n=10
                                )
                                st.pyplot(fig)
                            else:
                                st.info("No marketing source data available.")
                        
                        with col2:
                            if "marketing_source_analysis" in analytics_results and not analytics_results["marketing_source_analysis"].empty:
                                st.dataframe(analytics_results["marketing_source_analysis"])
                            else:
                                st.info("No marketing source data available.")
                    
                    # Booking Types
                    with analysis_tabs[2]:
                        st.subheader("Booking Type Conversion Analysis")
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            if "booking_type_analysis" in analytics_results and not analytics_results["booking_type_analysis"].empty:
                                fig, ax = plt.subplots(figsize=(10, 6))
                                plot_conversion_by_category(
                                    df, 
                                    "booking_type", 
                                    "Conversion Rate by Booking Type",
                                    ax=ax,
                                    sort_by="conversion"
                                )
                                st.pyplot(fig)
                            else:
                                st.info("No booking type data available.")
                        
                        with col2:
                            if "booking_type_analysis" in analytics_results and not analytics_results["booking_type_analysis"].empty:
                                st.dataframe(analytics_results["booking_type_analysis"])
                            else:
                                st.info("No booking type data available.")
                    
                    # Price Per Guest
                    with analysis_tabs[3]:
                        st.subheader("Price Per Guest Conversion Analysis")
                        
                        if "price_per_guest_analysis" in analytics_results and not analytics_results["price_per_guest_analysis"].empty:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sns.barplot(
                                data=analytics_results["price_per_guest_analysis"],
                                x="price_per_guest_bucket",
                                y="conversion_rate",
                                ax=ax
                            )
                            ax.set_title("Conversion Rate by Price Per Guest")
                            ax.set_xlabel("Price Per Guest Range")
                            ax.set_ylabel("Conversion Rate")
                            st.pyplot(fig)
                            
                            st.dataframe(analytics_results["price_per_guest_analysis"])
                        else:
                            st.info("No price per guest data available or insufficient price/guest data.")
                    
                    # Event Month
                    with analysis_tabs[4]:
                        st.subheader("Event Month Conversion Analysis")
                        
                        if "event_month_analysis" in analytics_results and not analytics_results["event_month_analysis"].empty:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            month_order = ["January", "February", "March", "April", "May", "June", 
                                          "July", "August", "September", "October", "November", "December"]
                            
                            # Convert index to category with the right order
                            analytics_results["event_month_analysis"]["month"] = pd.Categorical(
                                analytics_results["event_month_analysis"].index,
                                categories=month_order,
                                ordered=True
                            )
                            
                            # Sort by the ordered category
                            sorted_data = analytics_results["event_month_analysis"].sort_values("month")
                            
                            # Plot
                            sns.barplot(
                                data=sorted_data,
                                x=sorted_data.index,
                                y="conversion_rate",
                                ax=ax,
                                order=month_order
                            )
                            ax.set_title("Conversion Rate by Event Month")
                            ax.set_xlabel("Event Month")
                            ax.set_ylabel("Conversion Rate")
                            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
                            st.pyplot(fig)
                            
                            st.dataframe(analytics_results["event_month_analysis"])
                        else:
                            st.info("No event month data available or insufficient event date data.")
                    
                    # Inquiry Day
                    with analysis_tabs[5]:
                        st.subheader("Inquiry Day of Week Conversion Analysis")
                        
                        if "inquiry_weekday_analysis" in analytics_results and not analytics_results["inquiry_weekday_analysis"].empty:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                            
                            # Convert index to category with the right order
                            analytics_results["inquiry_weekday_analysis"]["weekday"] = pd.Categorical(
                                analytics_results["inquiry_weekday_analysis"].index,
                                categories=weekday_order,
                                ordered=True
                            )
                            
                            # Sort by the ordered category
                            sorted_data = analytics_results["inquiry_weekday_analysis"].sort_values("weekday")
                            
                            # Plot
                            sns.barplot(
                                data=sorted_data,
                                x=sorted_data.index,
                                y="conversion_rate",
                                ax=ax,
                                order=weekday_order
                            )
                            ax.set_title("Conversion Rate by Inquiry Day of Week")
                            ax.set_xlabel("Day of Week")
                            ax.set_ylabel("Conversion Rate")
                            st.pyplot(fig)
                            
                            st.dataframe(analytics_results["inquiry_weekday_analysis"])
                        else:
                            st.info("No inquiry weekday data available or insufficient inquiry date data.")
                    
                    # Staff Ratio
                    with analysis_tabs[6]:
                        st.subheader("Staff-to-Guest Ratio Conversion Analysis")
                        
                        if "staff_ratio_analysis" in analytics_results and not analytics_results["staff_ratio_analysis"].empty:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            
                            # Sort the buckets in ascending order
                            sorted_data = analytics_results["staff_ratio_analysis"].sort_index()
                            
                            # Plot
                            sns.barplot(
                                data=sorted_data,
                                x=sorted_data.index,
                                y="conversion_rate",
                                ax=ax
                            )
                            ax.set_title("Conversion Rate by Staff-to-Guest Ratio")
                            ax.set_xlabel("Guests Per Bartender")
                            ax.set_ylabel("Conversion Rate")
                            st.pyplot(fig)
                            
                            st.dataframe(sorted_data)
                        else:
                            st.info("No staff ratio data available or insufficient bartender/guest data.")
                    
                except Exception as e:
                    st.error(f"Error during advanced analytics: {str(e)}")
            else:
                st.error("Please load or select data first.")
        
        # Display previous results if available
        elif "analytics_results" in st.session_state:
            analytics_results = st.session_state.analytics_results
            
            # Show results
            st.success("Showing previously calculated advanced analytics.")
            
            # Display results
            st.header("Results")
            
            # Create tabs for different analysis types
            analysis_tabs = st.tabs([
                "Referral Sources", 
                "Marketing Sources", 
                "Booking Types",
                "Price Per Guest",
                "Event Month",
                "Inquiry Day",
                "Staff Ratio"
            ])
            
            # Referral Sources
            with analysis_tabs[0]:
                st.subheader("Referral Source Conversion Analysis")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    if "referral_source_analysis" in analytics_results and not analytics_results["referral_source_analysis"].empty:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        plot_conversion_by_category(
                            filtered_df, 
                            "referral_source", 
                            "Conversion Rate by Referral Source",
                            ax=ax,
                            sort_by="conversion",
                            top_n=10
                        )
                        st.pyplot(fig)
                    else:
                        st.info("No referral source data available.")
                
                with col2:
                    if "referral_source_analysis" in analytics_results and not analytics_results["referral_source_analysis"].empty:
                        st.dataframe(analytics_results["referral_source_analysis"])
                    else:
                        st.info("No referral source data available.")
                        
            # Continue with the remaining tabs...
            # (omitting for brevity, but would follow the same pattern as above)

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