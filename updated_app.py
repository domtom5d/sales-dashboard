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
from utils import load_and_normalize_data
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

# Define tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Conversion Analysis", 
    "Lead Scoring", 
    "Contact Matching", 
    "Insights", 
    "Admin"
])

# Load data - use a single source of truth without filtering
filtered_df = load_and_normalize_data()
raw_df = filtered_df.copy()

# Setup default filters for backward compatibility
if 'date_filter' not in st.session_state:
    # Set default date range to last 90 days if data is available
    if 'inquiry_date' in filtered_df.columns:
        max_date = filtered_df['inquiry_date'].max()
        if pd.notna(max_date):
            default_start = max_date - datetime.timedelta(days=90)
            st.session_state.date_filter = (default_start.date(), max_date.date())
        else:
            st.session_state.date_filter = None
    else:
        st.session_state.date_filter = None

if 'status_filter' not in st.session_state:
    st.session_state.status_filter = 'All'

if 'region_filter' not in st.session_state:
    st.session_state.region_filter = ['All']

# Conversion Analysis Tab
with tab1:
    # --- Conversion Analysis Tab (with filters disabled) ---
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    st.markdown(
        "## Conversion Analysis  \n"
        "*Filters are currently disabled – showing all data.*"
    )

    # Use the unfiltered dataframe
    df = filtered_df

    # KPI Cards
    total = len(df)
    won   = int(df['outcome'].sum()) if 'outcome' in df else 0
    lost  = total - won
    rate  = won/total if total else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Leads",     f"{total:,}")
    c2.metric("Won Deals",       f"{won:,}")
    c3.metric("Lost Deals",      f"{lost:,}")
    c4.metric("Conversion Rate", f"{rate:.1%}")

    # Trends
    st.subheader("Conversion Trends")
    date_col = None
    for col_name in ['inquiry_date', 'created', 'created_at', 'date']:
        if col_name in df.columns:
            date_col = col_name
            break
            
    if date_col:
        df['created_dt'] = pd.to_datetime(df[date_col], errors='coerce')
        weekly = (
            df.dropna(subset=['created_dt','outcome'])
              .set_index('created_dt')
              .resample('W')['outcome']
              .agg(['size','sum'])
        )
        if not weekly.empty:
            weekly['rate'] = weekly['sum'] / weekly['size']
            fig, ax = plt.subplots(1,2, figsize=(10,3))
            ax[0].plot(weekly.index, weekly['rate'], marker='o')
            ax[0].set_title("Conversion Rate Over Time")
            ax[0].set_ylabel("Rate")
            ax[1].bar(weekly.index, weekly['size'])
            ax[1].set_title("Lead Volume Over Time")
            st.pyplot(fig)
        else:
            st.info("No timestamped data to build trend charts.")
    else:
        st.info("No date field for trend analysis.")

    # Helper: safe category plot
    def plot_category(col, title, min_count=10):
        st.subheader(title)
        if col not in df or df[col].dropna().empty:
            st.info(f"No {title} data available")
            return
        cats = df[col].fillna("Missing").astype(str)
        counts = cats.value_counts()
        keep = counts[counts>=min_count].index
        cats = cats.where(cats.isin(keep), "Other")
        summary = cats.to_frame("cat").join(df['outcome']).groupby("cat") \
                     .agg(total=("outcome","size"), won=("outcome","sum"))
        summary['rate'] = summary['won']/summary['total']
        top = summary.sort_values('rate', ascending=False).head(5)
        
        if not top.empty:  # Check if we have any data to plot
            fig, ax = plt.subplots()
            ax.barh(top.index, top['rate'], color='teal')
            for i,(idx,row) in enumerate(top.iterrows()):
                ax.text(row['rate']+0.005, i,
                        f"{row['rate']:.1%} (n={int(row['total'])})",
                        va='center')
            ax.set_xlabel("Conversion Rate")
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

    # Conversion by Category
    plot_category('booking_type',      "Conversion by Booking Type")
    plot_category('referral_source',   "Conversion by Referral Source")
    plot_category('event_type',        "Conversion by Event Type")
    plot_category('marketing_source',  "Conversion by Marketing Source")

    # Helper: safe numeric-bin plot
    def plot_numeric_bin(col, title, bins, labels):
        st.subheader(title)
        if col not in df:
            st.info(f"No {title} data available")
            return
        nums = pd.to_numeric(df[col], errors='coerce')
        valid = df.loc[nums.notna()].copy()
        valid['bin'] = pd.cut(nums.dropna(), bins=bins, labels=labels)
        summary = valid.groupby('bin')['outcome'] \
                       .agg(total='size', won='sum')
        if summary.empty:
            st.info(f"No valid {title} data to plot.")
            return
        summary['rate'] = summary['won']/summary['total']
        st.bar_chart(summary['rate'])
        
        # Display table with formatted rates
        display_summary = summary.copy()
        display_summary['rate'] = display_summary['rate'].apply(lambda x: f"{x:.1%}")
        st.table(display_summary)

    # Timing Factors
    plot_numeric_bin(
        'days_until_event', "Conversion by Days Until Event",
        bins=[0,7,30,90,np.inf], labels=['≤7d','8–30d','31–90d','90+d']
    )
    plot_numeric_bin(
        'days_since_inquiry', "Conversion by Days Since Inquiry",
        bins=[0,1,3,7,30,np.inf], labels=['0d','1–3d','4–7d','8–30d','30+d']
    )

    # Weekday
    st.subheader("Conversion by Weekday")
    if 'created_dt' in df:
        wd = df.dropna(subset=['created_dt']).copy()
        wd['weekday'] = wd['created_dt'].dt.day_name()
        rates = wd.groupby('weekday')['outcome'].mean().reindex(
            ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        ).fillna(0)
        st.bar_chart(rates)
    else:
        st.info("No timestamp data for weekday analysis.")

    # Event Month
    st.subheader("Conversion by Event Month")
    if 'event_date' in df:
        df['event_dt'] = pd.to_datetime(df['event_date'], errors='coerce')
        mf = df.dropna(subset=['event_dt']).copy()
        mf['month'] = mf['event_dt'].dt.month_name()
        rates = mf.groupby('month')['outcome'].mean()
        st.bar_chart(rates)
    else:
        st.info("No event_date field for month analysis.")

    # Price & Size
    plot_numeric_bin(
        'number_of_guests', "Conversion by Number of Guests",
        bins=[0,50,100,200,500,np.inf], labels=['1–50','51–100','101–200','201–500','500+']
    )
    
    # Staff-to-guest
    st.subheader("Conversion by Staff-to-Guest Ratio")
    if {'bartenders_needed','number_of_guests'}.issubset(df.columns):
        ratio = df['bartenders_needed']/df['number_of_guests'].replace(0,np.nan)
        df['ratio_bin'] = pd.cut(ratio, bins=[0,0.01,0.02,0.05,np.inf],
                                 labels=['<1%','1–2%','2–5%','5%+'])
        sr = df.groupby('ratio_bin')['outcome'].mean().fillna(0)
        st.bar_chart(sr)
    else:
        st.info("Insufficient data for staffing analysis.")

    # Geography
    plot_category('state', "Conversion by State/Region", min_count=20)

    # Phone-match
    st.subheader("Conversion by Phone Area Code Match")
    if 'phone_number' in df.columns and 'state' in df.columns:
        # More graceful approach to phone matching
        try:
            phone_match_data = analyze_phone_matches(df)
            if phone_match_data:
                match_conv, match_counts = phone_match_data
                if not match_conv.empty:
                    st.table(match_conv)
                else:
                    st.info("No valid phone match data available.")
            else:
                st.info("Phone matching analysis returned no results.")
        except Exception as e:
            # Fallback to a simpler approach if the external function fails
            st.info("Unable to perform detailed phone analysis. Using simplified approach.")
            # dummy example: local if area code starts with same letter as state
            df['phone_match'] = df['phone_number'].astype(str).str[:1] == df['state'].astype(str).str[:1]
            pm = df.groupby('phone_match')['outcome'].mean()
            st.table(pm.map(lambda v: f"{v:.1%}"))
    else:
        st.info("Insufficient data for phone-match analysis.")

    # Data Quality & Anomalies
    st.subheader("Data Quality & Anomalies")
    miss = df.isna().mean().mul(100).round(1)
    st.table(miss.to_frame("% Missing"))

    if 'days_until_event' in df.columns:
        neg = pd.to_numeric(df['days_until_event'], errors='coerce') < 0
        cnt = int(neg.sum())
        if cnt:
            st.warning(f"{cnt} leads have negative Days Until Event.")

# Lead Scoring Tab
with tab2:
    # --- Lead Scoring Tab ---
    st.markdown("## Lead Scoring")
    st.markdown("This tab contains tools for developing and applying lead scoring models based on conversion patterns.")
    
    # Reload data without filtering for model training
    model_df = filtered_df.copy()
    
    lead_scoring_tabs = st.tabs(["Score Calculator", "Model Development", "Batch Scoring"])
    
    # Score Calculator subtab
    with lead_scoring_tabs[0]:
        st.subheader("Lead Score Calculator")
        st.markdown("Enter lead details to calculate a conversion probability score.")
        
        # Generate scorecard if it doesn't exist
        if 'scorecard' not in st.session_state or 'thresholds' not in st.session_state:
            with st.spinner("Generating lead scoring model..."):
                try:
                    weights_df, thresholds, metrics = generate_lead_scorecard()
                    st.session_state.scorecard = weights_df
                    st.session_state.thresholds = thresholds
                    st.session_state.metrics = metrics
                except Exception as e:
                    st.error(f"Error generating lead scorecard: {str(e)}")
                    weights_df = pd.DataFrame(columns=['feature', 'weight'])
                    thresholds = {'hot': 5, 'warm': 3, 'cool': 1}
                    metrics = {}
        else:
            weights_df = st.session_state.scorecard
            thresholds = st.session_state.thresholds
            metrics = st.session_state.metrics
        
        # Input form for lead details
        with st.form("lead_score_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                guest_count = st.number_input("Number of Guests", min_value=0, max_value=1000, value=100)
                days_until = st.number_input("Days Until Event", min_value=0, max_value=365, value=30)
                
                booking_types = model_df['booking_type'].dropna().unique().tolist() if 'booking_type' in model_df.columns else []
                booking_type = st.selectbox("Booking Type", options=[''] + booking_types)
                
                states = model_df['state'].dropna().unique().tolist() if 'state' in model_df.columns else []
                state = st.selectbox("State", options=[''] + states)
            
            with col2:
                bartenders = st.number_input("Bartenders Needed", min_value=0, max_value=50, value=2)
                price = st.number_input("Deal Value ($)", min_value=0, max_value=100000, value=1000)
                
                referral_sources = model_df['referral_source'].dropna().unique().tolist() if 'referral_source' in model_df.columns else []
                referral_source = st.selectbox("Referral Source", options=[''] + referral_sources)
                
                marketing_sources = model_df['marketing_source'].dropna().unique().tolist() if 'marketing_source' in model_df.columns else []
                marketing_source = st.selectbox("Marketing Source", options=[''] + marketing_sources)
            
            submit_button = st.form_submit_button("Calculate Score")
        
        if submit_button:
            lead_data = {
                'number_of_guests': guest_count,
                'days_until_event': days_until,
                'booking_type': booking_type,
                'state': state,
                'bartenders_needed': bartenders,
                'actual_deal_value': price,
                'referral_source': referral_source,
                'marketing_source': marketing_source,
            }
            
            try:
                score, category = score_lead(lead_data, weights_df)
                
                # Display the score and category
                st.markdown("### Lead Score Results")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Score", f"{score:.1f} points")
                
                with col2:
                    color = "#e74c3c" if category == "Hot" else "#f39c12" if category == "Warm" else "#3498db"
                    st.markdown(f"""
                    <div style="background-color: {color}; padding: 10px; border-radius: 5px; text-align: center;">
                        <h3 style="color: white; margin: 0;">{category}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    win_prob = metrics.get('y_pred_proba', [0.5])[0] if metrics else 0.5
                    st.metric("Win Probability", f"{win_prob:.1%}")
                
                # Show top factors
                st.markdown("### Top Factors")
                
                # Get top positive and negative factors
                if not weights_df.empty:
                    factor_cols = ['feature', 'weight', 'impact']
                    factors_df = pd.DataFrame(columns=factor_cols)
                    
                    for feature in weights_df['feature']:
                        if feature in lead_data and pd.notna(lead_data[feature]):
                            weight = weights_df.loc[weights_df['feature'] == feature, 'weight'].values[0]
                            impact = weight
                            new_row = pd.DataFrame([[feature, weight, impact]], columns=factor_cols)
                            factors_df = pd.concat([factors_df, new_row], ignore_index=True)
                    
                    # Sort by impact
                    factors_df = factors_df.sort_values('impact', ascending=False)
                    
                    # Display top positive and negative factors
                    pos_factors = factors_df[factors_df['impact'] > 0].head(3)
                    neg_factors = factors_df[factors_df['impact'] < 0].head(3)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Positive Factors")
                        if not pos_factors.empty:
                            for i, row in pos_factors.iterrows():
                                st.markdown(f"- **{row['feature']}**: +{row['impact']:.1f} points")
                        else:
                            st.markdown("No significant positive factors found.")
                    
                    with col2:
                        st.markdown("#### Negative Factors")
                        if not neg_factors.empty:
                            for i, row in neg_factors.iterrows():
                                st.markdown(f"- **{row['feature']}**: {row['impact']:.1f} points")
                        else:
                            st.markdown("No significant negative factors found.")
            except Exception as e:
                st.error(f"Error calculating lead score: {str(e)}")
    
    # Model Development subtab
    with lead_scoring_tabs[1]:
        st.subheader("Lead Scoring Model Development")
        st.markdown("Explore the predictive model and its performance on historical data.")
        
        # Generate model metrics if needed
        if 'metrics' not in st.session_state:
            with st.spinner("Generating model metrics..."):
                try:
                    _, _, metrics = generate_lead_scorecard()
                    st.session_state.metrics = metrics
                except Exception as e:
                    st.error(f"Error generating model metrics: {str(e)}")
                    metrics = {}
        else:
            metrics = st.session_state.metrics
        
        # Display model metrics and performance
        if metrics:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Model Performance")
                
                # Check if metrics contains the keys we need
                if 'roc_auc' in metrics:
                    st.metric("ROC AUC", f"{metrics['roc_auc']:.3f}")
                
                if 'fpr' in metrics and 'tpr' in metrics:
                    # Plot ROC curve
                    fig, ax = plt.subplots()
                    ax.plot(metrics['fpr'], metrics['tpr'], lw=2)
                    ax.plot([0, 1], [0, 1], 'k--', lw=1)
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_title('ROC Curve')
                    st.pyplot(fig)
            
            with col2:
                st.markdown("### Score Distribution")
                
                if 'won_scores' in metrics and 'lost_scores' in metrics:
                    # Plot score distribution
                    fig, ax = plt.subplots()
                    sns.histplot(metrics['won_scores'], ax=ax, color='green', alpha=0.5, label='Won')
                    sns.histplot(metrics['lost_scores'], ax=ax, color='red', alpha=0.5, label='Lost')
                    ax.set_xlabel('Score')
                    ax.set_ylabel('Count')
                    ax.set_title('Score Distribution by Outcome')
                    ax.legend()
                    st.pyplot(fig)
            
            # Feature importance
            st.markdown("### Feature Importance")
            
            if 'scorecard' in st.session_state:
                weights_df = st.session_state.scorecard
                
                if not weights_df.empty:
                    # Sort features by absolute weight
                    weights_df = weights_df.sort_values('weight', key=abs, ascending=False)
                    
                    # Plot feature importance
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.barh(weights_df['feature'], weights_df['weight'])
                    
                    # Color positive and negative weights differently
                    for i, bar in enumerate(bars):
                        bar.set_color('green' if bar.get_width() >= 0 else 'red')
                    
                    ax.set_xlabel('Weight')
                    ax.set_title('Feature Importance')
                    st.pyplot(fig)
            
            # Confusion matrix
            st.markdown("### Confusion Matrix")
            
            if 'confusion_matrix' in metrics:
                cm = metrics['confusion_matrix']
                
                # Plot confusion matrix
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.set_title('Confusion Matrix')
                ax.set_xticklabels(['Lost', 'Won'])
                ax.set_yticklabels(['Lost', 'Won'])
                st.pyplot(fig)
        else:
            st.warning("Model metrics not available. Please run the lead scoring model generation first.")
    
    # Batch Scoring subtab
    with lead_scoring_tabs[2]:
        st.subheader("Batch Lead Scoring")
        st.markdown("Score multiple leads at once to identify high-potential opportunities.")
        
        # Load unfiltered data for batch scoring
        batch_df = model_df.copy()
        
        # Check if scorecard exists
        if 'scorecard' not in st.session_state:
            st.warning("Lead scoring model not available. Please visit the Model Development tab to generate a model.")
        else:
            weights_df = st.session_state.scorecard
            thresholds = st.session_state.thresholds
            
            # Apply scoring to all leads
            try:
                # Select only relevant columns for scoring
                score_cols = weights_df['feature'].unique().tolist()
                available_cols = [col for col in score_cols if col in batch_df.columns]
                
                if available_cols:
                    # Create a copy with only needed columns
                    scoring_df = batch_df[available_cols].copy()
                    
                    # Add any missing columns with nulls
                    for col in score_cols:
                        if col not in scoring_df.columns:
                            scoring_df[col] = np.nan
                    
                    # Apply scoring function to each row
                    scores = []
                    categories = []
                    
                    for idx, row in scoring_df.iterrows():
                        try:
                            lead_data = row.to_dict()
                            score, category = score_lead(lead_data, weights_df)
                            scores.append(score)
                            categories.append(category)
                        except:
                            scores.append(np.nan)
                            categories.append('Unknown')
                    
                    # Add scores to dataframe
                    batch_df['lead_score'] = scores
                    batch_df['lead_category'] = categories
                    
                    # Display results
                    st.markdown("### Batch Scoring Results")
                    
                    # Summary by category
                    category_counts = batch_df['lead_category'].value_counts()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Plot category distribution
                        fig, ax = plt.subplots()
                        category_counts.plot(kind='bar', ax=ax)
                        ax.set_xlabel('Category')
                        ax.set_ylabel('Count')
                        ax.set_title('Leads by Score Category')
                        st.pyplot(fig)
                    
                    with col2:
                        # Display count and percentage
                        category_pct = category_counts / category_counts.sum() * 100
                        summary_df = pd.DataFrame({
                            'Count': category_counts,
                            'Percentage': category_pct.round(1).astype(str) + '%'
                        })
                        st.dataframe(summary_df)
                    
                    # Display top scored leads
                    st.markdown("### Top Scored Leads")
                    
                    # Select columns to display
                    display_cols = []
                    
                    if 'name' in batch_df.columns:
                        display_cols.append('name')
                    elif 'first_name' in batch_df.columns and 'last_name' in batch_df.columns:
                        batch_df['full_name'] = batch_df['first_name'] + ' ' + batch_df['last_name']
                        display_cols.append('full_name')
                    
                    essential_cols = ['lead_score', 'lead_category']
                    info_cols = ['booking_type', 'number_of_guests', 'days_until_event', 'state']
                    
                    # Add available columns to display list
                    for col in info_cols:
                        if col in batch_df.columns:
                            display_cols.append(col)
                    
                    display_cols.extend(essential_cols)
                    
                    # Get available display columns
                    valid_display_cols = [col for col in display_cols if col in batch_df.columns]
                    
                    # Display top 20 leads by score
                    top_leads = batch_df.sort_values('lead_score', ascending=False).head(20)
                    if valid_display_cols:
                        st.dataframe(top_leads[valid_display_cols])
                    else:
                        st.dataframe(top_leads)
                else:
                    st.warning("No scoring features found in the data. Please ensure your data includes the necessary fields.")
            except Exception as e:
                st.error(f"Error performing batch scoring: {str(e)}")

# Contact Matching Tab
with tab3:
    # --- Contact Matching Tab ---
    st.markdown("## Contact Matching Analysis")
    st.markdown("Analyze phone number matching and contact patterns between leads and conversions.")
    
    # Process contact matching if needed
    if 'phone_matching_done' not in st.session_state:
        with st.spinner("Processing phone matching..."):
            try:
                matched_count, total_leads, total_operations = process_phone_matching()
                st.session_state.phone_matching_done = True
                st.session_state.phone_matching_stats = {
                    'matched_count': matched_count,
                    'total_leads': total_leads,
                    'total_operations': total_operations
                }
            except Exception as e:
                st.error(f"Error processing phone matching: {str(e)}")
                st.session_state.phone_matching_done = False
    
    # Display phone matching results
    if st.session_state.get('phone_matching_done', False):
        stats = st.session_state.phone_matching_stats
        
        # Display matching summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Leads", stats['total_leads'])
        
        with col2:
            st.metric("Total Operations", stats['total_operations'])
        
        with col3:
            match_rate = stats['matched_count'] / stats['total_leads'] if stats['total_leads'] > 0 else 0
            st.metric("Match Rate", f"{match_rate:.1%}")
    
    # Phone matching analysis
    try:
        phone_match_data = analyze_phone_matches(filtered_df)
        
        if phone_match_data:
            match_conv, match_counts = phone_match_data
            
            st.subheader("Phone Match Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Conversion Rate by Phone Match")
                st.dataframe(match_conv)
                
                # Plot conversion rate by match
                fig, ax = plt.subplots()
                match_conv['conversion_rate'].plot(kind='bar', ax=ax)
                ax.set_ylabel('Conversion Rate')
                ax.set_title('Conversion Rate by Phone Match Status')
                st.pyplot(fig)
            
            with col2:
                st.markdown("### Lead Count by Phone Match")
                st.dataframe(match_counts)
                
                # Plot counts by match
                fig, ax = plt.subplots()
                match_counts['count'].plot(kind='bar', ax=ax)
                ax.set_ylabel('Count')
                ax.set_title('Lead Count by Phone Match Status')
                st.pyplot(fig)
    except Exception as e:
        st.error(f"Error analyzing phone matches: {str(e)}")
    
    # Time to conversion analysis
    st.subheader("Time to Conversion Analysis")
    
    try:
        time_analysis = analyze_time_to_conversion(filtered_df)
        
        if time_analysis:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Average Days to Convert", f"{time_analysis['average_days']:.1f}")
            
            with col2:
                st.metric("Median Days to Convert", f"{time_analysis['median_days']:.1f}")
            
            with col3:
                st.metric("Min-Max Range", f"{time_analysis['min_days']}-{time_analysis['max_days']} days")
            
            # Histogram of conversion times
            if 'histogram_data' in time_analysis:
                fig, ax = plt.subplots()
                sns.histplot(time_analysis['histogram_data'], ax=ax)
                ax.set_xlabel('Days to Conversion')
                ax.set_ylabel('Count')
                ax.set_title('Distribution of Time to Conversion')
                st.pyplot(fig)
            
            # Conversion time by outcome
            if 'by_outcome' in time_analysis:
                st.markdown("### Average Time by Outcome")
                st.dataframe(time_analysis['by_outcome'])
            
            # Conversion time by booking type
            if 'by_booking_type' in time_analysis:
                st.markdown("### Average Time by Booking Type")
                
                # Plot by booking type
                fig, ax = plt.subplots()
                by_type = time_analysis['by_booking_type']
                by_type.plot(kind='bar', ax=ax)
                ax.set_xlabel('Booking Type')
                ax.set_ylabel('Average Days to Conversion')
                ax.set_title('Average Conversion Time by Booking Type')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
    except Exception as e:
        st.error(f"Error analyzing time to conversion: {str(e)}")

# Insights Tab
with tab4:
    # --- Insights Tab ---
    st.markdown("## AI-Powered Insights")
    st.markdown("Discover key insights and recommendations from your conversion data.")
    
    # Check if insights are already generated
    if 'insights' not in st.session_state:
        st.session_state.insights = None
    
    # Add a button to generate insights
    if st.button("Generate New Insights"):
        with st.spinner("Generating AI insights..."):
            try:
                # Get insights from Mistral API
                insights = generate_insights(filtered_df)
                st.session_state.insights = insights
            except Exception as e:
                st.error(f"Error generating insights: {str(e)}")
                st.session_state.insights = None
    
    # Display insights
    if st.session_state.insights:
        for i, insight in enumerate(st.session_state.insights):
            with st.expander(f"Insight {i+1}: {insight['title']}", expanded=i==0):
                st.markdown(insight['content'])
                
                if 'recommendations' in insight:
                    st.markdown("### Recommendations")
                    for rec in insight['recommendations']:
                        st.markdown(f"- {rec}")
                
                if 'chart' in insight:
                    try:
                        # Display chart if available
                        chart_type = insight.get('chart_type', 'bar')
                        
                        if chart_type == 'bar':
                            fig, ax = plt.subplots()
                            ax.bar(insight['chart']['x'], insight['chart']['y'])
                            ax.set_xlabel(insight['chart'].get('x_label', ''))
                            ax.set_ylabel(insight['chart'].get('y_label', ''))
                            ax.set_title(insight['chart'].get('title', ''))
                            st.pyplot(fig)
                        elif chart_type == 'line':
                            fig, ax = plt.subplots()
                            ax.plot(insight['chart']['x'], insight['chart']['y'])
                            ax.set_xlabel(insight['chart'].get('x_label', ''))
                            ax.set_ylabel(insight['chart'].get('y_label', ''))
                            ax.set_title(insight['chart'].get('title', ''))
                            st.pyplot(fig)
                        elif chart_type == 'pie':
                            fig, ax = plt.subplots()
                            ax.pie(insight['chart']['y'], labels=insight['chart']['x'], autopct='%1.1f%%')
                            ax.set_title(insight['chart'].get('title', ''))
                            st.pyplot(fig)
                    except Exception as e:
                        st.warning(f"Could not display chart: {str(e)}")
    else:
        # Show a placeholder
        st.info("Click 'Generate New Insights' to get AI-powered recommendations based on your conversion data.")
        
        # Sample insights for demonstration
        with st.expander("Sample Insight: High-Value Booking Types", expanded=True):
            st.markdown("""
            ### Analysis of High-Value Booking Types
            
            Wedding and Corporate bookings show significantly higher conversion rates compared to other booking types. 
            Wedding bookings convert at 45% while Corporate events convert at 37%, compared to an average of 28% across all types.
            
            This pattern suggests that focusing sales efforts on these two booking types could yield better results.
            
            #### Recommendations:
            - Prioritize follow-up for Wedding and Corporate leads
            - Consider separate sales processes for these high-value categories
            - Analyze what makes these bookings more likely to convert and apply those insights to other booking types
            """)
        
        with st.expander("Sample Insight: Optimal Lead Response Time", expanded=False):
            st.markdown("""
            ### Impact of Lead Response Time
            
            Leads that receive a response within 24 hours have a 65% higher conversion rate than those that wait longer.
            
            The data shows that leads contacted within 8 hours have a 42% conversion rate, while those contacted after 48+ hours drop to just 19%.
            
            #### Recommendations:
            - Implement a rapid response system for all new leads
            - Prioritize leads by booking type and potential value
            - Consider automated initial responses for after-hours inquiries
            """)

# Admin Tab
with tab5:
    # --- Admin Tab ---
    st.markdown("## Dashboard Administration")
    st.markdown("Import data, manage database, and configure dashboard settings.")
    
    admin_tabs = st.tabs(["Data Import", "Database Admin", "Settings"])
    
    # Data Import subtab
    with admin_tabs[0]:
        st.subheader("Import Data")
        st.markdown("Upload Streak export files to populate the dashboard with your data.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Leads Import")
            leads_file = st.file_uploader("Upload Leads CSV", type=['csv'])
            
            if leads_file is not None:
                try:
                    # Save uploaded file temporarily
                    temp_path = "temp_leads.csv"
                    with open(temp_path, "wb") as f:
                        f.write(leads_file.getbuffer())
                    
                    # Import the data
                    imported_count = import_leads_data(temp_path)
                    st.success(f"Successfully imported {imported_count} lead records!")
                except Exception as e:
                    st.error(f"Error importing leads data: {str(e)}")
        
        with col2:
            st.markdown("### Operations Import")
            operations_file = st.file_uploader("Upload Operations CSV", type=['csv'])
            
            if operations_file is not None:
                try:
                    # Save uploaded file temporarily
                    temp_path = "temp_operations.csv"
                    with open(temp_path, "wb") as f:
                        f.write(operations_file.getbuffer())
                    
                    # Import the data
                    imported_count = import_operations_data(temp_path)
                    st.success(f"Successfully imported {imported_count} operation records!")
                except Exception as e:
                    st.error(f"Error importing operations data: {str(e)}")
        
        # Reload button
        if st.button("Reload Dashboard Data"):
            st.experimental_rerun()
    
    # Database Admin subtab
    with admin_tabs[1]:
        st.subheader("Database Administration")
        st.markdown("View and manage the underlying database.")
        
        # Load current data
        try:
            leads_df = get_lead_data()
            operations_df = get_operation_data()
            
            # Display summary
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Leads", len(leads_df) if leads_df is not None else 0)
            
            with col2:
                st.metric("Total Operations", len(operations_df) if operations_df is not None else 0)
            
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
    
    # Settings subtab
    with admin_tabs[2]:
        st.subheader("Dashboard Settings")
        st.markdown("Configure dashboard appearance and behavior.")
        
        # Theme settings
        st.markdown("### Theme Settings")
        primary_color = st.color_picker("Primary Color", "#1E88E5")
        
        # Default view settings
        st.markdown("### Default View")
        default_tab = st.selectbox("Default Tab", options=[
            "Conversion Analysis", 
            "Lead Scoring", 
            "Contact Matching", 
            "Insights", 
            "Admin"
        ])
        
        # Save settings button
        if st.button("Save Settings"):
            # Normally we'd save these to a settings file or database
            st.success("Settings saved!")
            
            # For now, just store in session state
            st.session_state.settings = {
                'primary_color': primary_color,
                'default_tab': default_tab
            }

# Advanced analytics tab (if time permits)
# Additional CSS for theming
custom_css = f"""
<style>
    .main-header {{
        color: {st.session_state.settings['primary_color'] if 'settings' in st.session_state else "#1E88E5"};
    }}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)