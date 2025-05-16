"""
explanations_tab.py - Explanations Tab Module

This module provides the implementation for the Explanations tab
in the Sales Conversion Analytics Dashboard.
"""

import streamlit as st

def render_explanations_tab():
    """
    Render the Explanations tab with static documentation
    about how to use the dashboard
    """
    st.subheader("Dashboard Guide")
    st.markdown("This section explains how to use the Sales Conversion Analytics Dashboard and interpret the data shown.")
    
    # Tabs explanation
    with st.expander("Dashboard Tabs Overview", expanded=True):
        st.markdown("""
        The dashboard is organized into several tabs, each with a specific focus:
        
        1. **Conversion Analysis** - Visualize conversion rates across different dimensions like booking types, referral sources, and time periods.
        
        2. **Feature Correlation** - Analyze which factors most strongly correlate with successful conversions.
        
        3. **Lead Scoring** - Build and apply a predictive model to score new leads based on their likelihood to convert.
        
        4. **Raw Data** - View the complete dataset and perform quality assessments.
        
        5. **Key Findings** - See a synthesis of the most important insights from your data.
        
        6. **Explanations** - This tab, providing guidance on using the dashboard.
        
        7. **Advanced Analytics** - Access deeper analytical tools for segment analysis and detailed metrics.
        
        8. **AI Insights** - Generate natural language insights and recommendations using Mistral AI.
        """)
    
    # Conversion Analysis explanation
    with st.expander("Conversion Analysis Tab", expanded=False):
        st.markdown("""
        ### Understanding the Conversion Analysis Tab
        
        This tab helps you visualize your conversion rates across different dimensions.
        
        **Key Performance Indicators (KPIs):**
        - **Total Leads** - The total number of leads in your dataset
        - **Conversion Rate** - The percentage of leads that converted to won deals
        - **Average Deal Value** - The average value of won deals
        - **Total Revenue** - The sum of all won deal values
        
        **Charts and Their Interpretation:**
        
        1. **Booking Types Chart** - Shows conversion rates by event or booking type. Higher bars indicate better-performing categories.
        
        2. **Referral Sources Chart** - Displays conversion rates by lead source. This helps identify which channels bring your highest-quality leads.
        
        3. **Timing Factors** - Analyzes how lead timing impacts conversion:
           - Days until event - Does urgency affect closing rate?
           - Inquiry day of week - Are certain days better for conversions?
           - Event month - Is there seasonality in your conversions?
        
        4. **Size Factors** - Examines how event size relates to conversion:
           - Guest count ranges - Do larger or smaller events convert better?
           - Price per guest - Is there a sweet spot for pricing?
        
        5. **Geographic Insights** - Shows conversion rates by location.
        """)
    
    # Feature Correlation explanation
    with st.expander("Feature Correlation Tab", expanded=False):
        st.markdown("""
        ### Understanding the Feature Correlation Tab
        
        This tab helps you understand which factors most strongly influence your conversion outcomes.
        
        **Correlation Coefficient:**
        - Values range from -1 to 1
        - Positive values (closer to 1) indicate factors that increase conversion likelihood
        - Negative values (closer to -1) indicate factors that decrease conversion likelihood
        - Values near zero suggest minimal impact
        
        **Key Visualizations:**
        
        1. **Correlation Heatmap** - Shows relationships between all features and outcome
        
        2. **Top Correlations Chart** - Highlights the factors with strongest positive and negative correlations to conversion
        
        3. **Scatter Plots** - For numerical factors, these show the relationship between the factor and conversion outcomes
        
        **How to Use This Information:**
        - Focus sales efforts on leads with strong positive correlation factors
        - Develop strategies to overcome negative correlation factors
        - Use insights to refine your targeting and qualifying process
        """)
    
    # Lead Scoring explanation
    with st.expander("Lead Scoring Tab", expanded=False):
        st.markdown("""
        ### Understanding the Lead Scoring Tab
        
        This tab uses machine learning to predict which leads are most likely to convert based on historical patterns.
        
        **Key Components:**
        
        1. **Model Training** - The system builds a predictive model using your historical data
        
        2. **Feature Importance** - Shows which factors the model considers most important in predicting conversion
        
        3. **Model Performance Metrics**:
           - **ROC AUC** - Measures how well the model distinguishes between won and lost deals (higher is better)
           - **Precision** - Of all leads predicted to convert, what percentage actually converted
           - **Recall** - Of all leads that actually converted, what percentage did the model correctly identify
           - **F1 Score** - A balance between precision and recall
        
        4. **Lead Score Distribution** - Shows how your leads are distributed across different score ranges
        
        5. **Score Thresholds** - Automatically determined cut-offs for categorizing leads as Hot, Warm, Cool, or Cold
        
        **How to Use Lead Scores:**
        - Prioritize sales resources on higher-scoring leads
        - Develop different follow-up strategies for each score category
        - Monitor which factors most strongly influence scores and try to improve those areas
        """)
    
    # Raw Data explanation
    with st.expander("Raw Data Tab", expanded=False):
        st.markdown("""
        ### Understanding the Raw Data Tab
        
        This tab shows your complete dataset and provides tools for assessing data quality.
        
        **Key Features:**
        
        1. **Data Overview** - View the complete, unfiltered dataset
        
        2. **Download Option** - Export your data as a CSV file
        
        3. **Data Information** - See column descriptions, types, and summary statistics
        
        4. **Data Quality Assessment**:
           - Missing Values Analysis - Identify and understand data gaps
           - Duplicate Rows Check - Find and analyze duplicate entries
           - Outlier Detection - Identify unusual values that may affect analysis
        
        **How to Use This Tab:**
        - Review your data to understand its structure and content
        - Check for data quality issues that might affect your analysis
        - Export data for use in other tools if needed
        """)
    
    # Key Findings explanation
    with st.expander("Key Findings Tab", expanded=False):
        st.markdown("""
        ### Understanding the Key Findings Tab
        
        This tab synthesizes the most important insights from your data and model analysis.
        
        **Key Components:**
        
        1. **Automated Insights** - The system identifies patterns and notable metrics from your data
        
        2. **Feature Importance Analysis** - Shows which factors most strongly drive conversion outcomes
        
        3. **Dynamic Updates** - Findings update automatically when you filter data or regenerate the model
        
        **How to Use This Tab:**
        - Use these insights to quickly understand what's working and what's not
        - Share these findings with your team to inform strategy
        - Look for actionable insights that can directly improve your conversion process
        """)
    
    # Advanced Analytics explanation
    with st.expander("Advanced Analytics Tab", expanded=False):
        st.markdown("""
        ### Understanding the Advanced Analytics Tab
        
        This tab provides more sophisticated analytical tools for deeper insights.
        
        **Key Features:**
        
        1. **Segmentation Analysis** - Group leads into meaningful segments and analyze performance by segment
        
        2. **Time Series Analysis** - Track conversion trends over time
        
        3. **Multi-factor Analysis** - See how combinations of factors affect conversion rates
        
        **How to Use This Tab:**
        - Explore advanced insights after understanding the basics from earlier tabs
        - Look for specific patterns or hypotheses to test
        - Use segmentation to develop targeted strategies for different lead types
        """)
    
    # AI Insights explanation
    with st.expander("AI Insights Tab", expanded=False):
        st.markdown("""
        ### Understanding the AI Insights Tab
        
        This tab uses Mistral AI to generate natural language insights and recommendations.
        
        **Analysis Types:**
        
        1. **Sales Opportunity Analysis** - Identifies untapped opportunities and suggests strategies
        
        2. **Booking Type Recommendations** - Provides tailored advice for different event types
        
        3. **Customer Segment Insights** - Analyzes patterns across different customer groups
        
        **How to Use This Tab:**
        - Select the analysis type you're interested in
        - Click "Generate AI Insights" to create a new analysis
        - Review the insights and consider how to implement the recommendations
        - Note: This feature requires a Mistral API key to be configured
        """)
    
    # General usage tips
    with st.expander("General Usage Tips", expanded=False):
        st.markdown("""
        ### Dashboard Usage Tips
        
        **Data Loading:**
        - You can load data either by uploading CSV files or connecting to the database
        - For CSV uploads, you need both the Leads and Operations export files from Streak
        
        **Filters:**
        - By default, the dashboard shows all available data
        - You can apply date range filters, status filters, and region filters if needed
        
        **Interpreting Results:**
        - Look for notable differences in conversion rates across categories
        - Pay attention to sample sizes - metrics based on very few leads may not be reliable
        - Consider implementing strategies based on the most significant findings
        
        **Sharing Insights:**
        - Use the download options to export data and visualizations
        - Reference specific tabs and metrics when discussing findings with your team
        
        **Refreshing Data:**
        - Re-upload new data or reload from the database to see the latest information
        - Models and insights will update automatically with new data
        """)
    
    # Example use cases
    with st.expander("Example Use Cases", expanded=False):
        st.markdown("""
        ### Example Use Cases
        
        **Scenario 1: Prioritizing Leads**
        
        *Question:* How do I decide which leads to focus on first?
        
        *Solution:*
        1. Go to the Lead Scoring tab and generate a scoring model
        2. Look at the Feature Importance chart to understand what drives conversions
        3. Use the score categories (Hot, Warm, Cool, Cold) to prioritize your follow-up
        
        **Scenario 2: Improving Marketing Strategy**
        
        *Question:* Which marketing channels should we invest more in?
        
        *Solution:*
        1. Go to the Conversion Analysis tab
        2. Check the Referral Sources chart to see which channels have highest conversion rates
        3. Use the Advanced Analytics tab to analyze ROI by marketing source
        
        **Scenario 3: Seasonal Planning**
        
        *Question:* How should we adjust our staffing and resources throughout the year?
        
        *Solution:*
        1. Go to the Conversion Analysis tab
        2. Look at the Event Month chart to identify high and low seasons
        3. Use the Key Findings tab to get insights on seasonal patterns
        
        **Scenario 4: Price Optimization**
        
        *Question:* Are we pricing our services optimally?
        
        *Solution:*
        1. Go to the Feature Correlation tab
        2. Look at how price-related variables correlate with conversion
        3. Use the AI Insights tab to get pricing strategy recommendations
        """)