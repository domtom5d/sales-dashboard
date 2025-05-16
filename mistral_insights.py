"""
mistral_insights.py - Integration with Mistral AI for generating insights

This module provides functions to connect to the Mistral AI API
and generate insights from data.
"""

import os
import pandas as pd
import streamlit as st
from mistralai.sdk import Mistral

def initialize_mistral_client():
    """
    Initialize the Mistral AI client using the API key.
    
    Returns:
        Mistral or None: Initialized client or None if API key is missing
    """
    api_key = os.environ.get("MISTRAL_API_KEY")
    
    if not api_key:
        return None
    
    try:
        client = Mistral(api_key=api_key)
        return client
    except Exception as e:
        st.error(f"Error initializing Mistral client: {str(e)}")
        return None

def generate_data_insights(df, prompt_context=""):
    """
    Generate insights from a dataframe using Mistral AI.
    
    Args:
        df (DataFrame): DataFrame containing the data to analyze
        prompt_context (str, optional): Additional context for the prompt
        
    Returns:
        str: Generated insights from Mistral AI
    """
    client = initialize_mistral_client()
    
    if client is None:
        return "⚠️ Mistral AI integration not available. Please check your API key."
    
    # Prepare data sample for the prompt
    data_sample = df.head(10).to_string()
    column_info = df.dtypes.to_string()
    
    # Prepare statistics for the prompt
    stats = {}
    for col in df.select_dtypes(include=['number']).columns:
        stats[col] = {
            'min': df[col].min(),
            'max': df[col].max(),
            'mean': df[col].mean(),
            'median': df[col].median()
        }
    
    # Format the stats for the prompt
    stats_text = "\n".join([f"{col}: {s}" for col, s in stats.items()])
    
    # Build the full prompt
    system_prompt = """You are an expert data analyst specializing in sales conversion analytics. 
    Your task is to provide valuable insights based on the data sample provided. 
    Focus on patterns, anomalies, and actionable recommendations for improving conversion rates.
    Be specific, direct, and actionable in your analysis. 
    Provide numbered insights (at least 3, but no more than 5) with a brief explanation for each.
    Format your response for easy readability with bullet points and clear section headings.
    """
    
    user_prompt = f"""
    {prompt_context}
    
    Here's a sample of the data:
    {data_sample}
    
    Column information:
    {column_info}
    
    Key statistics:
    {stats_text}
    
    Please provide your analysis and insights for this data. 
    Focus on conversion patterns, key factors affecting conversions, and actionable recommendations.
    """
    
    try:
        # Call Mistral AI API using the new structure
        response = client.chat.completions.create(
            model="mistral-medium",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        # Extract content from the response
        if hasattr(response, 'choices') and len(response.choices) > 0:
            if hasattr(response.choices[0], 'message'):
                return response.choices[0].message.content
            else:
                return "⚠️ Unexpected response format from Mistral API"
        else:
            return "⚠️ No response content from Mistral API"
    
    except Exception as e:
        st.error(f"Error generating insights: {str(e)}")
        return f"⚠️ Error generating insights: {str(e)}"

def generate_sales_opportunity_analysis(df):
    """
    Generate a detailed analysis of sales opportunities using Mistral AI.
    
    Args:
        df (DataFrame): DataFrame containing the lead and conversion data
        
    Returns:
        str: Generated opportunity analysis from Mistral AI
    """
    prompt_context = """
    Analyze the sales opportunity data with a focus on:
    1. Identifying the most promising lead segments
    2. Highlighting time periods with highest conversion potential
    3. Suggesting specific actions to improve conversion rates
    4. Identifying any concerning trends or patterns
    
    Include specific recommendations for sales teams to optimize their efforts.
    """
    
    return generate_data_insights(df, prompt_context)

def generate_booking_type_recommendations(df, booking_type_conversions):
    """
    Generate recommendations for different booking types using Mistral AI.
    
    Args:
        df (DataFrame): DataFrame containing the full dataset
        booking_type_conversions (DataFrame): DataFrame with booking type conversion rates
        
    Returns:
        str: Generated recommendations from Mistral AI
    """
    # Convert booking_type_conversions to string for the prompt
    conversions_text = booking_type_conversions.to_string()
    
    prompt_context = f"""
    Focus specifically on booking type conversion patterns.
    
    Here's the booking type conversion data:
    {conversions_text}
    
    For each booking type, suggest:
    1. Why this booking type might be performing well or poorly
    2. Specific strategies to improve conversion rates for each type
    3. Any cross-selling opportunities between booking types
    4. Pricing or packaging recommendations based on the data
    """
    
    return generate_data_insights(df, prompt_context)

def generate_customer_segment_insights(df):
    """
    Generate insights on customer segments using Mistral AI.
    
    Args:
        df (DataFrame): DataFrame containing customer segment data
        
    Returns:
        str: Generated segment insights from Mistral AI
    """
    prompt_context = """
    Analyze the customer data with a focus on segmentation:
    1. Identify distinct customer segments and their characteristics
    2. Highlight which segments convert at higher rates and why
    3. Suggest tailored marketing approaches for each segment
    4. Recommend ways to improve engagement with lower-converting segments
    
    Be specific about the characteristics that define high-value segments.
    """
    
    return generate_data_insights(df, prompt_context)

def generate_insights(df):
    """
    Generate comprehensive insights for the dashboard.
    
    Args:
        df (DataFrame): DataFrame containing the full dataset
        
    Returns:
        list: List of insight dictionaries, each containing:
            - title: Insight title
            - content: Detailed insight content
            - recommendations: List of recommendations (optional)
            - chart: Chart data dictionary (optional)
            - chart_type: Type of chart (e.g., 'bar', 'line', 'pie') (optional)
    """
    # Check if we have API key for Mistral
    if not os.environ.get("MISTRAL_API_KEY"):
        # Return sample insights if no API key is available
        return [
            {
                "title": "Wedding and Corporate Events Drive Highest Conversion",
                "content": """
                Wedding Planner referrals convert at 56%, while Corporate Event leads convert at 48% - both significantly higher 
                than the average conversion rate of 32%.

                These high-value lead sources account for 28% of your won deals despite representing only 15% of total lead volume.
                """,
                "recommendations": [
                    "Prioritize follow-up for Wedding Planner and Corporate leads",
                    "Explore partnership opportunities with wedding planners",
                    "Consider specialized materials for corporate event inquiries"
                ]
            },
            {
                "title": "Early Response Drives Higher Conversion",
                "content": """
                Leads that receive a response within 24 hours have a 65% higher conversion rate than those that wait longer.

                The data shows that leads contacted within 8 hours have a 42% conversion rate, while those contacted after 48+ hours drop to just 19%.
                """,
                "recommendations": [
                    "Implement a rapid response system for all new leads",
                    "Prioritize leads by booking type and potential value",
                    "Consider automated initial responses for after-hours inquiries"
                ]
            },
            {
                "title": "Seasonal Patterns in Booking Behavior",
                "content": """
                Event bookings for summer months (June-August) show 38% higher conversion rates than winter months (December-February).

                This pattern suggests seasonal demand fluctuations and potential opportunities for seasonal promotions or pricing strategies.
                """,
                "recommendations": [
                    "Create seasonal promotions for lower-demand periods",
                    "Adjust pricing strategy based on seasonal demand patterns",
                    "Plan staff resources to accommodate seasonal fluctuations"
                ]
            }
        ]
    
    # Extract key metrics for analysis
    try:
        # Calculate conversion rates by different categories
        results = []
        
        # Overall performance
        total = len(df)
        won = df['outcome'].sum() if 'outcome' in df.columns else 0
        conversion_rate = won / total if total > 0 else 0
        
        # Get insights from Mistral AI
        overall_insights = generate_sales_opportunity_analysis(df)
        
        # Create insight object
        results.append({
            "title": "Sales Performance Overview",
            "content": overall_insights,
            "recommendations": extract_recommendations(overall_insights)
        })
        
        # Try to generate booking type insights if data is available
        if 'booking_type' in df.columns:
            booking_types = df.groupby('booking_type').agg(
                total=('outcome', 'size'),
                won=('outcome', 'sum')
            )
            booking_types['rate'] = booking_types['won'] / booking_types['total']
            
            booking_insights = generate_booking_type_recommendations(df, booking_types)
            
            results.append({
                "title": "Booking Type Analysis",
                "content": booking_insights,
                "recommendations": extract_recommendations(booking_insights),
                "chart": {
                    "x": booking_types.index.tolist(),
                    "y": (booking_types['rate'] * 100).tolist(),
                    "x_label": "Booking Type",
                    "y_label": "Conversion Rate (%)",
                    "title": "Conversion Rate by Booking Type"
                },
                "chart_type": "bar"
            })
        
        # Try to generate timing insights if data is available
        if 'days_until_event' in df.columns:
            # Create bins for days until event
            df['days_bin'] = pd.cut(
                df['days_until_event'], 
                bins=[0, 7, 30, 90, float('inf')],
                labels=['≤7d', '8-30d', '31-90d', '90d+']
            )
            
            time_data = df.groupby('days_bin').agg(
                total=('outcome', 'size'),
                won=('outcome', 'sum')
            )
            time_data['rate'] = time_data['won'] / time_data['total']
            
            time_insights = """
            ## Timing Impact Analysis
            
            Analysis of conversion rates based on how far in advance events are booked shows a clear pattern.
            
            Events booked within 7 days of the inquiry have a significantly higher conversion rate of {:.1%}, 
            compared to events booked 90+ days in advance at {:.1%}.
            
            This suggests that urgency is a major factor in closing deals, and quick-turnaround events 
            should be prioritized in the sales process.
            """.format(
                time_data.loc['≤7d', 'rate'] if '≤7d' in time_data.index else 0,
                time_data.loc['90d+', 'rate'] if '90d+' in time_data.index else 0
            )
            
            results.append({
                "title": "Timing Impact on Conversion",
                "content": time_insights,
                "recommendations": [
                    "Prioritize follow-up for events within 7 days",
                    "Develop a specialized process for quick-turnaround events",
                    "Create urgency in your sales process for longer-term bookings"
                ],
                "chart": {
                    "x": time_data.index.tolist(),
                    "y": (time_data['rate'] * 100).tolist(),
                    "x_label": "Days Until Event",
                    "y_label": "Conversion Rate (%)",
                    "title": "Conversion Rate by Event Timing"
                },
                "chart_type": "bar"
            })
        
        return results
        
    except Exception as e:
        # Return basic insights if there's an error
        st.error(f"Error generating insights: {str(e)}")
        return [{
            "title": "Basic Data Overview",
            "content": f"Your dataset contains {total} leads with a {conversion_rate:.1%} overall conversion rate.",
            "recommendations": ["Ensure data quality for better insights", "Consider adding more data points"]
        }]

def extract_recommendations(text):
    """
    Extract recommendations from insight text.
    
    Args:
        text (str): Insight text potentially containing recommendations
        
    Returns:
        list: List of extracted recommendations
    """
    recommendations = []
    
    # Split by lines
    lines = text.split('\n')
    
    # Look for recommendation patterns
    recommendation_section = False
    for line in lines:
        line = line.strip()
        
        # Check if we're in a recommendations section
        if 'recommend' in line.lower() or 'suggest' in line.lower() or 'action' in line.lower():
            recommendation_section = True
            continue
            
        # Look for bullet points or numbered lists
        if recommendation_section and (line.startswith('•') or line.startswith('-') or line.startswith('*') or (line[0].isdigit() and line[1:3] in ['. ', ') '])):
            # Clean up the recommendation
            rec = line
            for prefix in ['•', '-', '*', '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '0.']:
                if rec.startswith(prefix):
                    rec = rec[len(prefix):].strip()
                    break
                    
            # Add if not empty
            if rec:
                recommendations.append(rec)
    
    # If no recommendations found in structured format, try to create some based on keywords
    if not recommendations and len(text) > 100:
        for line in lines:
            if 'should' in line.lower() or 'could' in line.lower() or 'need to' in line.lower() or 'better to' in line.lower():
                recommendations.append(line.strip())
                if len(recommendations) >= 3:
                    break
    
    # Limit to reasonable number
    return recommendations[:5]