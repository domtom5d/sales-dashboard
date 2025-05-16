"""
agent_mistral.py - Agent integration for enhanced AI insights

This module provides integration with MistralAI for generating 
advanced insights and recommendations based on sales data.
"""

import os
import streamlit as st
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

def get_agent_client():
    """
    Initialize and return a Mistral client with authentication
    from environment variables.
    
    Returns:
        MistralClient: Authenticated Mistral client
    """
    # Securely load the agent secret
    agent_secret = os.getenv("AG_SECRET")
    
    if not agent_secret:
        return None
    
    try:
        # Parse the secret format (2db7b0b8:20250516:sales:7dbfb252)
        parts = agent_secret.split(':')
        if len(parts) != 4:
            st.error("Invalid agent secret format")
            return None
            
        api_key = parts[0]  # Using the first part as API key
        
        # Initialize the Mistral client
        client = MistralClient(api_key=api_key)
        return client
    except Exception as e:
        st.error(f"Error initializing agent client: {str(e)}")
        return None

def generate_agent_insight(data_summary, question, model="mistral-medium"):
    """
    Generate AI insights using the Mistral agent.
    
    Args:
        data_summary (str): Summary of the data being analyzed
        question (str): Specific question or insight request
        model (str, optional): Model to use for completion
        
    Returns:
        str: Generated insight or None if error
    """
    client = get_agent_client()
    
    if not client:
        return "Agent integration not available - please add AG_SECRET to your environment variables."
    
    try:
        # Prepare the system prompt with the data context
        system_prompt = f"""You are a sales analytics expert analyzing CRM data.
        Based on the following data summary, provide valuable business insights:
        
        {data_summary}
        
        Respond with clear, action-oriented recommendations. Be specific and practical.
        Focus on patterns, opportunities, and actionable next steps.
        """
        
        # Create message context
        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=question)
        ]
        
        # Get completion from Mistral
        chat_response = client.chat(
            model=model,
            messages=messages,
            max_tokens=750,
            temperature=0.7,
        )
        
        # Extract the response text
        if chat_response and hasattr(chat_response, 'choices') and len(chat_response.choices) > 0:
            return chat_response.choices[0].message.content
        else:
            return "Unable to generate insight with the provided data."
    except Exception as e:
        return f"Error generating insight: {str(e)}"

def analyze_conversion_opportunities(df):
    """
    Use the agent to analyze conversion opportunities in the dataset
    
    Args:
        df (DataFrame): The processed sales data with conversion outcomes
        
    Returns:
        str: AI-generated analysis of conversion opportunities
    """
    # Prepare a summary of the data for the agent
    # Handle potential missing columns
    total_leads = len(df)
    conversion_rate = df['outcome'].mean() if 'outcome' in df.columns else 0
    
    # Safely get top booking types
    top_booking_types = ', '.join(df['booking_type'].value_counts().head(3).index.tolist()) if 'booking_type' in df.columns else 'Unknown'
    
    # Safely get date range
    start_date = 'Unknown'
    end_date = 'Unknown'
    if 'inquiry_date' in df.columns:
        if not df['inquiry_date'].empty and pd.notna(df['inquiry_date']).any():
            min_date = df['inquiry_date'].min()
            max_date = df['inquiry_date'].max()
            if pd.notna(min_date) and pd.notna(max_date):
                start_date = min_date.strftime('%Y-%m-%d') if hasattr(min_date, 'strftime') else str(min_date)
                end_date = max_date.strftime('%Y-%m-%d') if hasattr(max_date, 'strftime') else str(max_date)
    
    data_summary = f"""
    Dataset overview:
    - Total leads: {total_leads}
    - Conversion rate: {conversion_rate:.1%}
    - Top booking types: {top_booking_types}
    - Date range: {start_date} to {end_date}
    """
    
    # Define the specific question
    question = """
    Based on this sales data, what specific opportunities exist to improve conversion rates?
    Focus on:
    1. Patterns in successful conversions vs. lost opportunities
    2. Timing factors that influence conversion
    3. Lead source quality differences
    4. Guest size or event type correlations
    
    Provide 3-5 concrete, actionable recommendations with brief explanations.
    """
    
    # Generate insight using the agent
    return generate_agent_insight(data_summary, question)

def recommend_pricing_strategy(df):
    """
    Use the agent to recommend optimal pricing strategies
    
    Args:
        df (DataFrame): The processed sales data with deal values
        
    Returns:
        str: AI-generated pricing strategy recommendations
    """
    # Calculate price metrics if available
    avg_price = df['actual_deal_value'].mean() if 'actual_deal_value' in df.columns else 'Unknown'
    avg_price_per_guest = df['price_per_guest'].mean() if 'price_per_guest' in df.columns else 'Unknown'
    
    # Prepare data summary
    data_summary = f"""
    Pricing data overview:
    - Average deal value: ${avg_price:.2f} if isinstance(avg_price, (int, float)) else avg_price
    - Average price per guest: ${avg_price_per_guest:.2f} if isinstance(avg_price_per_guest, (int, float)) else avg_price_per_guest
    - Deal value range: ${df['actual_deal_value'].min():.2f} to ${df['actual_deal_value'].max():.2f} if 'actual_deal_value' in df.columns else 'Unknown'
    - Top booking types by volume: {', '.join(df['booking_type'].value_counts().head(3).index.tolist())}
    """
    
    # Define the specific question
    question = """
    Based on this pricing data, what pricing strategy recommendations would you make?
    Focus on:
    1. Optimal price points for different booking types
    2. Seasonal pricing adjustments
    3. Guest count-based pricing tiers
    4. Premium service opportunities
    
    Provide 3-4 specific pricing recommendations that could increase revenue without hurting conversion rates.
    """
    
    # Generate insight using the agent
    return generate_agent_insight(data_summary, question)

def suggest_lead_targeting(df):
    """
    Use the agent to suggest lead targeting improvements
    
    Args:
        df (DataFrame): The processed sales data with lead source info
        
    Returns:
        str: AI-generated lead targeting suggestions
    """
    # Get lead source data if available
    top_sources = df['referral_source'].value_counts().head(5).to_dict() if 'referral_source' in df.columns else {}
    
    # Calculate conversion by source if possible
    source_conv = {}
    if 'referral_source' in df.columns and 'outcome' in df.columns:
        source_conv = df.groupby('referral_source')['outcome'].mean().sort_values(ascending=False).head(5).to_dict()
    
    # Prepare data summary
    data_summary = f"""
    Lead source data:
    - Top sources by volume: {top_sources}
    - Top converting sources: {source_conv}
    - Best performing states: {df.groupby('state')['outcome'].mean().sort_values(ascending=False).head(3).index.tolist() if 'state' in df.columns else 'Unknown'}
    """
    
    # Define the specific question
    question = """
    Based on this lead source data, how should we optimize our lead targeting and acquisition?
    Focus on:
    1. Which lead sources provide the best ROI combining volume and quality
    2. Geographic targeting recommendations
    3. Seasonal or timing-based acquisition strategies
    4. Lead nurturing recommendations for different lead sources
    
    Provide 3-5 specific targeting recommendations that could improve both lead quality and conversion rates.
    """
    
    # Generate insight using the agent
    return generate_agent_insight(data_summary, question)