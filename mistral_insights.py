"""
mistral_insights.py - Integration with Mistral AI for generating insights

This module provides functions to connect to the Mistral AI API
and generate insights from data.
"""

import os
import pandas as pd
import streamlit as st
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

def initialize_mistral_client():
    """
    Initialize the Mistral AI client using the API key.
    
    Returns:
        MistralClient or None: Initialized client or None if API key is missing
    """
    api_key = os.environ.get("MISTRAL_API_KEY")
    
    if not api_key:
        return None
    
    try:
        client = MistralClient(api_key=api_key)
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
        # Call Mistral AI API
        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt)
        ]
        
        response = client.chat(
            model="mistral-medium",
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    
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