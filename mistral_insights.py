"""
ai_insights.py - AI-powered insights and recommendations for lead analysis

This module provides intelligent lead insights using AI,
suggesting next actions, ideal timing, and personalized talk tracks.
"""

import os
import json
import pandas as pd
from openai import OpenAI

# Check for API key
def get_ai_client():
    """Initialize and return an authenticated AI client"""
    # First check for MISTRAL_API_KEY for backward compatibility
    api_key = os.environ.get("MISTRAL_API_KEY")
    
    if not api_key:
        # Fall back to OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("No API key found in environment variables")
    
    # Use OpenAI client
    return OpenAI(api_key=api_key)

def generate_lead_recommendation(lead_data, model="gpt-3.5-turbo"):
    """
    Generate AI recommendation for next action on a lead
    
    Args:
        lead_data (dict): Lead data dictionary with score and category
        model (str): AI model to use (default: gpt-3.5-turbo)
    
    Returns:
        dict: Recommendation data or error message
    """
    # Ensure we have the key required fields
    required_fields = ['score', 'category', 'event_type', 'guest_count', 'days_since_inquiry']
    missing = [f for f in required_fields if f not in lead_data]
    
    if missing:
        return {
            'error': f"Missing required fields: {', '.join(missing)}",
            'next_action': "Complete lead profile",
            'confidence': 0.0,
            'talk_track': "Lead data is incomplete"
        }
    
    try:
        # Get AI client
        client = get_ai_client()
        
        # Format lead data for better prompt presentation
        formatted_lead = json.dumps(lead_data, indent=2)
        
        # Construct the prompt
        system_prompt = """
        You are a virtual sales coach for an event/venue business. Your job is to analyze lead data and recommend 
        the best next action to move the lead toward booking. 
        
        Based on the lead's profile (including score, category, event type, guest count, and days since inquiry), 
        recommend ONE specific, actionable next step the sales team should take.
        
        Your response should be structured as a JSON with:
        - next_action: short, specific action (max 10 words)
        - confidence: numeric value between 0 and 1
        - talk_track: suggested message (1-2 sentences) personalized to this lead
        - reasoning: brief explanation of why this action is recommended (max 25 words)
        
        Do not include any other text outside the JSON.
        """
        
        user_prompt = f"""
        Here is the lead data:
        {formatted_lead}
        
        Recommend the single best next action based on this data. Return ONLY a JSON object.
        """
        
        # Call the AI API
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"}
        )
        
        # Extract the response and parse JSON
        result_text = response.choices[0].message.content
        
        # Parse JSON response
        try:
            # Try to parse the entire response as JSON first
            result = json.loads(result_text)
        except json.JSONDecodeError:
            # If that fails, try to find and extract JSON within the text
            import re
            json_pattern = r'\{[\s\S]*\}'
            match = re.search(json_pattern, result_text)
            if match:
                try:
                    result = json.loads(match.group(0))
                except json.JSONDecodeError:
                    # If we still can't parse it, return a fallback response
                    return {
                        'error': "Failed to parse AI response",
                        'next_action': "Review lead manually",
                        'confidence': 0.5,
                        'talk_track': "Unable to generate automated recommendation",
                        'raw_response': result_text
                    }
            else:
                return {
                    'error': "No JSON found in AI response",
                    'next_action': "Review lead manually",
                    'confidence': 0.5,
                    'talk_track': "Unable to generate automated recommendation",
                    'raw_response': result_text
                }
        
        # Ensure all expected keys are present
        required_keys = ['next_action', 'confidence', 'talk_track', 'reasoning']
        for key in required_keys:
            if key not in result:
                result[key] = ""
        
        return result
        
    except Exception as e:
        # Return error with fallback recommendation
        return {
            'error': str(e),
            'next_action': "Review lead manually",
            'confidence': 0.5,
            'talk_track': "Unable to generate automated recommendation"
        }

def batch_generate_recommendations(df, limit=10):
    """
    Generate recommendations for multiple leads
    
    Args:
        df (DataFrame): DataFrame containing lead data
        limit (int): Maximum number of leads to process
    
    Returns:
        DataFrame: Original DataFrame with recommendation columns added
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Add recommendation columns
    result_df['ai_next_action'] = ""
    result_df['ai_confidence'] = 0.0
    result_df['ai_talk_track'] = ""
    result_df['ai_reasoning'] = ""
    
    # Process only a limited number of leads
    process_count = min(limit, len(df))
    
    # Process each lead
    for i in range(process_count):
        try:
            # Convert row to dict
            lead_data = df.iloc[i].to_dict()
            
            # Generate recommendation
            recommendation = generate_lead_recommendation(lead_data)
            
            # Add to DataFrame
            result_df.loc[df.index[i], 'ai_next_action'] = recommendation.get('next_action', '')
            result_df.loc[df.index[i], 'ai_confidence'] = recommendation.get('confidence', 0.0)
            result_df.loc[df.index[i], 'ai_talk_track'] = recommendation.get('talk_track', '')
            result_df.loc[df.index[i], 'ai_reasoning'] = recommendation.get('reasoning', '')
        except Exception as e:
            # Log error and continue
            print(f"Error processing lead {i}: {str(e)}")
    
    return result_df

def get_top_converting_actions(df):
    """
    Analyze which AI-recommended actions led to the highest conversion rates
    
    Args:
        df (DataFrame): DataFrame with AI recommendations and outcomes
        
    Returns:
        DataFrame: Summary of conversion rates by action
    """
    if 'ai_next_action' not in df.columns or 'outcome' not in df.columns:
        return pd.DataFrame()
    
    # Group by action and calculate conversion rate
    action_stats = df.groupby('ai_next_action').agg(
        total=('outcome', 'count'),
        converted=('outcome', 'sum'),
    ).reset_index()
    
    # Calculate conversion rate
    action_stats['conversion_rate'] = action_stats['converted'] / action_stats['total']
    action_stats['conversion_pct'] = action_stats['conversion_rate'] * 100
    
    # Sort by conversion rate
    return action_stats.sort_values('conversion_rate', ascending=False)

def webhook_send_recommendation(lead_data, endpoint_url=None):
    """
    Send a lead recommendation to a webhook endpoint (e.g., Zapier)
    
    Args:
        lead_data (dict): Lead data dictionary
        endpoint_url (str): Webhook URL to send the recommendation to
        
    Returns:
        dict: Response from the webhook endpoint
    """
    if not endpoint_url:
        return {'error': 'No webhook endpoint URL provided'}
    
    # Generate recommendation
    recommendation = generate_lead_recommendation(lead_data)
    
    # Combine lead data with recommendation
    payload = {
        'lead': lead_data,
        'recommendation': recommendation
    }
    
    # Send to webhook
    try:
        import requests
        response = requests.post(endpoint_url, json=payload)
        return {
            'status_code': response.status_code,
            'response': response.text
        }
    except Exception as e:
        return {'error': str(e)}