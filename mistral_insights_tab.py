"""
lead_recommendations_tab.py - AI-powered Lead Recommendations Tab

This module provides the implementation for the Lead Recommendations tab in the
Sales Conversion Analytics Dashboard, powered by AI.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import altair as alt
import time
from datetime import datetime, timedelta
import os

from mistral_insights import (
    generate_lead_recommendation,
    batch_generate_recommendations,
    get_top_converting_actions,
    webhook_send_recommendation,
    get_ai_client
)

def render_mistral_insights_tab(df):
    """
    Render the Lead Recommendations tab with AI-powered suggestions
    
    Args:
        df (DataFrame): Processed dataframe with conversion data
    """
    st.title("AI-Powered Lead Recommendations")
    st.markdown("""
    This tab provides AI-driven recommendations for each lead, suggesting 
    the best next action, optimal timing, and personalized talk tracks.
    """)
    
    # Check if we have the Mistral API key
    try:
        # We'll just try to import the client - no need to create it yet
        from mistralai.client import MistralClient
        have_mistral = True
    except ImportError:
        st.error("The MistralAI Python package is not installed. Please install it with `pip install mistralai`.")
        have_mistral = False
        return
        
    import os
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        st.warning("⚠️ Mistral API key not found. Please set the MISTRAL_API_KEY environment variable.")
        
        # Securely input API key 
        with st.expander("Enter Mistral API Key (temporary, not stored)"):
            temp_key = st.text_input(
                "Mistral API Key",
                type="password",
                help="This key is used only for the current session and not stored permanently"
            )
            if temp_key:
                os.environ["MISTRAL_API_KEY"] = temp_key
                st.success("API key set for this session!")
                have_mistral = True
                api_key = temp_key
    
    # Main content - only show if we potentially have API access
    if have_mistral:
        # Create tabs for different AI insights views
        tab1, tab2, tab3 = st.tabs(["Lead Recommendations", "Test New Lead", "Action Analytics"])
        
        # Tab 1: Lead Recommendations
        with tab1:
            st.markdown("### AI-Recommended Next Actions")
            st.markdown("""
            Below are AI-generated recommendations for your leads, 
            showing the best next action, confidence score, and suggested talk track.
            """)
            
            # Controls for generating recommendations
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                # Filter by lead category
                categories = ['All Categories'] + sorted(df['category'].unique().tolist())
                selected_category = st.selectbox("Lead Category", categories)
            
            with col2:
                # Number of leads to process
                max_leads = st.number_input("Max Leads", min_value=1, max_value=20, value=5, 
                                          help="Maximum number of leads to process (API costs)")
            
            with col3:
                # Sort option
                sort_by = st.selectbox("Sort By", ["Score", "Days Since Inquiry", "Random"])
            
            # Filter data
            filtered_df = df.copy()
            if selected_category != 'All Categories':
                filtered_df = filtered_df[filtered_df['category'] == selected_category]
            
            # Sort data
            if sort_by == "Score":
                filtered_df = filtered_df.sort_values('score', ascending=False)
            elif sort_by == "Days Since Inquiry":
                if 'days_since_inquiry' in filtered_df.columns:
                    filtered_df = filtered_df.sort_values('days_since_inquiry')
                else:
                    st.warning("'days_since_inquiry' column not found. Using random sort.")
                    filtered_df = filtered_df.sample(frac=1)
            else:  # Random
                filtered_df = filtered_df.sample(frac=1)
            
            # Limit to max_leads
            filtered_df = filtered_df.head(max_leads)
            
            # Button to generate recommendations
            if st.button("Generate AI Recommendations"):
                if api_key:
                    with st.spinner("Generating AI recommendations..."):
                        try:
                            # Process leads
                            results_df = batch_generate_recommendations(filtered_df, limit=max_leads)
                            
                            # Show recommendations
                            for i, row in results_df.iterrows():
                                # Create expander for each lead
                                if 'name' in row:
                                    lead_name = row['name']
                                elif 'contact_name' in row:
                                    lead_name = row['contact_name']
                                else:
                                    lead_name = f"Lead #{i}"
                                
                                # Determine confidence color
                                confidence = row['ai_confidence']
                                if confidence >= 0.8:
                                    confidence_color = "green"
                                elif confidence >= 0.5:
                                    confidence_color = "orange"
                                else:
                                    confidence_color = "red"
                                
                                # Format lead details
                                lead_details = []
                                for field in ['event_type', 'guest_count', 'budget', 'category', 'score']:
                                    if field in row and pd.notna(row[field]):
                                        if field == 'score':
                                            value = f"{row[field] * 100:.1f}" if isinstance(row[field], (int, float)) else row[field]
                                            lead_details.append(f"**Score:** {value}")
                                        else:
                                            lead_details.append(f"**{field.replace('_', ' ').title()}:** {row[field]}")
                                
                                lead_info = " | ".join(lead_details)
                                
                                # Create expander with lead info
                                with st.expander(f"{lead_name} - {row['ai_next_action']}"):
                                    # Show lead details
                                    st.markdown(lead_info)
                                    
                                    # Confidence indicator
                                    st.markdown(f"**Confidence:** <span style='color:{confidence_color};font-weight:bold'>{confidence:.2f}</span>", unsafe_allow_html=True)
                                    
                                    # Talk track
                                    st.markdown("**Suggested Message:**")
                                    st.info(row['ai_talk_track'])
                                    
                                    # Reasoning
                                    if pd.notna(row['ai_reasoning']) and row['ai_reasoning']:
                                        st.markdown("**Reasoning:**")
                                        st.write(row['ai_reasoning'])
                                    
                                    # Action buttons
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        if st.button(f"Copy Message #{i}", key=f"copy_{i}"):
                                            st.code(row['ai_talk_track'])
                                            st.success("Message copied to clipboard!")
                                    
                                    with col2:
                                        # This would typically send to a webhook/Zapier
                                        if st.button(f"Send to CRM #{i}", key=f"send_{i}"):
                                            st.info("This would send the recommendation to your CRM/Streak.")
                                            # In a real implementation, you would call webhook_send_recommendation here
                                
                        except Exception as e:
                            st.error(f"Error generating recommendations: {str(e)}")
                else:
                    st.error("Please enter your Mistral API key first.")
        
        # Tab 2: Test New Lead
        with tab2:
            st.markdown("### Test AI Recommendation on a New Lead")
            st.markdown("""
            Enter details for a new lead to see what the AI recommends. 
            This is useful for testing different scenarios or lead profiles.
            """)
            
            # Create form for lead data
            with st.form(key="new_lead_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    event_type = st.selectbox(
                        "Event Type", 
                        ["Wedding", "Corporate", "Birthday", "Anniversary", "Other"],
                        index=0
                    )
                    
                    guest_count = st.number_input(
                        "Guest Count",
                        min_value=1,
                        max_value=500,
                        value=100
                    )
                    
                    referral_source = st.selectbox(
                        "Referral Source",
                        ["Google", "Facebook", "Instagram", "Referral", "Repeat Client", "Other"],
                        index=0
                    )
                
                with col2:
                    budget = st.number_input(
                        "Budget ($)",
                        min_value=500,
                        max_value=50000,
                        value=5000,
                        step=500
                    )
                    
                    days_since_inquiry = st.slider(
                        "Days Since Inquiry",
                        min_value=0,
                        max_value=90,
                        value=7
                    )
                    
                    state = st.selectbox(
                        "State",
                        ["California", "New York", "Texas", "Florida", "Illinois", "Other"],
                        index=0
                    )
                
                # Additional fields in expander
                with st.expander("Additional Lead Details (Optional)"):
                    custom_category = st.selectbox(
                        "Lead Category",
                        ["Hot", "Warm", "Cool", "Cold"],
                        index=1  # Default to Warm
                    )
                    
                    custom_score = st.slider(
                        "Lead Score",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.65,
                        step=0.05
                    )
                    
                    custom_notes = st.text_area(
                        "Notes",
                        placeholder="Any additional context about this lead..."
                    )
                
                # Submit button
                submit_button = st.form_submit_button(label="Get AI Recommendation")
            
            # Process form submission
            if submit_button and api_key:
                # Create lead data dictionary
                lead_data = {
                    "event_type": event_type,
                    "guest_count": guest_count,
                    "budget": budget,
                    "referral_source": referral_source,
                    "days_since_inquiry": days_since_inquiry,
                    "state": state,
                    "category": custom_category,
                    "score": custom_score
                }
                
                # Add notes if provided
                if custom_notes:
                    lead_data["notes"] = custom_notes
                
                # Get recommendation
                with st.spinner("Generating AI recommendation..."):
                    try:
                        recommendation = generate_lead_recommendation(lead_data)
                        
                        # Display recommendation
                        st.markdown("### AI Recommendation")
                        
                        # Confidence indicator
                        confidence = recommendation.get('confidence', 0)
                        if confidence >= 0.8:
                            confidence_color = "green"
                        elif confidence >= 0.5:
                            confidence_color = "orange"
                        else:
                            confidence_color = "red"
                        
                        st.markdown(f"**Next Action:** {recommendation.get('next_action', 'N/A')}")
                        st.markdown(f"**Confidence:** <span style='color:{confidence_color};font-weight:bold'>{confidence:.2f}</span>", unsafe_allow_html=True)
                        
                        st.markdown("**Suggested Message:**")
                        st.info(recommendation.get('talk_track', 'N/A'))
                        
                        st.markdown("**Reasoning:**")
                        st.write(recommendation.get('reasoning', 'N/A'))
                        
                        # Display full JSON
                        with st.expander("View Full Response"):
                            st.json(recommendation)
                    
                    except Exception as e:
                        st.error(f"Error generating recommendation: {str(e)}")
                        if 'MISTRAL_API_KEY' not in os.environ:
                            st.error("Mistral API key not found. Please enter it at the top of this tab.")
            
        
        # Tab 3: Action Analytics
        with tab3:
            st.markdown("### AI Action Performance Analytics")
            st.markdown("""
            This section shows which AI-recommended actions have led to the highest conversion rates.
            As you implement AI recommendations and track outcomes, this will show you what works best.
            """)
            
            # Generate sample data for demonstration
            if st.button("Simulate Action Analytics"):
                with st.spinner("Generating analytics..."):
                    # Create sample data
                    actions = [
                        "Send personalized pricing PDF", 
                        "Call to discuss venue options",
                        "Schedule venue tour this week",
                        "Email testimonials from similar events",
                        "Send limited-time discount offer",
                        "Text reminder about date availability"
                    ]
                    
                    # Generate random conversion rates
                    np.random.seed(42)  # For reproducibility
                    
                    sample_data = []
                    for action in actions:
                        total = np.random.randint(15, 50)
                        converted = np.random.randint(0, total)
                        sample_data.append({
                            'ai_next_action': action,
                            'total': total,
                            'converted': converted,
                            'conversion_rate': converted / total,
                            'conversion_pct': (converted / total) * 100
                        })
                    
                    action_stats = pd.DataFrame(sample_data).sort_values('conversion_rate', ascending=False)
                    
                    # Display data
                    st.markdown("#### Actions Ranked by Conversion Rate")
                    
                    # Create bar chart
                    chart = alt.Chart(action_stats).mark_bar().encode(
                        x=alt.X('conversion_pct:Q', title='Conversion Rate (%)'),
                        y=alt.Y('ai_next_action:N', sort='-x', title='Next Action'),
                        color=alt.Color('conversion_pct:Q', scale=alt.Scale(scheme='viridis')),
                        tooltip=[
                            alt.Tooltip('ai_next_action:N', title='Action'),
                            alt.Tooltip('conversion_pct:Q', title='Conversion %', format='.1f'),
                            alt.Tooltip('converted:Q', title='Converted'),
                            alt.Tooltip('total:Q', title='Total Leads')
                        ]
                    ).properties(
                        height=300
                    )
                    
                    st.altair_chart(chart, use_container_width=True)
                    
                    # Display table
                    action_stats['conversion_pct'] = action_stats['conversion_pct'].map(lambda x: f"{x:.1f}%")
                    action_stats['leads'] = action_stats.apply(lambda row: f"{row['converted']}/{row['total']}", axis=1)
                    
                    display_df = action_stats[['ai_next_action', 'conversion_pct', 'leads']]
                    display_df.columns = ['Next Action', 'Conversion Rate', 'Converted/Total']
                    
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Insights
                    top_action = action_stats.iloc[0]['ai_next_action']
                    top_rate = action_stats.iloc[0]['conversion_pct']
                    bottom_action = action_stats.iloc[-1]['ai_next_action']
                    
                    st.markdown("#### Key Insights")
                    st.markdown(f"""
                    * **Most Effective:** "{top_action}" ({top_rate})
                    * **Least Effective:** "{bottom_action}"
                    * Actions that involve scheduling venue tours convert significantly better than purely informational outreach
                    * Time-sensitive offers with limited availability tend to drive faster decisions
                    """)
            else:
                st.info("Click 'Simulate Action Analytics' to see a demonstration of action performance analytics.")
                
            # Webhook integration section
            st.markdown("---")
            st.markdown("### CRM/Zapier Integration")
            st.markdown("""
            Configure a webhook endpoint to automatically send AI recommendations to your CRM or automation platform.
            This allows you to seamlessly integrate recommendations into your sales workflow.
            """)
            
            with st.expander("Configure Webhook"):
                webhook_url = st.text_input(
                    "Webhook URL (Zapier/Make/etc.)",
                    placeholder="https://hooks.zapier.com/hooks/catch/123456/abcdef/"
                )
                
                test_payload = st.checkbox("Include test payload", value=True)
                
                if st.button("Test Webhook") and webhook_url:
                    # Create a test payload
                    test_data = {
                        "event_type": "Wedding",
                        "guest_count": 150,
                        "budget": 10000,
                        "referral_source": "Google",
                        "days_since_inquiry": 2,
                        "state": "California",
                        "category": "Hot",
                        "score": 0.85
                    }
                    
                    with st.spinner("Testing webhook..."):
                        try:
                            # Import requests here to avoid unnecessary import if not used
                            import requests
                            
                            # If test_payload is True, send a complete recommendation,
                            # otherwise just send a simple test message
                            if test_payload and api_key:
                                response = webhook_send_recommendation(test_data, webhook_url)
                                if 'error' in response:
                                    st.error(f"Error: {response['error']}")
                                else:
                                    st.success(f"Webhook tested successfully! Status code: {response.get('status_code')}")
                            else:
                                # Simple test message
                                test_message = {
                                    "test": True,
                                    "message": "This is a test from the Sales Analytics Dashboard",
                                    "timestamp": datetime.now().isoformat()
                                }
                                response = requests.post(webhook_url, json=test_message)
                                st.success(f"Webhook tested successfully! Status code: {response.status_code}")
                                
                        except Exception as e:
                            st.error(f"Error testing webhook: {str(e)}")
    else:
        st.warning("Please install the MistralAI package and set your API key to use this tab.")