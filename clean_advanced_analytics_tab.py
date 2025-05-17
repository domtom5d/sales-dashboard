"""
advanced_analytics_tab.py - Advanced Analytics Tab Module

This module provides the implementation for the Advanced Analytics tab
in the Sales Conversion Analytics Dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from advanced_analytics import run_all_analytics, plot_conversion_by_category

def render_advanced_analytics_tab(df):
    """
    Render the Advanced Analytics tab with specialized analyses and visualizations
    
    Args:
        df (DataFrame): Processed dataframe with conversion data
    """
    st.subheader("Advanced Analytics")
    st.markdown("""
    This tab provides deeper insights into your sales conversion data through specialized analyses.
    Explore how different factors affect your conversion rates and identify opportunities for improvement.
    """)
    
    # Check if analytics results already exist in session state
    if 'analytics_results' in st.session_state:
        st.success("Showing previously computed analytics results.")
        analytics_results = st.session_state.analytics_results
    else:
        # Show computing message
        with st.spinner("Computing advanced analytics..."):
            # Run all analytics functions
            analytics_results = run_all_analytics(df)
            # Store in session state for faster access on tab revisits
            st.session_state.analytics_results = analytics_results
    
    # Create a tabbed interface for different analyses
    analysis_tabs = st.tabs([
        "Referral & Marketing", 
        "Booking Types",
        "Timing Factors",
        "Price Analysis", 
        "Staffing Ratio"
    ])
    
    # Tab 1: Referral and Marketing Sources Analysis
    with analysis_tabs[0]:
        st.markdown("### Lead Source Analysis")
        st.markdown("Compare conversion rates across different lead sources to identify your best-performing channels.")
        
        col1, col2 = st.columns(2)
        
        # Referral Sources Analysis
        with col1:
            st.markdown("#### Referral Sources")
            if (analytics_results.get('referral_sources') is not None and 
                not analytics_results['referral_sources'].empty):
                
                # Plot referral sources
                fig, ax = plt.subplots(figsize=(10, 6))
                plot_conversion_by_category(
                    df, 
                    'referral_source', 
                    "Conversion by Referral Source", 
                    ax=ax,
                    sort_by='conversion', 
                    top_n=8
                )
                st.pyplot(fig)
                
                # Display data table
                st.markdown("**Detailed Metrics:**")
                st.dataframe(analytics_results['referral_sources'])
                
                # Add insights
                if len(analytics_results['referral_sources']) >= 2:
                    best = analytics_results['referral_sources'].iloc[0]
                    worst = analytics_results['referral_sources'].iloc[-1]
                    st.info(f"💡 **Insight:** '{best['referral_source']}' leads convert at {best['Conversion %']} compared to '{worst['referral_source']}' at {worst['Conversion %']}.")
            else:
                st.info("No referral source data available. Make sure your data includes a 'referral_source' column.")
        
        # Marketing Sources Analysis
        with col2:
            st.markdown("#### Marketing Sources")
            if (analytics_results.get('marketing_sources') is not None and 
                not analytics_results['marketing_sources'].empty):
                
                # Plot marketing sources
                fig, ax = plt.subplots(figsize=(10, 6))
                plot_conversion_by_category(
                    df, 
                    'marketing_source', 
                    "Conversion by Marketing Source", 
                    ax=ax,
                    sort_by='conversion', 
                    top_n=8
                )
                st.pyplot(fig)
                
                # Display data table
                st.markdown("**Detailed Metrics:**")
                st.dataframe(analytics_results['marketing_sources'])
                
                # Add insights
                if len(analytics_results['marketing_sources']) >= 2:
                    best = analytics_results['marketing_sources'].iloc[0]
                    worst = analytics_results['marketing_sources'].iloc[-1]
                    st.info(f"💡 **Insight:** '{best['marketing_source']}' campaigns convert at {best['Conversion %']} compared to '{worst['marketing_source']}' at {worst['Conversion %']}.")
            else:
                st.info("No marketing source data available. Make sure your data includes a 'marketing_source' column.")
    
    # Tab 2: Booking Types Analysis
    with analysis_tabs[1]:
        # Get column used from analytics results (booking_type or event_type)
        column_used = analytics_results.get('booking_type_column_used', 'booking_type')
        title_suffix = "Booking Type" if column_used == 'booking_type' else "Event Type"
        
        # Set appropriate header title based on which data is being used
        st.markdown(f"### {title_suffix} Analysis")
        st.markdown("Understand which types of events have the highest conversion rates and revenue potential.")
        
        if (analytics_results.get('booking_types') is not None and 
            not analytics_results['booking_types'].empty):
            
            # Plot booking/event types using the correct column
            fig, ax = plt.subplots(figsize=(10, 8))
            plot_conversion_by_category(
                df, 
                column_used,  # Use the column that was used in the analysis
                f"Conversion by {title_suffix}", 
                ax=ax,
                sort_by='conversion'
            )
            st.pyplot(fig)
            
            # Display data table
            st.markdown("**Detailed Metrics:**")
            st.dataframe(analytics_results['booking_types'])
            
            # Add insights
            if len(analytics_results['booking_types']) >= 2:
                best = analytics_results['booking_types'].iloc[0]
                worst = analytics_results['booking_types'].iloc[-1]
                
                # Get the field name from the dataframe
                field_name = analytics_results['booking_types'].columns[0]
                
                st.info(f"""
                💡 **Insights:** 
                - '{best[field_name]}' has the highest conversion rate at {best['Conversion %']}
                - '{worst[field_name]}' has the lowest conversion rate at {worst['Conversion %']}
                - Consider adjusting your sales approach based on event type
                """)
                
                # Add additional analysis by event/booking type if available
                if 'deal_value' in df.columns or 'actual_deal_value' in df.columns:
                    value_col = 'actual_deal_value' if 'actual_deal_value' in df.columns else 'deal_value'
                    
                    # Filter for won deals with non-null values
                    won_deals = df[df['outcome'] == 1].dropna(subset=[value_col])
                    
                    if not won_deals.empty and column_used in won_deals.columns:
                        avg_by_type = won_deals.groupby(column_used)[value_col].mean().reset_index()
                        avg_by_type[f'Average {value_col}'] = avg_by_type[value_col].map('${:,.2f}'.format)
                        
                        st.markdown("#### Average Deal Value by Type")
                        st.dataframe(avg_by_type[[column_used, f'Average {value_col}']].sort_values(value_col, ascending=False))
        else:
            st.info(f"No {title_suffix.lower()} data available. Make sure your data includes a '{column_used}' column.")
    
    # Tab 3: Timing Factors Analysis
    with analysis_tabs[2]:
        st.markdown("### Timing Factors Analysis")
        st.markdown("Examine how timing affects your conversion rates, including seasonality and day-of-week patterns.")
        
        col1, col2 = st.columns(2)
        
        # Event Month Analysis
        with col1:
            st.markdown("#### Seasonal Trends")
            if (analytics_results.get('event_month') is not None and 
                not analytics_results['event_month'].empty):
                
                # Create a proper plot with ordered months
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot data (already ordered by month in the analysis function)
                month_data = analytics_results['event_month'].copy()
                bars = ax.bar(month_data['event_month'], month_data['Conversion'], color='skyblue')
                
                # Add data labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.1%}', ha='center', va='bottom')
                
                # Customize plot
                ax.set_title('Conversion Rate by Month', fontsize=14)
                ax.set_xlabel('Month', fontsize=12)
                ax.set_ylabel('Conversion Rate', fontsize=12)
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                st.pyplot(fig)
                
                # Display data table
                st.markdown("**Monthly Conversion Details:**")
                st.dataframe(month_data)
                
                # Add insights
                if len(month_data) >= 2:
                    best_month = month_data.sort_values('Conversion', ascending=False).iloc[0]
                    worst_month = month_data.sort_values('Conversion').iloc[0]
                    st.info(f"💡 **Seasonal Insight:** {best_month['event_month']} has your highest conversion rate at {best_month['Conversion %']}, while {worst_month['event_month']} has the lowest at {worst_month['Conversion %']}.")
            else:
                st.info("No event date data available for seasonal analysis. Make sure your data includes valid 'event_date' values.")
        
        # Inquiry Weekday Analysis
        with col2:
            st.markdown("#### Day of Week Patterns")
            if (analytics_results.get('inquiry_weekday') is not None and 
                not analytics_results['inquiry_weekday'].empty):
                
                # Create a proper plot with ordered weekdays
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot data (already ordered by weekday in the analysis function)
                weekday_data = analytics_results['inquiry_weekday'].copy()
                bars = ax.bar(weekday_data['inquiry_weekday'], weekday_data['Conversion'], color='lightgreen')
                
                # Add data labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.1%}', ha='center', va='bottom')
                
                # Customize plot
                ax.set_title('Conversion Rate by Day of Week', fontsize=14)
                ax.set_xlabel('Day of Week', fontsize=12)
                ax.set_ylabel('Conversion Rate', fontsize=12)
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                
                st.pyplot(fig)
                
                # Display data table
                st.markdown("**Weekday Conversion Details:**")
                st.dataframe(weekday_data)
                
                # Add insights
                if len(weekday_data) >= 2:
                    best_day = weekday_data.sort_values('Conversion', ascending=False).iloc[0]
                    worst_day = weekday_data.sort_values('Conversion').iloc[0]
                    st.info(f"💡 **Weekday Insight:** Inquiries on {best_day['inquiry_weekday']} convert at {best_day['Conversion %']}, while {worst_day['inquiry_weekday']} inquiries convert at {worst_day['Conversion %']}.")
            else:
                st.info("No inquiry date data available for weekday analysis. Make sure your data includes valid 'inquiry_date' values.")
                
        # Days Since Inquiry Analysis
        st.markdown("#### Lead Time Analysis")
        if (analytics_results.get('days_since_inquiry') is not None and 
            not analytics_results['days_since_inquiry'].empty):
            
            # Create a proper plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot data
            dsi_data = analytics_results['days_since_inquiry'].copy()
            bars = ax.bar(dsi_data['dsi_bin'], dsi_data['Conversion'], color='orchid')
            
            # Add data labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.1%}', ha='center', va='bottom')
            
            # Customize plot
            ax.set_title('Conversion Rate by Days Since Inquiry', fontsize=14)
            ax.set_xlabel('Days Since Inquiry', fontsize=12)
            ax.set_ylabel('Conversion Rate', fontsize=12)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            st.pyplot(fig)
            
            # Display data table
            st.markdown("**Lead Time Conversion Details:**")
            st.dataframe(dsi_data)
            
            # Add insights
            if len(dsi_data) >= 2:
                best_dsi = dsi_data.sort_values('Conversion', ascending=False).iloc[0]
                worst_dsi = dsi_data.sort_values('Conversion').iloc[0]
                st.info(f"💡 **Lead Time Insight:** Leads in the {best_dsi['dsi_bin']} range convert at {best_dsi['Conversion %']}, while leads in the {worst_dsi['dsi_bin']} range convert at {worst_dsi['Conversion %']}.")
        else:
            st.info("No lead time data available. Make sure your data includes 'days_since_inquiry' values.")
    
    # Tab 4: Price Analysis
    with analysis_tabs[3]:
        st.markdown("### Price Analysis")
        st.markdown("Understand how pricing affects your conversion rates and identify optimal price points.")
        
        if (analytics_results.get('price_per_guest') is not None and 
            not analytics_results['price_per_guest'].empty):
            
            # Create a proper plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot data
            price_data = analytics_results['price_per_guest'].copy()
            bars = ax.bar(price_data['price_bin'], price_data['Conversion'], color='gold')
            
            # Add data labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.1%}', ha='center', va='bottom')
            
            # Customize plot
            ax.set_title('Conversion Rate by Price Per Guest', fontsize=14)
            ax.set_xlabel('Price Per Guest', fontsize=12)
            ax.set_ylabel('Conversion Rate', fontsize=12)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            st.pyplot(fig)
            
            # Display data table
            st.markdown("**Price Conversion Details:**")
            st.dataframe(price_data)
            
            # Add insights
            if len(price_data) >= 2:
                best_price = price_data.sort_values('Conversion', ascending=False).iloc[0]
                worst_price = price_data.sort_values('Conversion').iloc[0]
                st.info(f"💡 **Price Insight:** The {best_price['price_bin']} price range converts at {best_price['Conversion %']}, while the {worst_price['price_bin']} price range converts at {worst_price['Conversion %']}.")
        else:
            st.info("No price data available. Make sure your data includes 'price_per_guest' values.")
    
    # Tab 5: Staffing Ratio Analysis
    with analysis_tabs[4]:
        st.markdown("### Staffing Ratio Analysis")
        st.markdown("Examine how staffing levels affect your conversion rates and identify optimal staff-to-guest ratios.")
        
        if (analytics_results.get('staff_ratio') is not None and 
            not analytics_results['staff_ratio'].empty):
            
            # Create a proper plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot data
            staff_data = analytics_results['staff_ratio'].copy()
            bars = ax.bar(staff_data['staff_ratio_bin'], staff_data['Conversion'], color='lightcoral')
            
            # Add data labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.1%}', ha='center', va='bottom')
            
            # Customize plot
            ax.set_title('Conversion Rate by Staff-to-Guest Ratio', fontsize=14)
            ax.set_xlabel('Staff-to-Guest Ratio', fontsize=12)
            ax.set_ylabel('Conversion Rate', fontsize=12)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            st.pyplot(fig)
            
            # Display data table
            st.markdown("**Staffing Ratio Conversion Details:**")
            st.dataframe(staff_data)
            
            # Add insights
            if len(staff_data) >= 2:
                best_ratio = staff_data.sort_values('Conversion', ascending=False).iloc[0]
                worst_ratio = staff_data.sort_values('Conversion').iloc[0]
                st.info(f"💡 **Staffing Insight:** The {best_ratio['staff_ratio_bin']} staff ratio converts at {best_ratio['Conversion %']}, while the {worst_ratio['staff_ratio_bin']} staff ratio converts at {worst_ratio['Conversion %']}.")
        else:
            st.info("No staffing data available. Make sure your data includes both 'bartenders_needed' and 'number_of_guests' values.")