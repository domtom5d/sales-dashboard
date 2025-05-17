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
                
                # Get month order for proper sorting
                month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                              'July', 'August', 'September', 'October', 'November', 'December']
                
                # Create a proper plot with ordered months
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Convert month to categorical with correct order
                month_data = analytics_results['event_month'].copy()
                month_data['month'] = pd.Categorical(
                    month_data['month'], 
                    categories=month_order, 
                    ordered=True
                )
                month_data = month_data.sort_values('month')
                
                # Plot
                bars = ax.bar(month_data['month'], month_data['conversion'], color='skyblue')
                
                # Add data labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.1%}', ha='center', va='bottom')
                
                # Customize
                ax.set_title('Conversion Rate by Month', fontsize=14)
                ax.set_xlabel('Month', fontsize=12)
                ax.set_ylabel('Conversion Rate', fontsize=12)
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                plt.xticks(rotation=45)
                
                st.pyplot(fig)
                
                # Display data table
                st.markdown("**Monthly Conversion Rates:**")
                st.dataframe(month_data[['month', 'total', 'won', 'conversion']])
                
                # Add insights
                best_month = month_data.loc[month_data['conversion'].idxmax()]
                worst_month = month_data.loc[month_data['conversion'].idxmin()]
                st.info(f"""
                💡 **Seasonal Insights:** 
                - Best month: {best_month['month']} ({best_month['conversion']:.1%})
                - Worst month: {worst_month['month']} ({worst_month['conversion']:.1%})
                - Plan your sales efforts accordingly, focusing more resources during {worst_month['month']} to improve performance.
                """)
            else:
                st.info("No event month data available. Make sure your data includes event date information.")
        
        # Inquiry Weekday Analysis
        with col2:
            st.markdown("#### Day of Week Patterns")
            if (analytics_results.get('inquiry_weekday') is not None and 
                not analytics_results['inquiry_weekday'].empty):
                
                # Get weekday order for proper sorting
                weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                
                # Create a proper plot with ordered weekdays
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Convert weekday to categorical with correct order
                weekday_data = analytics_results['inquiry_weekday'].copy()
                weekday_data['weekday'] = pd.Categorical(
                    weekday_data['weekday'], 
                    categories=weekday_order, 
                    ordered=True
                )
                weekday_data = weekday_data.sort_values('weekday')
                
                # Plot
                bars = ax.bar(weekday_data['weekday'], weekday_data['conversion'], color='lightgreen')
                
                # Add data labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.1%}', ha='center', va='bottom')
                
                # Customize
                ax.set_title('Conversion Rate by Day of Week', fontsize=14)
                ax.set_xlabel('Day of Week', fontsize=12)
                ax.set_ylabel('Conversion Rate', fontsize=12)
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                
                st.pyplot(fig)
                
                # Display data table
                st.markdown("**Weekday Conversion Rates:**")
                st.dataframe(weekday_data[['weekday', 'total', 'won', 'conversion']])
                
                # Add insights
                best_day = weekday_data.loc[weekday_data['conversion'].idxmax()]
                worst_day = weekday_data.loc[weekday_data['conversion'].idxmin()]
                st.info(f"""
                💡 **Day of Week Insights:** 
                - Best day: {best_day['weekday']} ({best_day['conversion']:.1%})
                - Worst day: {worst_day['weekday']} ({worst_day['conversion']:.1%})
                - Consider adjusting your follow-up strategy based on the day of week.
                """)
            else:
                st.info("No inquiry weekday data available. Make sure your data includes inquiry date information.")
        
        # Days Since Inquiry Analysis
        st.markdown("#### Lead Time Analysis")
        if (analytics_results.get('days_since_inquiry') is not None and 
            not analytics_results['days_since_inquiry'].empty):
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Sort by numeric bin value to ensure correct order
            # First extract lower bound of each bin and convert to numeric
            time_data = analytics_results['days_since_inquiry'].copy()
            
            # Try to sort the data if possible
            try:
                # This approach handles bins like '0-7', '8-14', etc.
                time_data['bin_order'] = time_data['days_bin'].str.split('-').str[0].astype(float)
                time_data = time_data.sort_values('bin_order')
            except:
                # Fallback to using the data as is
                pass
            
            # Plot
            bars = ax.bar(time_data['days_bin'], time_data['conversion'], color='coral')
            
            # Add data labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.1%}', ha='center', va='bottom')
            
            # Customize
            ax.set_title('Conversion Rate by Days Since Inquiry', fontsize=14)
            ax.set_xlabel('Days Since Inquiry', fontsize=12)
            ax.set_ylabel('Conversion Rate', fontsize=12)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            st.pyplot(fig)
            
            # Display data table
            st.markdown("**Lead Time Conversion Rates:**")
            st.dataframe(time_data[['days_bin', 'total', 'won', 'conversion']])
            
            # Add insights
            best_bin = time_data.loc[time_data['conversion'].idxmax()]
            st.info(f"""
            💡 **Lead Time Insights:** 
            - Best conversion occurs {best_bin['days_bin']} days after inquiry ({best_bin['conversion']:.1%})
            - Understanding your optimal follow-up window can significantly improve conversion rates
            """)
        else:
            st.info("No days since inquiry data available. Make sure your data includes appropriate date fields.")
    
    # Tab 4: Price Analysis
    with analysis_tabs[3]:
        st.markdown("### Price Analysis")
        st.markdown("Analyze how pricing affects conversion rates and identify optimal price points.")
        
        # Price per Guest Analysis
        if (analytics_results.get('price_per_guest') is not None and 
            not analytics_results['price_per_guest'].empty):
            
            # Plot
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Sort data by bin order if possible
            price_data = analytics_results['price_per_guest'].copy()
            
            # Try to sort the price bins
            try:
                # This approach handles bins like '$0-$50', '$51-$100', etc.
                price_data['bin_order'] = price_data['price_bin'].str.replace('$', '').str.split('-').str[0].astype(float)
                price_data = price_data.sort_values('bin_order')
            except:
                # Fallback to using the data as is
                pass
            
            # Plot with dual axes - bars for conversion, line for volume
            bars = ax.bar(price_data['price_bin'], price_data['conversion'], color='lightblue', alpha=0.7)
            
            # Create twin axis for volume
            ax2 = ax.twinx()
            ax2.plot(price_data['price_bin'], price_data['total'], 'ro-', linewidth=2, markersize=8)
            
            # Add data labels for conversion bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.1%}', ha='center', va='bottom')
            
            # Customize
            ax.set_title('Conversion Rate and Volume by Price per Guest', fontsize=14)
            ax.set_xlabel('Price per Guest', fontsize=12)
            ax.set_ylabel('Conversion Rate', fontsize=12)
            ax2.set_ylabel('Number of Leads', fontsize=12, color='red')
            ax2.tick_params(axis='y', colors='red')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            plt.xticks(rotation=45)
            
            # Add legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='lightblue', lw=0, marker='s', markersize=15, label='Conversion Rate'),
                Line2D([0], [0], color='red', marker='o', markersize=8, label='Lead Volume')
            ]
            ax.legend(handles=legend_elements, loc='upper right')
            
            st.pyplot(fig)
            
            # Display data table
            st.markdown("**Price per Guest Analysis:**")
            st.dataframe(price_data[['price_bin', 'total', 'won', 'conversion']])
            
            # Add insights
            best_price = price_data.loc[price_data['conversion'].idxmax()]
            highest_volume = price_data.loc[price_data['total'].idxmax()]
            
            st.info(f"""
            💡 **Price Insights:** 
            - Highest conversion at: {best_price['price_bin']} ({best_price['conversion']:.1%})
            - Most popular price range: {highest_volume['price_bin']} ({highest_volume['total']} leads)
            - Optimize your pricing strategy based on the sweet spot between volume and conversion
            """)
        else:
            st.info("No price per guest data available. Make sure your data includes price and guest count information.")
    
    # Tab 5: Staffing Ratio Analysis
    with analysis_tabs[4]:
        st.markdown("### Staffing Ratio Analysis")
        st.markdown("Explore how staff-to-guest ratios affect conversion rates and client satisfaction.")
        
        # Staff Ratio Analysis
        if (analytics_results.get('staff_ratio') is not None and 
            not analytics_results['staff_ratio'].empty):
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Sort data by ratio bin if possible
            ratio_data = analytics_results['staff_ratio'].copy()
            
            # Try to sort by ratio bin numerically
            try:
                # This extracts numeric value from ratio bin labels
                ratio_data['bin_order'] = ratio_data['ratio_bin'].str.extract('(\d+)').astype(float)
                ratio_data = ratio_data.sort_values('bin_order')
            except:
                # Fall back to original data order
                pass
            
            # Plot
            bars = ax.bar(ratio_data['ratio_bin'], ratio_data['conversion'], color='mediumpurple')
            
            # Add data labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.1%}', ha='center', va='bottom')
            
            # Customize
            ax.set_title('Conversion Rate by Staff-to-Guest Ratio', fontsize=14)
            ax.set_xlabel('Staff-to-Guest Ratio', fontsize=12)
            ax.set_ylabel('Conversion Rate', fontsize=12)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            st.pyplot(fig)
            
            # Display data table
            st.markdown("**Staff Ratio Analysis:**")
            st.dataframe(ratio_data[['ratio_bin', 'total', 'won', 'conversion']])
            
            # Add insights
            best_ratio = ratio_data.loc[ratio_data['conversion'].idxmax()]
            st.info(f"""
            💡 **Staffing Insights:** 
            - Optimal staff-to-guest ratio: {best_ratio['ratio_bin']} ({best_ratio['conversion']:.1%} conversion)
            - Consider how staffing recommendations affect client decisions
            - Higher ratios may suggest premium service but could affect pricing sensitivity
            """)
        else:
            st.info("No staff ratio data available. Make sure your data includes information about staffing and guest counts.")
    
    # Additional notes and recommendations
    st.markdown("---")
    st.markdown("### Analysis Recommendations")
    st.info("""
    **How to use these insights:**
    
    1. **Focus marketing efforts** on your highest-converting referral and marketing sources
    2. **Adjust your sales approach** based on event type and seasonal patterns
    3. **Optimize your follow-up timing** based on the day of week and lead time analyses
    4. **Review your pricing strategy** to find the optimal balance between conversion rate and volume
    5. **Consider your staffing recommendations** to clients based on the staffing ratio analysis
    """)