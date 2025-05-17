"""
findings.py - Generate dynamic key findings from analysis data

This module contains functions to extract key findings from the analyzed data
and generate human-readable insights.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

def generate_findings(df, y_scores, thresholds, weights_df=None):
    """
    Generate key findings from the analyzed data
    
    Args:
        df (DataFrame): Processed dataframe with outcome
        y_scores (array-like): Model prediction scores
        thresholds (dict): Dictionary of thresholds for categories
        weights_df (DataFrame, optional): DataFrame with feature weights/importance
        
    Returns:
        list: List of findings as formatted strings
    """
    findings = []
    
    # Add insights from model feature importance if available
    if weights_df is not None and not weights_df.empty:
        try:
            # Get top positive and negative features
            weights_df_sorted = weights_df.sort_values(by='weight', key=abs, ascending=False)
            
            if 'weight' in weights_df_sorted.columns:
                positive_features = weights_df_sorted[weights_df_sorted['weight'] > 0].head(3)
                negative_features = weights_df_sorted[weights_df_sorted['weight'] < 0].head(3)
                
                if not positive_features.empty:
                    top_pos = positive_features.iloc[0]
                    findings.append(f"**Top Predictor:** Higher **{top_pos['feature']}** values strongly correlate with won deals (weight: +{top_pos['weight']:.2f}).")
                
                if not negative_features.empty:
                    top_neg = negative_features.iloc[0]
                    findings.append(f"**Top Risk Factor:** Higher **{top_neg['feature']}** values correlate with lost deals (weight: {top_neg['weight']:.2f}).")
        except Exception as e:
            # Silently skip if feature importance analysis fails
            pass
    
    # Add overall conversion rate
    try:
        overall_rate = df['outcome'].mean()
        findings.insert(0, f"**Overall Conversion Rate:** {overall_rate:.1%} of leads convert to won deals.")
    except Exception as e:
        # Silently skip if calculation fails
        pass
        
    # Add high vs. low score conversion comparison
    try:
        if y_scores is not None and len(y_scores) > 0:
            # Create a DataFrame with scores and outcomes
            score_df = pd.DataFrame({
                'score': y_scores,
                'outcome': df['outcome'].values
            })
            
            # Split into high and low score groups using median
            median_score = np.median(y_scores)
            high_scores = score_df[score_df['score'] >= median_score]
            low_scores = score_df[score_df['score'] < median_score]
            
            # Calculate conversion rates for each group
            high_conv_rate = high_scores['outcome'].mean()
            low_conv_rate = low_scores['outcome'].mean()
            
            # Add finding if there's a meaningful difference
            if high_conv_rate > low_conv_rate * 1.25:  # At least 25% better
                findings.append(f"**Model Validation:** Leads with high scores convert at {high_conv_rate:.1%}, vs. low scores at {low_conv_rate:.1%}, a {(high_conv_rate/low_conv_rate - 1):.0%} improvement.")
    except Exception as e:
        # Silently skip if calculation fails
        pass
    
    # Add time to conversion analysis
    try:
        from conversion import analyze_time_to_conversion
        time_result = analyze_time_to_conversion(df)
        if not time_result.get('error'):
            avg_days = time_result.get('average_days')
            median_days = time_result.get('median_days')
            findings.append(f"**Time to Conversion:** Average: {avg_days:.1f} days, Median: {median_days:.1f} days.")
            
            # Add insights by event type if available - using the categorized event types
            if 'by_event_type' in time_result and not time_result['by_event_type'].empty:
                # Find fastest and slowest converting event types with at least 3 data points
                event_df = time_result['by_event_type']
                event_df_filtered = event_df[event_df['count'] >= 3]
                
                if not event_df_filtered.empty:
                    fastest = event_df_filtered.loc[event_df_filtered['mean'].idxmin()]
                    slowest = event_df_filtered.loc[event_df_filtered['mean'].idxmax()]
                    
                    # Only add if there's a meaningful difference (more than 1 day)
                    if abs(fastest['mean'] - slowest['mean']) > 1:
                        findings.append(f"**Event Type Conversion Speed:** {fastest['event_type']} events convert fastest ({fastest['mean']:.1f} days), while {slowest['event_type']} events take longest ({slowest['mean']:.1f} days).")
                        
                        # Add the comparison with average
                        avg_days = time_result.get('average_days')
                        if fastest['mean'] < avg_days * 0.7:  # If notably faster than average
                            findings.append(f"**Conversion Opportunity:** {fastest['event_type']} events convert {(avg_days - fastest['mean']):.1f} days faster than average, suggesting a sales focus area.")
                        
                        if slowest['mean'] > avg_days * 1.3:  # If notably slower than average
                            findings.append(f"**Conversion Challenge:** {slowest['event_type']} events take {(slowest['mean'] - avg_days):.1f} days longer than average, suggesting a need for improved follow-up.")
            
            # Add insights by booking type if available
            if 'by_booking_type' in time_result and not time_result['by_booking_type'].empty:
                # Find fastest and slowest converting booking types with at least 3 data points
                booking_df = time_result['by_booking_type']
                booking_df_filtered = booking_df[booking_df['count'] >= 3]
                
                if not booking_df_filtered.empty:
                    fastest = booking_df_filtered.loc[booking_df_filtered['mean'].idxmin()]
                    slowest = booking_df_filtered.loc[booking_df_filtered['mean'].idxmax()]
                    
                    # Only add if there's a meaningful difference (more than 1 day)
                    if abs(fastest['mean'] - slowest['mean']) > 1:
                        findings.append(f"**Booking Type Conversion Speed:** {fastest['booking_type']} bookings convert fastest ({fastest['mean']:.1f} days), while {slowest['booking_type']} bookings take longest ({slowest['mean']:.1f} days).")
    except Exception as e:
        # Uncomment for debugging but keep silently failing in production
        # print(f"Error generating time to conversion findings: {e}")
        pass  # Silently skip if the time analysis fails

    # 1. Urgency
    try:
        urg = df.assign(bin=lambda d: pd.cut(d['days_until_event'], [0, 7, 30, np.inf], 
                                         labels=['≤7d', '8–30d', '>30d']))
        urg_rates = urg.groupby('bin')['outcome'].mean()
        best_urg = urg_rates.idxmax(), urg_rates.max()
        worst_urg = urg_rates.idxmin(), urg_rates.min()
        findings.append(f"**Urgency:** Leads {best_urg[0]} convert at {best_urg[1]:.0%}, vs. {worst_urg[0]} at {worst_urg[1]:.0%}.")
    except Exception as e:
        findings.append(f"**Urgency:** Insufficient data to analyze days-to-event impact. ({str(e)})")

    # 2. Region
    try:
        if 'region' in df.columns or 'state' in df.columns:
            region_col = 'region' if 'region' in df.columns else 'state'
            reg_rates = df.groupby(region_col)['outcome'].mean()
            top_reg = reg_rates.idxmax(), reg_rates.max()
            bot_reg = reg_rates.idxmin(), reg_rates.min()
            findings.append(f"**Geography:** {top_reg[0]} leads close at {top_reg[1]:.0%}, while {bot_reg[0]} at {bot_reg[1]:.0%}.")
    except Exception as e:
        findings.append(f"**Geography:** Insufficient data to analyze regional differences. ({str(e)})")

    # 3. Seasonality (Month)
    try:
        if 'inquiry_date' in df.columns:
            df['month'] = pd.to_datetime(df['inquiry_date']).dt.month_name()
            mon_rates = df.groupby('month')['outcome'].mean()
            best_mon = mon_rates.idxmax(), mon_rates.max()
            worst_mon = mon_rates.idxmin(), mon_rates.min()
            findings.append(f"**Seasonality:** {best_mon[0]} month has {best_mon[1]:.0%}, lowest is {worst_mon[0]} at {worst_mon[1]:.0%}.")
        elif 'event_date' in df.columns:
            df['month'] = pd.to_datetime(df['event_date']).dt.month_name()
            mon_rates = df.groupby('month')['outcome'].mean()
            best_mon = mon_rates.idxmax(), mon_rates.max()
            worst_mon = mon_rates.idxmin(), mon_rates.min()
            findings.append(f"**Seasonality:** {best_mon[0]} month has {best_mon[1]:.0%}, lowest is {worst_mon[0]} at {worst_mon[1]:.0%}.")
    except Exception as e:
        findings.append(f"**Seasonality:** Insufficient data to analyze seasonal trends. ({str(e)})")

    # 4. Corporate vs Social
    try:
        if 'booking_type' in df.columns:
            # Create is_corporate flag based on booking type
            df['is_corporate'] = df['booking_type'].str.lower().str.contains('corporate|company|business').fillna(False)
            corp_rates = df.groupby(df['is_corporate'].map({True: 'Corporate', False: 'Social'}))['outcome'].mean()
            findings.append(f"**Event Type:** Corporate at {corp_rates.get('Corporate', 0):.0%}, Social at {corp_rates.get('Social', 0):.0%}.")
    except Exception as e:
        findings.append(f"**Event Type:** Insufficient data to analyze event types. ({str(e)})")

    # 5. PhoneMatch uplift
    try:
        if 'phone_match' in df.columns:
            pm_rates = df.groupby('phone_match')['outcome'].mean()
            if True in pm_rates.index and False in pm_rates.index:
                findings.append(f"**Phone‐Match:** Local numbers convert at {pm_rates[True]:.0%} vs. non‐local at {pm_rates[False]:.0%}.")
    except Exception as e:
        findings.append(f"**Phone‐Match:** Insufficient data to analyze phone matching. ({str(e)})")

    # 6. Model performance
    try:
        roc = roc_auc_score(df['outcome'], y_scores)
        prec, rec, _ = precision_recall_curve(df['outcome'], y_scores)
        pr_auc = auc(rec, prec)
        findings.append(f"**Model AUC:** ROC={roc:.3f}, PR={pr_auc:.3f}.")
    except Exception as e:
        findings.append(f"**Model Performance:** Model metrics not available. ({str(e)})")

    # 7. Bucket distribution
    try:
        hot = (y_scores >= thresholds['hot']).sum()
        warm = ((y_scores >= thresholds['warm']) & (y_scores < thresholds['hot'])).sum()
        cool = ((y_scores >= thresholds['cool']) & (y_scores < thresholds['warm'])).sum()
        cold = (y_scores < thresholds['cool']).sum()
        findings.append(f"**Buckets:** {hot} Hot, {warm} Warm, {cool} Cool, {cold} Cold.")
    except Exception as e:
        findings.append(f"**Buckets:** Score distribution not available. ({str(e)})")

    return findings