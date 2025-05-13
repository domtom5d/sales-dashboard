import pandas as pd
import numpy as np

def process_data(leads_df, operations_df=None):
    """
    Process and prepare the data for analysis.
    
    Args:
        leads_df (DataFrame): The leads data from Streak export
        operations_df (DataFrame, optional): The operations data from Streak export
    
    Returns:
        DataFrame: Processed dataframe with outcome and binned columns
    """
    # Clean status & define outcome
    leads_df['Status'] = leads_df['Status'].astype(str).str.strip().str.lower()
    leads_df['Won'] = leads_df['Status'].isin(['definite', 'definte'])
    leads_df['Lost'] = leads_df['Status'] == 'lost'
    
    # Filter to only definitive outcomes
    df = leads_df[leads_df['Status'].isin(['definite', 'definte', 'lost'])].copy()
    df['Outcome'] = df['Won'].astype(int)
    
    # Convert numeric fields
    for col in ['Number Of Guests', 'Days Until Event', 'Days Since Inquiry', 'Bartenders Needed']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Define bins for guests
    if 'Number Of Guests' in df.columns:
        bins_guests = [0, 50, 100, 200, np.inf]
        labels_guests = ['0–50', '51–100', '101–200', '200+']
        # Convert to numeric and handle NaN values
        df['Number Of Guests'] = pd.to_numeric(df['Number Of Guests'], errors='coerce')
        df['Guests Bin'] = pd.cut(df['Number Of Guests'].fillna(-1), bins=bins_guests, labels=labels_guests)
    
    # Define bins for days until event
    if 'Days Until Event' in df.columns:
        bins_days = [0, 7, 30, 90, np.inf]
        labels_days = ['0–7 days', '8–30 days', '31–90 days', '91+ days']
        # Convert to numeric and handle NaN values
        df['Days Until Event'] = pd.to_numeric(df['Days Until Event'], errors='coerce')
        df['DaysUntilBin'] = pd.cut(df['Days Until Event'].fillna(-1), bins=bins_days, labels=labels_days)
    
    # Merge in operations data if available
    if operations_df is not None:
        try:
            # Check if 'Box Key' and 'Actual Deal Value' columns exist
            if 'Box Key' in operations_df.columns and 'Actual Deal Value' in operations_df.columns:
                # Merge on Box Key
                df = pd.merge(df, operations_df[['Box Key', 'Actual Deal Value']], 
                            on='Box Key', how='left')
                
                # Convert to numeric
                df['Actual Deal Value'] = pd.to_numeric(df['Actual Deal Value'], errors='coerce')
                
                # Calculate Price per Guest
                if 'Number Of Guests' in df.columns:
                    df['Price Per Guest'] = df['Actual Deal Value'] / df['Number Of Guests'].replace(0, np.nan)
        except Exception as e:
            print(f"Error merging operations data: {str(e)}")
    
    # Add additional advanced features
    
    # 1. Corporate Flag
    if 'Event Type' in df.columns:
        df['Is Corporate'] = df['Event Type'].str.lower().str.contains('corporate', na=False).astype(int)
    
    # 2. RFM Metrics (Using what's available)
    # Recency already captured in 'Days Since Inquiry'
    # Monetary captured in 'Actual Deal Value'
    # (Frequency would require tracking interactions, not available in the data)
    
    # 3. Event Time Analysis (if columns are available)
    if 'Service Start Time' in df.columns and 'Service End Time' in df.columns:
        try:
            # Parse times (assuming format is like "7:00 PM")
            df['Start Time'] = pd.to_datetime(df['Service Start Time'], format='%I:%M %p', errors='coerce')
            df['End Time'] = pd.to_datetime(df['Service End Time'], format='%I:%M %p', errors='coerce')
            
            # Handle overnight events (where end time is earlier than start time)
            mask = df['End Time'] < df['Start Time']
            if 'End Time' in df.columns and mask is not None:
                df.loc[mask, 'End Time'] = df.loc[mask, 'End Time'] + pd.Timedelta(days=1)
            
            # Calculate duration in hours
            if 'Start Time' in df.columns and 'End Time' in df.columns:
                df['Event Duration Hours'] = (df['End Time'] - df['Start Time']).dt.total_seconds() / 3600
        except Exception as e:
            print(f"Error processing event times: {str(e)}")
    
    # 4. Referral Quality Tiers (if column available)
    if 'Referral Source' in df.columns:
        tier_map = {
            'referral': 3,  # Highest quality
            'google': 1,
            'facebook': 2,
            'instagram': 2,
            'yelp': 1,
            'vendor': 3  # Assuming vendor referrals are high quality
        }
        
        # Create a function to map referral sources to tiers
        def map_to_tier(referral):
            if pd.isnull(referral) or not isinstance(referral, str):
                return 0
            
            referral = referral.lower()
            for key, value in tier_map.items():
                if key in referral:
                    return value
            return 0  # Default tier
        
        df['Referral Tier'] = df['Referral Source'].apply(map_to_tier)
    
    # 5. Seasonality (if event date is available)
    if 'Event Date' in df.columns:
        try:
            df['Event Date'] = pd.to_datetime(df['Event Date'], errors='coerce')
            df['Event Month'] = df['Event Date'].dt.month
            df['Event Season'] = df['Event Date'].dt.month.apply(
                lambda x: 'Winter' if x in [12, 1, 2] else
                         'Spring' if x in [3, 4, 5] else
                         'Summer' if x in [6, 7, 8] else 'Fall'
            )
        except Exception as e:
            print(f"Error processing event date: {str(e)}")
    
    return df

def calculate_conversion_rates(df):
    """
    Calculate conversion rates by different categories.
    
    Args:
        df (DataFrame): Processed dataframe with outcome
    
    Returns:
        dict: Dictionary of dataframes with conversion rates by category
    """
    conversion_rates = {}
    
    # Event Type conversion rates
    if 'Event Type' in df.columns:
        conv_event_type = df.groupby('Event Type')['Outcome'].mean().reset_index(name='Conversion Rate')
        if not conv_event_type.empty:
            conversion_rates['Event Type'] = conv_event_type
    
    # Referral Source conversion rates
    if 'Referral Source' in df.columns:
        conv_referral = df.groupby('Referral Source')['Outcome'].mean().reset_index(name='Conversion Rate')
        if not conv_referral.empty:
            conversion_rates['Referral Source'] = conv_referral
    
    # Marketing Source conversion rates
    if 'Marketing Source' in df.columns:
        conv_marketing = df.groupby('Marketing Source')['Outcome'].mean().reset_index(name='Conversion Rate')
        if not conv_marketing.empty:
            conversion_rates['Marketing Source'] = conv_marketing
    
    # State conversion rates
    if 'State' in df.columns:
        conv_state = df.groupby('State')['Outcome'].mean().reset_index(name='Conversion Rate')
        if not conv_state.empty:
            conversion_rates['State'] = conv_state
    
    # Guests Bin conversion rates
    if 'Guests Bin' in df.columns:
        conv_guests = df.groupby('Guests Bin', observed=True)['Outcome'].mean().reset_index(name='Conversion Rate')
        if not conv_guests.empty:
            conversion_rates['Guests Bin'] = conv_guests
    
    # Days Until Event Bin conversion rates
    if 'DaysUntilBin' in df.columns:
        conv_days = df.groupby('DaysUntilBin', observed=True)['Outcome'].mean().reset_index(name='Conversion Rate')
        if not conv_days.empty:
            conversion_rates['DaysUntilBin'] = conv_days
    
    return conversion_rates

def calculate_correlations(df):
    """
    Calculate correlations between numeric features and outcome.
    
    Args:
        df (DataFrame): Processed dataframe with outcome
    
    Returns:
        tuple: (
            DataFrame: Dataframe with correlations to outcome,
            DataFrame: Full correlation matrix between all features
        )
    """
    # Identify all potential numeric features - both original and new derived features
    all_numeric_features = [
        # Original features
        'Days Since Inquiry', 'Days Until Event', 'Number Of Guests', 'Bartenders Needed',
        # New derived features
        'Price Per Guest', 'Is Corporate', 'Event Duration Hours', 'Referral Tier',
        'Event Month', 'Actual Deal Value'
    ]
    
    # Filter to only include features that exist in the dataframe
    numeric_features = [col for col in all_numeric_features if col in df.columns]
    
    if numeric_features and 'Outcome' in df.columns:
        # Calculate correlations
        try:
            # Create a temporary dataframe with just the numeric columns we need
            temp_df = df[numeric_features + ['Outcome']].copy()
            
            # Convert all to numeric, handling errors
            for col in numeric_features:
                temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce')
            
            # Calculate full correlation matrix
            full_corr_matrix = temp_df.corr()
            
            # Extract and format the outcome correlations
            corr_outcome = full_corr_matrix['Outcome'].abs()
            corr_outcome = corr_outcome.sort_values(ascending=False).reset_index(name='Correlation with Outcome')
            
            return corr_outcome, full_corr_matrix
        except Exception as e:
            print(f"Error calculating correlations: {str(e)}")
            return pd.DataFrame(), pd.DataFrame()
    else:
        return pd.DataFrame(), pd.DataFrame()
