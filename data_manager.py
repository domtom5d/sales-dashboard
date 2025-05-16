"""
data_manager.py - Centralized data loading and processing

This module provides a single source of truth for data loading,
normalization, and preprocessing throughout the application.
"""

import pandas as pd
import numpy as np
import streamlit as st
import datetime
from database import get_lead_data, get_operation_data, import_leads_data, import_operations_data

@st.cache_data(ttl=3600)  # Cache data for 1 hour
def load_data(use_csv=False, leads_file=None, operations_file=None):
    """
    Load data from either CSV uploads or database
    
    Args:
        use_csv (bool): Whether to use CSV files (True) or database (False)
        leads_file (UploadedFile, optional): Uploaded leads CSV file
        operations_file (UploadedFile, optional): Uploaded operations CSV file
        
    Returns:
        tuple: (leads_df, operations_df, processed_df)
    """
    leads_df = None
    operations_df = None
    
    if use_csv and leads_file is not None:
        # Load from uploaded CSV files
        try:
            leads_df = pd.read_csv(leads_file)
            st.success(f"Successfully loaded {len(leads_df)} lead records")
        except Exception as e:
            st.error(f"Error loading leads file: {str(e)}")
    
        if operations_file is not None:
            try:
                operations_df = pd.read_csv(operations_file)
                st.success(f"Successfully loaded {len(operations_df)} operation records")
            except Exception as e:
                st.error(f"Error loading operations file: {str(e)}")
    else:
        # Load from database
        try:
            leads_df = get_lead_data()
            if leads_df is not None and not leads_df.empty:
                st.success(f"Successfully loaded {len(leads_df)} lead records from database")
            else:
                st.warning("No lead data found in database")
        except Exception as e:
            st.error(f"Error loading leads from database: {str(e)}")
        
        try:
            operations_df = get_operation_data()
            if operations_df is not None and not operations_df.empty:
                st.success(f"Successfully loaded {len(operations_df)} operation records from database")
        except Exception as e:
            st.error(f"Error loading operations from database: {str(e)}")
    
    # Process the data if leads are available
    processed_df = None
    if leads_df is not None and not leads_df.empty:
        processed_df = process_data(leads_df, operations_df)
    
    return leads_df, operations_df, processed_df

def normalize_column_names(df):
    """
    Normalize column names to lowercase with underscores
    
    Args:
        df (DataFrame): DataFrame with original column names
        
    Returns:
        DataFrame: DataFrame with normalized column names
    """
    if df is None:
        return None
        
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Convert all column names to lowercase with underscores
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    
    return df

def standardize_date_columns(df):
    """
    Convert date columns to datetime
    
    Args:
        df (DataFrame): DataFrame with date columns
        
    Returns:
        DataFrame: DataFrame with standardized date columns
    """
    if df is None:
        return None
        
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # List of potential date columns
    date_cols = ['inquiry_date', 'event_date', 'created', 'modified']
    
    # Convert each date column to datetime
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    return df

def clean_booking_type(booking_type):
    """
    Clean and standardize booking type strings
    
    Args:
        booking_type (str): Original booking type string
        
    Returns:
        str: Cleaned booking type
    """
    import re
    
    if pd.isna(booking_type) or booking_type is None:
        return "Unknown"
    
    # Convert to string
    booking_type = str(booking_type)
    
    # Convert to lowercase, strip trailing years, replace underscores with spaces
    cleaned = booking_type.lower()
    cleaned = re.sub(r'\d{4}$', '', cleaned)  # Remove trailing years (e.g., 2025)
    cleaned = cleaned.replace('_', ' ').strip()
    
    # Group similar types
    if 'wedding' in cleaned:
        return 'Wedding'
    elif 'corporate' in cleaned:
        return 'Corporate Event'
    elif 'birthday' in cleaned or 'bday' in cleaned:
        return 'Birthday'
    elif 'graduation' in cleaned or 'grad' in cleaned:
        return 'Graduation'
    elif 'holiday' in cleaned or 'christmas' in cleaned or 'halloween' in cleaned:
        return 'Holiday Party'
    elif 'anniversary' in cleaned:
        return 'Anniversary'
    elif 'fundraiser' in cleaned or 'charity' in cleaned:
        return 'Fundraiser'
    elif 'rehearsal' in cleaned:
        return 'Rehearsal Dinner'
    elif 'engagement' in cleaned:
        return 'Engagement'
    elif 'private' in cleaned:
        return 'Private Party'
    
    # Title case for display
    return cleaned.title()

def process_data(leads_df, operations_df=None):
    """
    Process and prepare the data for analysis
    
    Args:
        leads_df (DataFrame): The leads data from Streak export
        operations_df (DataFrame, optional): The operations data from Streak export
    
    Returns:
        DataFrame: Processed dataframe with outcome and binned columns
    """
    if leads_df is None or leads_df.empty:
        return None
    
    # First normalize column names
    df = normalize_column_names(leads_df)
    
    # Convert date columns
    df = standardize_date_columns(df)
    
    # Ensure required columns exist
    if 'status' not in df.columns:
        df['status'] = 'unknown'  # Default value if status column is missing
    
    # Use the Status column as the primary indicator of won/lost
    if 'status' in df.columns:
        # Normalize status values
        status = df['status'].astype(str).str.strip().str.lower()
        
        # Define win/loss sets based on your actual data
        wins = {'definite', 'tentative', 'won'}  # Including Tentative as wins
        losses = {'lost'}
        
        # Create derived columns
        df['won'] = status.isin(wins)
        df['lost'] = status.isin(losses)
        
        # Create numeric outcome (1 = won, 0 = lost)
        df['outcome'] = status.map(lambda s: 1 if s in wins else 0 if s in losses else np.nan)
        
        # Filter to leads with clear outcomes (won or lost)
        df = df.dropna(subset=['outcome']).copy()
        
        # Convert outcome to integer
        df['outcome'] = df['outcome'].astype(int)
    
    # If Status is missing or we have no definitive outcomes after filtering, 
    # fallback to Lead Trigger as a supplementary signal
    elif 'lead_trigger' in df.columns:
        # Map Lead Trigger statuses to a won/lost flag
        df['lead_trigger'] = df['lead_trigger'].astype(str)
        
        # Temperature-based model:
        # Hot, Warm, Super Lead = likely to convert (treat as won)
        # Cool, Cold = less likely to convert (treat as lost)
        df['won'] = df['lead_trigger'].str.lower().isin(['hot', 'warm', 'super lead'])
        df['lost'] = df['lead_trigger'].str.lower().isin(['cold', 'cool'])
        
        # Filter to leads that have a clear status
        df = df[df['lead_trigger'].str.lower().isin(['hot', 'warm', 'cool', 'cold', 'super lead'])].copy()
        
        # Set outcome (1 = won, 0 = lost)
        df['outcome'] = df['won'].astype(int)
    else:
        # Fallback if neither Status nor Lead Trigger are available
        df['won'] = False
        df['lost'] = False
        df['outcome'] = 0
    
    # Convert numeric fields
    for col in ['number_of_guests', 'days_until_event', 'days_since_inquiry', 'bartenders_needed']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Add clean booking type
    if 'booking_type' in df.columns:
        df['clean_booking_type'] = df['booking_type'].apply(clean_booking_type)
    
    # Define bins for guests
    if 'number_of_guests' in df.columns:
        bins_guests = [0, 50, 100, 200, np.inf]
        labels_guests = ['0–50', '51–100', '101–200', '200+']
        # Convert to numeric and handle NaN values
        df['guests_bin'] = pd.cut(df['number_of_guests'].fillna(-1), bins=bins_guests, labels=labels_guests)
    
    # Define bins for days until event
    if 'days_until_event' in df.columns:
        bins_days = [0, 7, 30, 90, np.inf]
        labels_days = ['0–7 days', '8–30 days', '31–90 days', '91+ days']
        # Convert to numeric and handle NaN values
        df['days_until_bin'] = pd.cut(df['days_until_event'].fillna(-1), bins=bins_days, labels=labels_days)
    
    # Derive weekday from inquiry date
    if 'inquiry_date' in df.columns:
        try:
            # Convert to datetime and get weekday
            df['weekday'] = df['inquiry_date'].dt.day_name()
        except:
            pass
    
    # Merge in operations data if available
    if operations_df is not None and not operations_df.empty:
        ops_df = normalize_column_names(operations_df)
        ops_df = standardize_date_columns(ops_df)
        
        try:
            # Check if box_key column exists in both dataframes
            if 'box_key' in df.columns and 'box_key' in ops_df.columns:
                # Merge on Box Key
                merge_cols = ['box_key', 'actual_deal_value']
                if 'region' in ops_df.columns:
                    merge_cols.append('region')
                
                df = pd.merge(df, ops_df[merge_cols], 
                          on='box_key', how='left')
                
                # Convert to numeric
                if 'actual_deal_value' in df.columns:
                    df['actual_deal_value'] = pd.to_numeric(df['actual_deal_value'], errors='coerce')
                    
                    # Calculate Price per Guest
                    if 'number_of_guests' in df.columns:
                        df['price_per_guest'] = df['actual_deal_value'] / df['number_of_guests'].replace(0, np.nan)
        except Exception as e:
            print(f"Error merging operations data: {str(e)}")
    
    # Add additional advanced features
    
    # 1. Corporate Flag
    if 'event_type' in df.columns:
        df['is_corporate'] = df['event_type'].str.lower().str.contains('corporate', na=False).astype(int)
    
    # 2. Seasonality (if event date is available)
    if 'event_date' in df.columns:
        try:
            df['event_month'] = df['event_date'].dt.month
            df['event_season'] = df['event_date'].dt.month.apply(
                lambda x: 'Winter' if x in [12, 1, 2] else
                        'Spring' if x in [3, 4, 5] else
                        'Summer' if x in [6, 7, 8] else 'Fall'
            )
        except Exception as e:
            print(f"Error processing event date: {str(e)}")
    
    return df

def apply_filters(df, filters=None):
    """
    Apply filters to the dataframe
    
    Args:
        df (DataFrame): Dataframe to filter
        filters (dict, optional): Dictionary of filter settings
            - date_range: Tuple of (start_date, end_date)
            - status: 'All', 'Won', or 'Lost'
            - states: List of state values or ['All']
            - date_col: Column to use for date filtering
    
    Returns:
        DataFrame: Filtered dataframe
    """
    if df is None or df.empty:
        return df
    
    # Create a copy to avoid modifying the original
    filtered_df = df.copy()
    
    # If no filters are provided, return the original dataframe
    if filters is None or len(filters) == 0:
        return filtered_df
    
    # Apply date filter if provided
    if 'date_range' in filters and filters['date_range'] is not None:
        date_col = filters.get('date_col', 'inquiry_date')
        start_date, end_date = filters['date_range']
        
        if date_col in filtered_df.columns and start_date and end_date:
            mask = (filtered_df[date_col] >= pd.to_datetime(start_date)) & \
                  (filtered_df[date_col] <= pd.to_datetime(end_date))
            filtered_df = filtered_df[mask]
    
    # Apply status filter if provided
    if 'status' in filters and filters['status'] != 'All':
        if filters['status'] == 'Won' and 'won' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['won'] == True]
        elif filters['status'] == 'Lost' and 'lost' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['lost'] == True]
    
    # Apply state filter if provided
    if 'states' in filters and 'All' not in filters['states'] and 'state' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['state'].isin(filters['states'])]
    
    return filtered_df