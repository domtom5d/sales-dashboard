import os
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, Table, MetaData, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

# Create a database engine
DATABASE_URL = "sqlite:///streak_data.db"
engine = create_engine(DATABASE_URL)
Base = declarative_base()
metadata = MetaData()

# Define models
class Lead(Base):
    __tablename__ = "leads"
    
    id = Column(Integer, primary_key=True)
    box_key = Column(String, unique=True, index=True)
    inquiry_date = Column(DateTime, nullable=True)
    lead_trigger = Column(String, nullable=True)
    event_date = Column(DateTime, nullable=True)
    name = Column(String, nullable=True)
    booking_type = Column(String, nullable=True)
    days_since_inquiry = Column(Integer, nullable=True)
    days_until_event = Column(Integer, nullable=True)
    city = Column(String, nullable=True)
    state = Column(String, nullable=True)
    bartenders_needed = Column(Integer, nullable=True)
    number_of_guests = Column(Integer, nullable=True)
    total_serve_time = Column(Float, nullable=True)
    total_bartender_time = Column(Float, nullable=True)
    marketing_source = Column(String, nullable=True)
    referral_source = Column(String, nullable=True)
    status = Column(String, nullable=True)
    first_name = Column(String, nullable=True)
    last_name = Column(String, nullable=True)
    email = Column(String, nullable=True)
    phone_number = Column(String, nullable=True)
    won = Column(Boolean, default=False)
    lost = Column(Boolean, default=False)
    outcome = Column(Integer, default=0)  # 1 = won, 0 = lost
    guests_bin = Column(String, nullable=True)
    days_until_bin = Column(String, nullable=True)
    
    def __repr__(self):
        return f"<Lead(box_key='{self.box_key}', name='{self.name}', outcome={self.outcome})>"


class Operation(Base):
    __tablename__ = "operations"
    
    id = Column(Integer, primary_key=True)
    box_key = Column(String, unique=True, index=True)
    event_date = Column(DateTime, nullable=True)
    actual_deal_value = Column(Float, nullable=True)
    region = Column(String, nullable=True)
    name = Column(String, nullable=True)
    days_until_event = Column(Integer, nullable=True)
    status = Column(String, nullable=True)
    booking_type = Column(String, nullable=True)
    event_type = Column(String, nullable=True)
    city = Column(String, nullable=True)
    state = Column(String, nullable=True)
    first_name = Column(String, nullable=True)
    last_name = Column(String, nullable=True)
    email = Column(String, nullable=True)
    phone_number = Column(String, nullable=True)
    
    def __repr__(self):
        return f"<Operation(box_key='{self.box_key}', name='{self.name}', actual_deal_value={self.actual_deal_value})>"


# Create tables
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

def migrate_database():
    """
    Update database schema to include new columns
    """
    try:
        # Connect to database
        conn = sqlite3.connect('streak_data.db')
        cursor = conn.cursor()
        
        # Check if the columns already exist in leads table
        cursor.execute("PRAGMA table_info(leads)")
        leads_columns = [column[1] for column in cursor.fetchall()]
        
        # Add new columns to leads table if they don't exist
        if 'first_name' not in leads_columns:
            cursor.execute("ALTER TABLE leads ADD COLUMN first_name TEXT")
        if 'last_name' not in leads_columns:
            cursor.execute("ALTER TABLE leads ADD COLUMN last_name TEXT")
        if 'email' not in leads_columns:
            cursor.execute("ALTER TABLE leads ADD COLUMN email TEXT")
        if 'phone_number' not in leads_columns:
            cursor.execute("ALTER TABLE leads ADD COLUMN phone_number TEXT")
            
        # Check if the columns already exist in operations table
        cursor.execute("PRAGMA table_info(operations)")
        operations_columns = [column[1] for column in cursor.fetchall()]
        
        # Add new columns to operations table if they don't exist
        if 'first_name' not in operations_columns:
            cursor.execute("ALTER TABLE operations ADD COLUMN first_name TEXT")
        if 'last_name' not in operations_columns:
            cursor.execute("ALTER TABLE operations ADD COLUMN last_name TEXT")
        if 'email' not in operations_columns:
            cursor.execute("ALTER TABLE operations ADD COLUMN email TEXT")
        if 'phone_number' not in operations_columns:
            cursor.execute("ALTER TABLE operations ADD COLUMN phone_number TEXT")
            
        # Commit changes
        conn.commit()
        print("Database migration successful")
        return True
    except Exception as e:
        print(f"Error migrating database: {e}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()

def import_leads_data(csv_path):
    """
    Import leads data from a CSV file into the database
    
    Args:
        csv_path (str): Path to the CSV file
        
    Returns:
        int: Number of records imported
    """
    session = None
    try:
        # Create backup directory
        os.makedirs("data/backups", exist_ok=True)
        
        # Create a backup of the database before import
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_csv = f"data/backups/leads_backup_{timestamp}.csv"
        
        # Backup existing data
        session = Session()
        existing_leads_count = session.query(Lead).count()
        if existing_leads_count > 0:
            existing_leads = pd.read_sql(session.query(Lead).statement, engine)
            existing_leads.to_csv(backup_csv, index=False)
            print(f"Created backup with {existing_leads_count} leads at {backup_csv}")
        session.close()
        
        # Read the CSV file with more permissive settings
        print(f"Importing leads from {csv_path}...")
        df = pd.read_csv(csv_path, low_memory=False, on_bad_lines='skip', encoding='utf-8')
        print(f"Found {len(df)} rows in CSV file")
        
        # Save a copy of the source data for reference
        source_copy = f"data/backups/source_leads_{timestamp}.csv"
        df.to_csv(source_copy, index=False)
        
        # Normalize column names to handle case sensitivity
        column_map = {}
        for col in df.columns:
            col_lower = col.lower()
            if col_lower == 'box key':
                column_map[col] = 'Box Key'
            elif col_lower == 'name':
                column_map[col] = 'Name'
            elif col_lower == 'inquiry date':
                column_map[col] = 'Inquiry Date'
            elif col_lower == 'event date':
                column_map[col] = 'Event Date'
            elif col_lower == 'lead trigger':
                column_map[col] = 'Lead Trigger'
            elif col_lower == 'days since inquiry':
                column_map[col] = 'Days Since Inquiry'
            elif col_lower == 'days until event':
                column_map[col] = 'Days Until Event'
            elif col_lower == 'city':
                column_map[col] = 'City'
            elif col_lower == 'state':
                column_map[col] = 'State'
            elif col_lower == 'bartenders needed':
                column_map[col] = 'Bartenders Needed'
            elif col_lower == 'number of guests':
                column_map[col] = 'Number Of Guests'
            elif col_lower == 'first name':
                column_map[col] = 'First Name'
            elif col_lower == 'last name':
                column_map[col] = 'Last Name'
            elif col_lower == 'email address' or col_lower == 'email':
                column_map[col] = 'Email Address'
            elif col_lower == 'phone number' or col_lower == 'phone #':
                column_map[col] = 'Phone Number'
            elif col_lower == 'total serve time':
                column_map[col] = 'Total Serve Time'
            elif col_lower == 'total bartender time':
                column_map[col] = 'Total Bartender Time'
            elif col_lower == 'marketing source':
                column_map[col] = 'Marketing Source'
            elif col_lower == 'referral source':
                column_map[col] = 'Referral Source'
            elif col_lower == 'status':
                column_map[col] = 'Status'
            elif col_lower == 'booking type':
                column_map[col] = 'Booking Type'
        
        # Rename columns
        if column_map:
            df = df.rename(columns=column_map)
            print(f"Normalized {len(column_map)} column names")
        
        # Print column names to help debug
        print(f"Available columns: {', '.join(df.columns)}")
        
        # Process data in batches
        session = Session()
        batch_size = 100
        total_records = 0
        
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            batch_records = 0
            
            for _, row in batch.iterrows():
                try:
                    # Extract Box Key from Name field if available
                    box_key = None
                    
                    # Check for Box Key column first
                    if 'Box Key' in row and pd.notna(row['Box Key']):
                        box_key = str(row['Box Key']).strip()
                    
                    # If not found, look for BK- pattern in Name
                    if not box_key and 'Name' in row and pd.notna(row['Name']):
                        name_str = str(row['Name'])
                        if 'BK-' in name_str:
                            parts = name_str.split(' ')
                            for part in parts:
                                if part.startswith('BK-'):
                                    box_key = part.strip()
                                    break
                    
                    # If still no box key, generate one based on row index and timestamp
                    if not box_key:
                        row_idx = total_records + batch_records
                        box_key = f"GEN-{timestamp}-{row_idx}"
                        print(f"Generated box_key: {box_key} for row without BK")
                    
                    # Process date fields
                    inquiry_date = None
                    if 'Inquiry Date' in row and pd.notna(row['Inquiry Date']):
                        try:
                            inquiry_date = pd.to_datetime(row['Inquiry Date'], errors='coerce')
                        except Exception as e:
                            print(f"Error parsing inquiry date: {e}")
                    
                    event_date = None
                    if 'Event Date' in row and pd.notna(row['Event Date']):
                        try:
                            event_date = pd.to_datetime(row['Event Date'], errors='coerce')
                        except Exception as e:
                            print(f"Error parsing event date: {e}")
                    
                    # Process numeric fields
                    days_since_inquiry = pd.to_numeric(row.get('Days Since Inquiry', None), errors='coerce')
                    days_until_event = pd.to_numeric(row.get('Days Until Event', None), errors='coerce')
                    bartenders_needed = pd.to_numeric(row.get('Bartenders Needed', None), errors='coerce')
                    number_of_guests = pd.to_numeric(row.get('Number Of Guests', None), errors='coerce')
                    total_serve_time = pd.to_numeric(row.get('Total Serve Time', None), errors='coerce')
                    total_bartender_time = pd.to_numeric(row.get('Total Bartender Time', None), errors='coerce')
                    
                    # Process status and outcome according to our updated logic
                    status = None
                    won = False
                    lost = False
                    outcome = 0
                    
                    if 'Status' in row and pd.notna(row['Status']):
                        status = str(row['Status']).strip()
                        status_lower = status.lower()
                        won = status_lower in ['definite', 'definte', 'tentative']
                        lost = status_lower == 'lost'
                        outcome = 1 if won else 0 if lost else None
                    
                    # Also check Lead Trigger if Status doesn't give a clear outcome
                    if outcome is None and 'Lead Trigger' in row and pd.notna(row['Lead Trigger']):
                        lead_trigger = str(row['Lead Trigger']).strip().lower()
                        if lead_trigger in ['hot', 'warm', 'super lead']:
                            won = True
                            lost = False
                            outcome = 1
                        elif lead_trigger in ['cool', 'cold']:
                            won = False
                            lost = True
                            outcome = 0
                    
                    # Create lead dict with all available fields
                    lead_data = {
                        'box_key': box_key,
                        'inquiry_date': inquiry_date,
                        'event_date': event_date,
                        'name': row.get('Name'),
                        'status': status,
                        'won': won,
                        'lost': lost,
                        'outcome': outcome if outcome is not None else 0
                    }
                    
                    # Add all other fields if they exist in the row
                    optional_fields = {
                        'Lead Trigger': 'lead_trigger',
                        'Booking Type': 'booking_type',
                        'Days Since Inquiry': 'days_since_inquiry',
                        'Days Until Event': 'days_until_event',
                        'City': 'city',
                        'State': 'state',
                        'Bartenders Needed': 'bartenders_needed',
                        'Number Of Guests': 'number_of_guests',
                        'Total Serve Time': 'total_serve_time',
                        'Total Bartender Time': 'total_bartender_time',
                        'Marketing Source': 'marketing_source',
                        'Referral Source': 'referral_source'
                    }
                    
                    for csv_col, db_col in optional_fields.items():
                        if csv_col in row and pd.notna(row[csv_col]):
                            lead_data[db_col] = row[csv_col]
                    
                    # Create Lead object
                    lead = Lead(**lead_data)
                    
                    # Check if record already exists
                    existing = session.query(Lead).filter(Lead.box_key == box_key).first()
                    if existing:
                        # Update existing record attributes
                        for key, value in lead_data.items():
                            if value is not None and hasattr(existing, key):
                                setattr(existing, key, value)
                    else:
                        # Add new record
                        session.add(lead)
                    
                    batch_records += 1
                    
                    # Commit every 10 records to avoid transaction bloat
                    if batch_records % 10 == 0:
                        session.commit()
                        
                except Exception as e:
                    print(f"Error processing row: {e}")
                    # Continue processing (don't stop the entire import for one bad row)
                    continue
            
            # Commit batch
            try:
                session.commit()
                total_records += batch_records
                print(f"Imported batch {i//batch_size + 1}: {batch_records} records (total: {total_records})")
            except Exception as e:
                print(f"Error committing batch: {e}")
                session.rollback()
        
        # Close session
        if session:
            session.close()
        
        print(f"Successfully imported {total_records} lead records")
        return total_records
    except Exception as e:
        print(f"Error in import_leads_data: {e}")
        if session:
            try:
                session.rollback()
                session.close()
            except:
                pass
        return 0


def import_operations_data(csv_path):
    """
    Import operations data from a CSV file into the database
    
    Args:
        csv_path (str): Path to the CSV file
        
    Returns:
        int: Number of records imported
    """
    session = None
    try:
        # Create backup directory
        os.makedirs("data/backups", exist_ok=True)
        
        # Create a backup of the database before import
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_csv = f"data/backups/operations_backup_{timestamp}.csv"
        
        # Backup existing data
        session = Session()
        existing_ops_count = session.query(Operation).count()
        if existing_ops_count > 0:
            existing_ops = pd.read_sql(session.query(Operation).statement, engine)
            existing_ops.to_csv(backup_csv, index=False)
            print(f"Created backup with {existing_ops_count} operations at {backup_csv}")
        session.close()
        
        # Read the CSV file with more permissive settings
        print(f"Importing operations from {csv_path}...")
        df = pd.read_csv(csv_path, low_memory=False, on_bad_lines='skip', encoding='utf-8')
        print(f"Found {len(df)} rows in CSV file")
        
        # Save a copy of the source data for reference
        source_copy = f"data/backups/source_operations_{timestamp}.csv"
        df.to_csv(source_copy, index=False)
        
        # Normalize column names to handle case sensitivity
        column_map = {}
        for col in df.columns:
            col_lower = col.lower()
            if col_lower == 'box key':
                column_map[col] = 'Box Key'
            elif col_lower == 'name':
                column_map[col] = 'Name'
            elif col_lower == 'event date':
                column_map[col] = 'Event Date'
            elif col_lower == 'actual deal value':
                column_map[col] = 'Actual Deal Value'
            elif col_lower == 'region':
                column_map[col] = 'Region'
            elif col_lower == 'days until event':
                column_map[col] = 'Days Until Event'
            elif col_lower == 'status':
                column_map[col] = 'Status'
            elif col_lower == 'booking type':
                column_map[col] = 'Booking Type'
            elif col_lower == 'event type':
                column_map[col] = 'Event Type'
            elif col_lower == 'city':
                column_map[col] = 'City'
            elif col_lower == 'state':
                column_map[col] = 'State'
        
        # Rename columns
        if column_map:
            df = df.rename(columns=column_map)
            print(f"Normalized {len(column_map)} column names")
        
        # Print column names to help debug
        print(f"Available columns: {', '.join(df.columns)}")
        
        # Process data in batches
        session = Session()
        batch_size = 100
        total_records = 0
        
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            batch_records = 0
            
            for _, row in batch.iterrows():
                try:
                    # Extract Box Key from Name field if available
                    box_key = None
                    
                    # Check for Box Key column first
                    if 'Box Key' in row and pd.notna(row['Box Key']):
                        box_key = str(row['Box Key']).strip()
                    
                    # If not found, look for BK- pattern in Name
                    if not box_key and 'Name' in row and pd.notna(row['Name']):
                        name_str = str(row['Name'])
                        if 'BK-' in name_str:
                            parts = name_str.split(' ')
                            for part in parts:
                                if part.startswith('BK-'):
                                    box_key = part.strip()
                                    break
                    
                    # If still no box key, generate one based on row index and timestamp
                    if not box_key:
                        row_idx = total_records + batch_records
                        box_key = f"GEN-OP-{timestamp}-{row_idx}"
                        print(f"Generated box_key: {box_key} for operation row without BK")
                    
                    # Process date fields
                    event_date = None
                    if 'Event Date' in row and pd.notna(row['Event Date']):
                        try:
                            event_date = pd.to_datetime(row['Event Date'], errors='coerce')
                        except Exception as e:
                            print(f"Error parsing event date: {e}")
                    
                    # Process numeric fields with more robust handling
                    actual_deal_value = None
                    if 'Actual Deal Value' in row and pd.notna(row['Actual Deal Value']):
                        # Handle currency symbols, commas, etc.
                        value_str = str(row['Actual Deal Value']).replace('$', '').replace(',', '')
                        try:
                            actual_deal_value = pd.to_numeric(value_str, errors='coerce')
                        except:
                            pass
                    
                    days_until_event = pd.to_numeric(row.get('Days Until Event', None), errors='coerce')
                    
                    # Create operation dict with all available fields
                    op_data = {
                        'box_key': box_key,
                        'event_date': event_date,
                        'actual_deal_value': actual_deal_value,
                        'days_until_event': days_until_event
                    }
                    
                    # Add all other fields if they exist in the row
                    optional_fields = {
                        'Region': 'region',
                        'Name': 'name',
                        'Status': 'status',
                        'Booking Type': 'booking_type',
                        'Event Type': 'event_type',
                        'First Name': 'first_name',
                        'Last Name': 'last_name',
                        'Email': 'email',
                        'Phone #': 'phone_number',
                        'City': 'city',
                        'State': 'state'
                    }
                    
                    for csv_col, db_col in optional_fields.items():
                        if csv_col in row and pd.notna(row[csv_col]):
                            op_data[db_col] = row[csv_col]
                    
                    # Try to detect a duplicate before creating a new operation
                    existing = session.query(Operation).filter(Operation.box_key == box_key).first()
                    
                    if existing:
                        # Update existing record with new data
                        for key, value in op_data.items():
                            if value is not None and hasattr(existing, key):
                                setattr(existing, key, value)
                    else:
                        # Create new Operation object
                        operation = Operation(**op_data)
                        session.add(operation)
                    
                    batch_records += 1
                    
                    # Commit every 10 records to avoid transaction bloat
                    if batch_records % 10 == 0:
                        try:
                            session.commit()
                        except Exception as e:
                            print(f"Error committing records: {e}")
                            session.rollback()
                            # Continue with next batch instead of aborting
                            continue
                    
                except Exception as e:
                    print(f"Error processing operation row: {e}")
                    # Continue processing (don't stop the entire import for one bad row)
                    continue
            
            # Commit batch
            try:
                session.commit()
                total_records += batch_records
                print(f"Imported operations batch {i//batch_size + 1}: {batch_records} records (total: {total_records})")
            except Exception as e:
                print(f"Error committing operations batch: {e}")
                session.rollback()
        
        # Close session
        if session:
            session.close()
        
        print(f"Successfully imported {total_records} operation records")
        return total_records
    except Exception as e:
        print(f"Error in import_operations_data: {e}")
        if session:
            try:
                session.rollback()
                session.close()
            except:
                pass
        return 0


def get_lead_data(filters=None):
    """
    Get lead data from the database with optional filters
    
    Args:
        filters (dict, optional): Dictionary of filter conditions
        
    Returns:
        pandas.DataFrame: DataFrame containing the lead data
    """
    try:
        # Create a session
        session = Session()
        
        # Query leads
        query = session.query(Lead)
        
        # Apply filters if provided
        if filters:
            for key, value in filters.items():
                if value != 'All' and value is not None:
                    query = query.filter(getattr(Lead, key) == value)
        
        # Execute query and convert to DataFrame
        leads = pd.read_sql(query.statement, session.bind)
        
        session.close()
        
        return leads
    except Exception as e:
        print(f"Error getting lead data: {e}")
        return pd.DataFrame()


def get_operation_data(filters=None):
    """
    Get operation data from the database with optional filters
    
    Args:
        filters (dict, optional): Dictionary of filter conditions
        
    Returns:
        pandas.DataFrame: DataFrame containing the operation data
    """
    try:
        # Create a session
        session = Session()
        
        # Query operations
        query = session.query(Operation)
        
        # Apply filters if provided
        if filters:
            for key, value in filters.items():
                if value != 'All' and value is not None:
                    query = query.filter(getattr(Operation, key) == value)
        
        # Execute query and convert to DataFrame
        operations = pd.read_sql(query.statement, session.bind)
        
        session.close()
        
        return operations
    except Exception as e:
        print(f"Error getting operation data: {e}")
        return pd.DataFrame()


def get_merged_data(lead_filters=None, operation_filters=None):
    """
    Get merged lead and operation data with optional filters
    
    Args:
        lead_filters (dict, optional): Dictionary of lead filter conditions
        operation_filters (dict, optional): Dictionary of operation filter conditions
        
    Returns:
        pandas.DataFrame: DataFrame containing the merged data
    """
    try:
        # Get lead and operation data
        leads = get_lead_data(lead_filters)
        operations = get_operation_data(operation_filters)
        
        # Merge data
        if not leads.empty and not operations.empty:
            merged = pd.merge(leads, operations, on='box_key', how='left', suffixes=('', '_operation'))
            return merged
        else:
            return leads
    except Exception as e:
        print(f"Error getting merged data: {e}")
        return pd.DataFrame()


def import_sample_data():
    """
    Import sample data from the data directory
    
    Returns:
        tuple: (leads_count, operations_count)
    """
    leads_count = import_leads_data('data/leads.csv')
    operations_count = import_operations_data('data/operations.csv')
    return (leads_count, operations_count)


# Initialize database with sample data if tables are empty
def initialize_db_if_empty():
    session = Session()
    lead_count = session.query(Lead).count()
    operation_count = session.query(Operation).count()
    session.close()
    
    if lead_count == 0 and operation_count == 0:
        leads_count, operations_count = import_sample_data()
        print(f"Initialized database with {leads_count} leads and {operations_count} operations")
    
    return lead_count > 0 or operation_count > 0


# Clean and normalize phone numbers
def clean_phone(phone):
    if not phone or pd.isna(phone) or phone == "#ERROR!":
        return None
    # Remove non-numeric characters
    cleaned = ''.join(filter(str.isdigit, str(phone)))
    # Keep only the last 10 digits if longer
    if len(cleaned) > 10:
        cleaned = cleaned[-10:]
    return cleaned if cleaned else None


def process_phone_matching():
    """
    Match lead inquiries with bookings based on phone numbers
    and update contact information.
    
    Returns:
        tuple: (matched_count, total_leads, total_operations)
    """
    try:
        # Migrate database first to ensure we have the needed columns
        migrate_database()
        
        session = Session()
        
        # Get all leads and operations
        leads = session.query(Lead).all()
        operations = session.query(Operation).all()
        
        total_leads = len(leads)
        total_operations = len(operations)
        matches = 0
        
        # Create dictionaries for faster lookups
        operation_by_phone = {}
        operation_by_email = {}
        operation_by_box_key = {}
        
        # We're now using the global clean_phone function
        
        # First pass: Build lookup dictionaries from operations
        for op in operations:
            # Add to box_key dictionary
            if op.box_key:
                operation_by_box_key[op.box_key] = op
                
            # Add to email dictionary if email exists
            if op.email and not pd.isna(op.email):
                email = op.email.lower().strip()
                operation_by_email[email] = op
                
            # Add to phone dictionary if phone exists
            if op.phone_number and not pd.isna(op.phone_number) and op.phone_number != "#ERROR!":
                clean_num = clean_phone(op.phone_number)
                if clean_num:
                    operation_by_phone[clean_num] = op
        
        # Second pass: Match leads to operations
        for lead in leads:
            matched = False
            
            # First try to match by box_key (direct match)
            if lead.box_key and lead.box_key in operation_by_box_key:
                matched = True
                matches += 1
                
            # Then try to match by email
            elif lead.email and not pd.isna(lead.email):
                email = lead.email.lower().strip()
                if email in operation_by_email:
                    matched = True
                    matches += 1
                    
                    # If we matched on email, update the operation with the lead's box_key
                    operation = operation_by_email[email]
                    if operation.box_key != lead.box_key:
                        print(f"Matched lead {lead.box_key} to operation {operation.box_key} via email")
                    
            # Finally try to match by phone
            elif lead.phone_number and not pd.isna(lead.phone_number) and lead.phone_number != "#ERROR!":
                clean_num = clean_phone(lead.phone_number)
                if clean_num and clean_num in operation_by_phone:
                    matched = True
                    matches += 1
                    
                    # If we matched on phone, update the operation with the lead's box_key
                    operation = operation_by_phone[clean_num]
                    if operation.box_key != lead.box_key:
                        print(f"Matched lead {lead.box_key} to operation {operation.box_key} via phone")
        
        session.commit()
        print(f"Found {matches} matches out of {total_leads} leads and {total_operations} operations")
        return matches, total_leads, total_operations
    
    except Exception as e:
        print(f"Error processing phone matching: {e}")
        import traceback
        traceback.print_exc()
        return 0, 0, 0
    finally:
        if 'session' in locals():
            session.close()


# Run initialization if this script is run directly
if __name__ == "__main__":
    initialize_db_if_empty()
    # Run phone matching
    process_phone_matching()