import os
import pandas as pd
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
    
    def __repr__(self):
        return f"<Operation(box_key='{self.box_key}', name='{self.name}', actual_deal_value={self.actual_deal_value})>"


# Create tables
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

def import_leads_data(csv_path):
    """
    Import leads data from a CSV file into the database
    
    Args:
        csv_path (str): Path to the CSV file
        
    Returns:
        int: Number of records imported
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path, low_memory=False)
        
        # Convert data
        records = []
        for _, row in df.iterrows():
            try:
                # Extract Box Key from Name field if available
                box_key = None
                if 'Name' in row and pd.notna(row['Name']):
                    parts = str(row['Name']).split(' ')
                    for part in parts:
                        if part.startswith('BK-'):
                            box_key = part
                            break
                
                # If Box Key not found, use from Box Key column if available
                if not box_key and 'Box Key' in row and pd.notna(row['Box Key']):
                    box_key = row['Box Key']
                
                # Ensure we have a box key
                if not box_key:
                    continue
                
                # Process date fields
                inquiry_date = None
                if 'Inquiry Date' in row and pd.notna(row['Inquiry Date']):
                    try:
                        inquiry_date = pd.to_datetime(row['Inquiry Date'])
                    except:
                        pass
                
                event_date = None
                if 'Event Date' in row and pd.notna(row['Event Date']):
                    try:
                        event_date = pd.to_datetime(row['Event Date'])
                    except:
                        pass
                
                # Process numeric fields
                days_since_inquiry = pd.to_numeric(row.get('Days Since Inquiry', None), errors='coerce')
                days_until_event = pd.to_numeric(row.get('Days Until Event', None), errors='coerce')
                bartenders_needed = pd.to_numeric(row.get('Bartenders Needed', None), errors='coerce')
                number_of_guests = pd.to_numeric(row.get('Number Of Guests', None), errors='coerce')
                total_serve_time = pd.to_numeric(row.get('Total Serve Time', None), errors='coerce')
                total_bartender_time = pd.to_numeric(row.get('Total Bartender Time', None), errors='coerce')
                
                # Process status and outcome
                status = None
                won = False
                lost = False
                outcome = 0
                
                if 'Status' in row and pd.notna(row['Status']):
                    status = str(row['Status']).strip().lower()
                    won = status in ['definite', 'definte'] 
                    lost = status == 'lost'
                    outcome = 1 if won else 0 if lost else None
                
                # Create a new Lead record
                record = Lead(
                    box_key=box_key,
                    inquiry_date=inquiry_date,
                    lead_trigger=row.get('Lead Trigger', None),
                    event_date=event_date,
                    name=row.get('Name', None),
                    booking_type=row.get('Booking Type', None),
                    days_since_inquiry=days_since_inquiry,
                    days_until_event=days_until_event,
                    city=row.get('City', None),
                    state=row.get('State', None),
                    bartenders_needed=bartenders_needed,
                    number_of_guests=number_of_guests,
                    total_serve_time=total_serve_time,
                    total_bartender_time=total_bartender_time,
                    marketing_source=row.get('Marketing Source', None),
                    referral_source=row.get('Referral Source', None),
                    status=status,
                    won=won,
                    lost=lost,
                    outcome=outcome
                )
                
                records.append(record)
            except Exception as e:
                print(f"Error processing row: {e}")
        
        # Insert records into the database
        session = Session()
        for record in records:
            try:
                session.merge(record)  # Use merge to handle updates if the record already exists
            except Exception as e:
                print(f"Error inserting record: {e}")
                session.rollback()
        
        session.commit()
        session.close()
        
        return len(records)
    except Exception as e:
        print(f"Error importing data: {e}")
        return 0


def import_operations_data(csv_path):
    """
    Import operations data from a CSV file into the database
    
    Args:
        csv_path (str): Path to the CSV file
        
    Returns:
        int: Number of records imported
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path, low_memory=False)
        
        # Convert data
        records = []
        for _, row in df.iterrows():
            try:
                # Extract Box Key from Name field if available
                box_key = None
                if 'Name' in row and pd.notna(row['Name']):
                    parts = str(row['Name']).split(' ')
                    for part in parts:
                        if part.startswith('BK-'):
                            box_key = part
                            break
                
                # If Box Key not found, use from Box Key column if available
                if not box_key and 'Box Key' in row and pd.notna(row['Box Key']):
                    box_key = row['Box Key']
                
                # Ensure we have a box key
                if not box_key:
                    continue
                
                # Process date fields
                event_date = None
                if 'Event Date' in row and pd.notna(row['Event Date']):
                    try:
                        event_date = pd.to_datetime(row['Event Date'])
                    except:
                        pass
                
                # Process numeric fields
                actual_deal_value = pd.to_numeric(row.get('Actual Deal Value', None), errors='coerce')
                days_until_event = pd.to_numeric(row.get('Days Until Event', None), errors='coerce')
                
                # Create a new Operation record
                record = Operation(
                    box_key=box_key,
                    event_date=event_date,
                    actual_deal_value=actual_deal_value,
                    region=row.get('Region', None),
                    name=row.get('Name', None),
                    days_until_event=days_until_event,
                    status=row.get('Status', None),
                    booking_type=row.get('Booking Type', None),
                    event_type=row.get('Event Type', None),
                    city=row.get('City', None),
                    state=row.get('State', None)
                )
                
                records.append(record)
            except Exception as e:
                print(f"Error processing row: {e}")
        
        # Insert records into the database
        session = Session()
        for record in records:
            try:
                session.merge(record)  # Use merge to handle updates if the record already exists
            except Exception as e:
                print(f"Error inserting record: {e}")
                session.rollback()
        
        session.commit()
        session.close()
        
        return len(records)
    except Exception as e:
        print(f"Error importing data: {e}")
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


# Run initialization if this script is run directly
if __name__ == "__main__":
    initialize_db_if_empty()