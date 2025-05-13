"""
Script to fix the database schema by adding missing columns
"""
import sqlite3
import os

def migrate_database():
    """
    Update database schema to include new columns
    """
    try:
        print("Starting database migration...")
        
        # Backup the current database
        if os.path.exists('streak_data.db'):
            import shutil
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs("backups", exist_ok=True)
            backup_path = f"backups/streak_data_{timestamp}.db"
            shutil.copy2('streak_data.db', backup_path)
            print(f"Created database backup at {backup_path}")
        
        # Connect to database
        conn = sqlite3.connect('streak_data.db')
        cursor = conn.cursor()
        
        # Check if the columns already exist in leads table
        cursor.execute("PRAGMA table_info(leads)")
        leads_columns = [column[1] for column in cursor.fetchall()]
        print(f"Existing leads columns: {leads_columns}")
        
        # Add new columns to leads table if they don't exist
        if 'first_name' not in leads_columns:
            cursor.execute("ALTER TABLE leads ADD COLUMN first_name TEXT")
            print("Added first_name column to leads table")
        if 'last_name' not in leads_columns:
            cursor.execute("ALTER TABLE leads ADD COLUMN last_name TEXT")
            print("Added last_name column to leads table")
        if 'email' not in leads_columns:
            cursor.execute("ALTER TABLE leads ADD COLUMN email TEXT")
            print("Added email column to leads table")
        if 'phone_number' not in leads_columns:
            cursor.execute("ALTER TABLE leads ADD COLUMN phone_number TEXT")
            print("Added phone_number column to leads table")
            
        # Check if the columns already exist in operations table
        cursor.execute("PRAGMA table_info(operations)")
        operations_columns = [column[1] for column in cursor.fetchall()]
        print(f"Existing operations columns: {operations_columns}")
        
        # Add new columns to operations table if they don't exist
        if 'first_name' not in operations_columns:
            cursor.execute("ALTER TABLE operations ADD COLUMN first_name TEXT")
            print("Added first_name column to operations table")
        if 'last_name' not in operations_columns:
            cursor.execute("ALTER TABLE operations ADD COLUMN last_name TEXT")
            print("Added last_name column to operations table")
        if 'email' not in operations_columns:
            cursor.execute("ALTER TABLE operations ADD COLUMN email TEXT")
            print("Added email column to operations table")
        if 'phone_number' not in operations_columns:
            cursor.execute("ALTER TABLE operations ADD COLUMN phone_number TEXT")
            print("Added phone_number column to operations table")
            
        # Commit changes
        conn.commit()
        print("Database migration successful")
        return True
    except Exception as e:
        print(f"Error migrating database: {e}")
        return False
    finally:
        if 'conn' in locals() and conn:
            conn.close()

if __name__ == "__main__":
    success = migrate_database()
    print(f"Migration {'succeeded' if success else 'failed'}")