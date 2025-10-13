#!/usr/bin/env python3
"""
Clear all data from the experiments database.

This script will delete all records from the database while preserving
the table structure. Useful for cleaning up test experiments.

WARNING: This action cannot be undone! All experiments, metrics, and 
gradient snapshot records will be permanently deleted.
"""

import sys
import sqlite3
from pathlib import Path
import argparse


def get_db_path(custom_path=None):
    """Get database path."""
    if custom_path:
        return Path(custom_path)
    
    # Default path relative to script location
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    return project_root / 'results' / 'experiments.db'


def get_table_info(cursor):
    """Get information about tables in the database."""
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    
    info = {}
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        info[table] = count
    
    return info


def clear_database(db_path, dry_run=False, confirm=True):
    """
    Clear all data from the database.
    
    Args:
        db_path: Path to database file
        dry_run: If True, only show what would be deleted
        confirm: If True, ask for confirmation before deleting
    
    Returns:
        True if successful, False otherwise
    """
    db_path = Path(db_path)
    
    if not db_path.exists():
        print(f"❌ Database file not found: {db_path}")
        return False
    
    print(f"Database: {db_path}")
    print(f"Size: {db_path.stat().st_size / 1024:.2f} KB")
    print()
    
    # Connect to database
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Get current state
    print("Current database contents:")
    print("="*60)
    table_info = get_table_info(cursor)
    
    total_records = 0
    for table, count in sorted(table_info.items()):
        print(f"  {table:30s}: {count:6d} records")
        total_records += count
    
    print("="*60)
    print(f"  {'TOTAL':30s}: {total_records:6d} records")
    print()
    
    if total_records == 0:
        print("✅ Database is already empty!")
        conn.close()
        return True
    
    if dry_run:
        print("🔍 DRY RUN MODE - No changes will be made")
        print()
        print("Would delete:")
        for table, count in sorted(table_info.items()):
            if count > 0:
                print(f"  ❌ {count} records from '{table}'")
        conn.close()
        return True
    
    # Confirmation
    if confirm:
        print("⚠️  WARNING: This will permanently delete all data!")
        print("   The database structure will be preserved.")
        print()
        response = input("Are you sure you want to continue? (yes/no): ").strip().lower()
        
        if response not in ['yes', 'y']:
            print("\n❌ Operation cancelled")
            conn.close()
            return False
        print()
    
    # Delete data from all tables
    # Order matters due to foreign key constraints
    deletion_order = [
        'metrics',
        'gradient_snapshots',
        'measurement_points',
        'experiments'
    ]
    
    print("Deleting data...")
    print("="*60)
    
    try:
        # Disable foreign key constraints temporarily
        cursor.execute("PRAGMA foreign_keys = OFF")
        
        deleted_counts = {}
        for table in deletion_order:
            if table in table_info and table_info[table] > 0:
                cursor.execute(f"DELETE FROM {table}")
                deleted = cursor.rowcount
                deleted_counts[table] = deleted
                print(f"  ✅ Deleted {deleted} records from '{table}'")
        
        # Re-enable foreign key constraints
        cursor.execute("PRAGMA foreign_keys = ON")
        
        # Commit changes
        conn.commit()
        
        print("="*60)
        print()
        
        # Verify deletion
        print("Verifying database is empty...")
        final_info = get_table_info(cursor)
        final_total = sum(final_info.values())
        
        if final_total == 0:
            print("✅ Database successfully cleared!")
            print()
            print("Summary:")
            for table, count in sorted(deleted_counts.items()):
                print(f"  🗑️  {table}: {count} records deleted")
            
            # Get new file size
            conn.close()
            new_size = db_path.stat().st_size / 1024
            print()
            print(f"New database size: {new_size:.2f} KB")
            print()
            print("💡 Tip: You can now run VACUUM to reclaim disk space:")
            print(f"   sqlite3 {db_path} 'VACUUM;'")
            
            return True
        else:
            print(f"⚠️  Warning: {final_total} records still remain")
            conn.close()
            return False
            
    except Exception as e:
        print(f"\n❌ Error clearing database: {e}")
        conn.rollback()
        conn.close()
        return False


def vacuum_database(db_path):
    """
    Run VACUUM to reclaim disk space.
    
    Args:
        db_path: Path to database file
    """
    print(f"\nRunning VACUUM on {db_path}...")
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        old_size = Path(db_path).stat().st_size / 1024
        
        cursor.execute("VACUUM")
        conn.commit()
        conn.close()
        
        new_size = Path(db_path).stat().st_size / 1024
        saved = old_size - new_size
        
        print(f"✅ VACUUM complete!")
        print(f"   Old size: {old_size:.2f} KB")
        print(f"   New size: {new_size:.2f} KB")
        print(f"   Saved: {saved:.2f} KB ({saved/old_size*100:.1f}%)")
        
    except Exception as e:
        print(f"❌ Error running VACUUM: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Clear all data from the experiments database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Clear with confirmation prompt
  python scripts/clear_database.py
  
  # Clear without confirmation (dangerous!)
  python scripts/clear_database.py --no-confirm
  
  # Dry run to see what would be deleted
  python scripts/clear_database.py --dry-run
  
  # Clear and vacuum
  python scripts/clear_database.py --vacuum
  
  # Clear custom database
  python scripts/clear_database.py --db-path /path/to/database.db
        """
    )
    
    parser.add_argument(
        '--db-path',
        type=str,
        default=None,
        help='Path to database file (default: results/experiments.db)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be deleted without actually deleting'
    )
    
    parser.add_argument(
        '--no-confirm',
        action='store_true',
        help='Skip confirmation prompt (USE WITH CAUTION!)'
    )
    
    parser.add_argument(
        '--vacuum',
        action='store_true',
        help='Run VACUUM after clearing to reclaim disk space'
    )
    
    args = parser.parse_args()
    
    # Get database path
    db_path = get_db_path(args.db_path)
    
    print("="*60)
    print("Database Cleaner")
    print("="*60)
    print()
    
    # Clear database
    success = clear_database(
        db_path,
        dry_run=args.dry_run,
        confirm=not args.no_confirm
    )
    
    if not success:
        sys.exit(1)
    
    # Vacuum if requested
    if args.vacuum and not args.dry_run:
        vacuum_database(db_path)
    
    print("\n✅ Done!")
    sys.exit(0)


if __name__ == '__main__':
    main()
