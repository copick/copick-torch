import os
import json
import sqlite3
import argparse
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import List, Dict
import multiprocessing
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@contextmanager
def get_db_connection(db_path):
    """Create and return a database connection with context management."""
    conn = sqlite3.connect(db_path)
    try:
        yield conn
    finally:
        conn.close()

def create_table(db_path):
    """Create the database table with appropriate indices."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS subvolumes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            coordinate_x REAL,
            coordinate_y REAL,
            coordinate_z REAL,
            dtype TEXT,
            hierarchy_path TEXT,
            label INTEGER,
            mb_proteins TEXT,
            membranes TEXT,
            misalign_max REAL,
            misalign_min REAL,
            misalign_sigma REAL,
            orientation BLOB,
            protein_name TEXT,
            proteins TEXT,
            random_seed INTEGER,
            run_name TEXT,
            shape TEXT,
            snr REAL,
            subvolume_index INTEGER,
            tilt_range_max REAL,
            tilt_range_min REAL,
            tilt_range_step REAL,
            timestamp TEXT,
            tomo_shape_x INTEGER,
            tomo_shape_y INTEGER,
            tomo_shape_z INTEGER,
            tomogram_id TEXT,
            voxel_size REAL
        );
        """)
        # Add indices for commonly queried fields
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_protein_name ON subvolumes(protein_name);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_coordinates ON subvolumes(coordinate_x, coordinate_y, coordinate_z);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_snr ON subvolumes(snr);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_hierarchy_path ON subvolumes(hierarchy_path);")
        conn.commit()

def process_metadata_batch(metadata_batch: List[Dict]) -> List[tuple]:
    """Process a batch of metadata entries and prepare them for DB insertion."""
    return [
        (
            meta.get("coordinate_x"),
            meta.get("coordinate_y"),
            meta.get("coordinate_z"),
            meta.get("dtype"),
            meta.get("hierarchy_path"),
            meta.get("label"),
            json.dumps(meta.get("mb_proteins")) if isinstance(meta.get("mb_proteins"), (list, dict)) else meta.get("mb_proteins"),
            json.dumps(meta.get("membranes")) if isinstance(meta.get("membranes"), (list, dict)) else meta.get("membranes"),
            meta.get("misalign_max"),
            meta.get("misalign_min"),
            meta.get("misalign_sigma"),
            meta.get("orientation"),
            meta.get("protein_name"),
            json.dumps(meta.get("proteins")) if isinstance(meta.get("proteins"), (list, dict)) else meta.get("proteins"),
            meta.get("random_seed"),
            meta.get("run_name"),
            json.dumps(meta.get("shape")),
            meta.get("snr"),
            meta.get("subvolume_index"),
            meta.get("tilt_range_max"),
            meta.get("tilt_range_min"),
            meta.get("tilt_range_step"),
            meta.get("timestamp"),
            meta.get("tomo_shape_x"),
            meta.get("tomo_shape_y"),
            meta.get("tomo_shape_z"),
            meta.get("tomogram_id"),
            meta.get("voxel_size"),
        )
        for meta in metadata_batch
    ]

def batch_insert_metadata(cursor, batch_data: List[tuple]):
    """Insert a batch of metadata entries into the database."""
    cursor.executemany("""
    INSERT INTO subvolumes (
        coordinate_x, coordinate_y, coordinate_z, dtype, hierarchy_path, label,
        mb_proteins, membranes, misalign_max, misalign_min, misalign_sigma,
        orientation, protein_name, proteins, random_seed, run_name, shape, snr,
        subvolume_index, tilt_range_max, tilt_range_min, tilt_range_step,
        timestamp, tomo_shape_x, tomo_shape_y, tomo_shape_z, tomogram_id,
        voxel_size
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, batch_data)

def read_zattrs(path: str) -> Dict:
    """Read and parse a .zattrs file."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Error reading {path}: {e}")
        return None

def process_zarr_chunk(paths: List[str]) -> List[Dict]:
    """Process a chunk of .zattrs files in parallel."""
    results = []
    for path in paths:
        metadata = read_zattrs(path)
        if metadata:
            results.append(metadata)
    return results

def extract_metadata_from_zattrs(base_path: str, db_path: str, batch_size: int = 10000):
    """Extract metadata from .zattrs files and store in a SQLite database."""
    logger.info(f"\nScanning directory: {base_path}")
    
    # Collect all .zattrs paths
    zattrs_paths = []
    for root, _, files in os.walk(base_path):
        if ".zattrs" in files:
            zattrs_paths.append(os.path.join(root, ".zattrs"))
    
    # Calculate chunk size for parallel processing
    num_cores = multiprocessing.cpu_count()
    chunk_size = max(1, len(zattrs_paths) // (num_cores * 8))  # More aggressive parallelization
    path_chunks = [zattrs_paths[i:i + chunk_size] for i in range(0, len(zattrs_paths), chunk_size)]
    
    metadata_list = []
    processed_count = 0
    total_paths = len(zattrs_paths)
    
    # Process chunks in parallel
    with ThreadPoolExecutor(max_workers=num_cores) as executor:
        for chunk_result in executor.map(process_zarr_chunk, path_chunks):
            metadata_list.extend(chunk_result)
            processed_count += len(chunk_result)
            logger.info(f"Processed {processed_count}/{total_paths} files")
    
    logger.info("\nInserting metadata into database...")
    
    # Optimize SQLite for bulk insert
    with get_db_connection(db_path) as conn:
        conn.execute("PRAGMA journal_mode = WAL")  # Write-Ahead Logging
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA cache_size = -8000000")  # Use 8GB memory for cache
        conn.execute("PRAGMA temp_store = MEMORY")
        cursor = conn.cursor()
        
        # Process in batches
        for i in range(0, len(metadata_list), batch_size):
            batch = metadata_list[i:i + batch_size]
            batch_data = process_metadata_batch(batch)
            batch_insert_metadata(cursor, batch_data)
            conn.commit()
            logger.info(f"Inserted batch {i//batch_size + 1}/{(len(metadata_list)-1)//batch_size + 1}")

def run_spot_checks(db_path):
    """Run spot checks on the database to verify data integrity."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    logger.info("\n=== Running Spot Checks ===\n")
    
    # Check 1: Total number of records
    cursor.execute("SELECT COUNT(*) FROM subvolumes")
    count = cursor.fetchone()[0]
    logger.info(f"Total number of subvolumes: {count}")
    
    if count == 0:
        logger.error("\nNo data found in database. Please check the data loading process.")
        conn.close()
        return
    
    # Check 2: Sample of unique protein names
    cursor.execute("""
        SELECT DISTINCT protein_name 
        FROM subvolumes 
        WHERE protein_name IS NOT NULL 
        LIMIT 5
    """)
    proteins = cursor.fetchall()
    logger.info("\nSample of unique protein names:")
    for protein in proteins:
        logger.info(f"- {protein[0]}")
    
    # Check 3: Distribution of SNR values
    cursor.execute("""
        SELECT 
            MIN(snr) as min_snr,
            MAX(snr) as max_snr,
            AVG(snr) as avg_snr,
            COUNT(DISTINCT snr) as unique_snr_count
        FROM subvolumes 
        WHERE snr IS NOT NULL
    """)
    snr_stats = cursor.fetchone()
    logger.info(f"\nSNR Statistics:")
    logger.info(f"- Min: {snr_stats[0]:.2f}")
    logger.info(f"- Max: {snr_stats[1]:.2f}")
    logger.info(f"- Avg: {snr_stats[2]:.2f}")
    logger.info(f"- Unique values: {snr_stats[3]}")
    
    # Check 4: Sample of complete records
    cursor.execute("""
        SELECT coordinate_x, coordinate_y, coordinate_z, 
               protein_name, snr, voxel_size
        FROM subvolumes 
        LIMIT 3
    """)
    logger.info("\nSample of 3 complete records:")
    columns = ['x', 'y', 'z', 'protein', 'snr', 'voxel_size']
    df = pd.DataFrame(cursor.fetchall(), columns=columns)
    logger.info(f"\n{df.to_string()}")
    
    conn.close()

def main():
    """Main function to create database index for zarr files."""
    parser = argparse.ArgumentParser(description="Create SQLite index for zarr metadata")
    parser.add_argument("--zarr_path", required=True, help="Path to the zarr directory")
    parser.add_argument("--db_path", required=True, help="Path to save the SQLite database")
    parser.add_argument("--batch_size", type=int, default=10000, help="Batch size for database insertion")
    parser.add_argument("--run_checks", action="store_true", help="Run spot checks after creation")
    
    args = parser.parse_args()
    
    # Convert to Path objects
    zarr_path = Path(args.zarr_path)
    db_path = Path(args.db_path)
    
    # Ensure parent directory for database exists
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info("Creating database and table...")
    create_table(db_path)
    
    logger.info("Starting metadata extraction...")
    extract_metadata_from_zattrs(zarr_path, db_path, args.batch_size)
    
    if args.run_checks:
        logger.info("Running spot checks...")
        run_spot_checks(db_path)

if __name__ == "__main__":
    main()