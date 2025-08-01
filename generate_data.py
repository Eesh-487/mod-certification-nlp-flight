"""
Generate sample data if the main dataset is not available
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from generate_sample_data import generate_sample_files

if __name__ == "__main__":
    print("Generating sample aircraft modification data...")
    
    # Set output directory relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    
    generate_sample_files(data_dir)
    print("Sample data generation complete!")
