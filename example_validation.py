"""
Example script showing how to integrate GEOS-5 FP file validation with downloads.

This demonstrates how to use the validation functionality after downloading
GEOS-5 FP files to ensure they are valid before processing.
"""

import os
import logging
from datetime import datetime

from GEOS5FP import (
    GEOS5FP,
    validate_GEOS5FP_NetCDF_file,
    validate_GEOS5FP_directory,
    is_valid_GEOS5FP_file
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_and_validate_example():
    """
    Example of downloading GEOS-5 FP data and validating the files.
    """
    
    print("GEOS-5 FP Download and Validation Example")
    print("=" * 50)
    
    # Initialize GEOS5FP connection
    geos5fp = GEOS5FP()
    
    # Example timestamp and geometry
    timestamp = "2025-02-22 12:00:00"
    
    try:
        # This would normally download a file, but for this example
        # we'll just show how validation would be integrated
        
        print(f"Example: Processing GEOS-5 FP data for {timestamp}")
        
        # In real usage, you would call methods like:
        # Ta_C = geos5fp.Ta_C(time_UTC=timestamp, geometry=geometry)
        
        # For demonstration, let's show how to validate files in a directory
        # that might contain downloaded GEOS-5 FP data
        
        download_directory = os.path.expanduser("~/data/GEOS5FP")
        
        if os.path.exists(download_directory):
            print(f"\nValidating GEOS-5 FP files in: {download_directory}")
            
            # Validate all NetCDF files in the directory
            results = validate_GEOS5FP_directory(download_directory, verbose=True)
            
            if results:
                print("\nValidation Results:")
                print("-" * 30)
                
                valid_count = 0
                for filename, result in results.items():
                    status = "✓ VALID" if result.is_valid else "✗ INVALID"
                    print(f"{status}: {filename}")
                    
                    if result.is_valid:
                        valid_count += 1
                        # Show some metadata for valid files
                        if 'file_size_mb' in result.metadata:
                            print(f"   Size: {result.metadata['file_size_mb']} MB")
                        if 'product_name' in result.metadata:
                            print(f"   Product: {result.metadata['product_name']}")
                    else:
                        # Show errors for invalid files
                        for error in result.errors[:2]:  # Show first 2 errors
                            print(f"   Error: {error}")
                    print()
                
                print(f"Summary: {valid_count}/{len(results)} files are valid")
            
            else:
                print("No GEOS-5 FP files found in directory")
        
        else:
            print(f"Download directory does not exist: {download_directory}")
            
    except Exception as e:
        logger.error(f"Error in example: {e}")


def validate_file_after_download():
    """
    Example of how to validate a file immediately after download.
    """
    
    print("\nFile Validation After Download Example")
    print("=" * 50)
    
    # Example filename that would be downloaded
    example_filename = "GEOS.fp.asm.tavg1_2d_slv_Nx.20250214_1200.V01.nc4"
    
    # In a real scenario, this would be the actual path after download
    download_path = os.path.join(os.path.expanduser("~/data/GEOS5FP"), example_filename)
    
    print(f"Example validation of: {example_filename}")
    
    # Quick boolean check
    if is_valid_GEOS5FP_file(download_path):
        print("✓ File is valid and ready for processing")
    else:
        print("✗ File validation failed")
        
        # Get detailed validation results
        result = validate_GEOS5FP_NetCDF_file(download_path, verbose=True)
        
        print("\nValidation Details:")
        print(result)
        
        # Handle validation failure
        if not result.is_valid:
            print("\nRecommended actions:")
            print("1. Re-download the file")
            print("2. Check network connection")
            print("3. Verify the source URL is accessible")
            
            # You might want to delete the invalid file
            # os.remove(download_path)


def enhanced_download_with_validation():
    """
    Example of an enhanced download function that includes validation.
    """
    
    def download_with_validation(url: str, filename: str, max_retries: int = 3):
        """
        Enhanced download function that validates files after download.
        """
        from GEOS5FP.download_file import download_file
        
        for attempt in range(max_retries):
            try:
                # Download the file
                downloaded_file = download_file(url, filename)
                
                # Validate the downloaded file
                result = validate_GEOS5FP_NetCDF_file(downloaded_file)
                
                if result.is_valid:
                    logger.info(f"Successfully downloaded and validated: {filename}")
                    logger.info(f"File size: {result.metadata.get('file_size_mb', 'unknown')} MB")
                    return downloaded_file
                else:
                    logger.warning(f"Downloaded file failed validation: {filename}")
                    for error in result.errors:
                        logger.warning(f"  Validation error: {error}")
                    
                    # Remove invalid file
                    if os.path.exists(downloaded_file):
                        os.remove(downloaded_file)
                        logger.info(f"Removed invalid file: {downloaded_file}")
                    
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying download (attempt {attempt + 2}/{max_retries})...")
                    
            except Exception as e:
                logger.error(f"Download attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying download (attempt {attempt + 2}/{max_retries})...")
        
        raise Exception(f"Failed to download valid file after {max_retries} attempts")
    
    print("\nEnhanced Download with Validation Example")
    print("=" * 50)
    print("This shows how you could integrate validation into the download process")
    print("to ensure only valid files are kept after download.")


def batch_validation_example():
    """
    Example of batch validation for quality assurance.
    """
    
    print("\nBatch Validation Example")
    print("=" * 50)
    
    # Example directories that might contain GEOS-5 FP data
    data_directories = [
        "~/data/GEOS5FP",
        "~/data/GEOS5FP/2025/02",
        "~/Downloads"
    ]
    
    all_results = {}
    
    for directory in data_directories:
        expanded_dir = os.path.expanduser(directory)
        if os.path.exists(expanded_dir):
            print(f"\nValidating files in: {expanded_dir}")
            results = validate_GEOS5FP_directory(expanded_dir, max_files=10)
            all_results[directory] = results
            
            if results:
                from GEOS5FP import get_validation_summary
                summary = get_validation_summary(results)
                print(f"  Found {summary['total_files']} files")
                print(f"  Valid: {summary['valid_files']} ({summary['validation_rate']}%)")
                print(f"  Errors: {summary['total_errors']}")
                print(f"  Warnings: {summary['total_warnings']}")
    
    # Generate overall report
    total_files = sum(len(results) for results in all_results.values())
    total_valid = sum(sum(1 for r in results.values() if r.is_valid) 
                     for results in all_results.values())
    
    print(f"\nOverall Summary:")
    print(f"Total files processed: {total_files}")
    print(f"Total valid files: {total_valid}")
    if total_files > 0:
        print(f"Overall validation rate: {total_valid/total_files*100:.1f}%")


if __name__ == "__main__":
    download_and_validate_example()
    validate_file_after_download()
    enhanced_download_with_validation()
    batch_validation_example()