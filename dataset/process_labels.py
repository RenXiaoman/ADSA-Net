#!/usr/bin/env python3

import os
import nibabel as nib
import numpy as np
from pathlib import Path
import shutil

def main():
    # Process both label directories
    label_dirs = [
        "dataset/Task205_picai_lesion/prostate158_labelsTr",
        "dataset/Task205_picai_lesion/prostate158_labelsTrVal",
        'dataset/Task205_picai_lesion/prostate158_imagesTr',
        'dataset/Task205_picai_lesion/prostate158_imagesTrVal'
    ]
    
    # Create backup directory
    backup_dir = Path("dataset/Task205_picai_lesion/backup")
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    for dir_path in label_dirs:
        dir_path = Path(dir_path)
        if not dir_path.exists():
            print(f"Directory {dir_path} does not exist, skipping...")
            continue
            
        print(f"Processing directory: {dir_path}")
        
        # Get all nii.gz files
        nii_files = list(dir_path.glob("*.nii.gz"))
        print(f"Found {len(nii_files)} files in {dir_path}")
        
        # Filter files: keep only those starting with letters, remove numeric starting files
        files_to_keep = []
        files_to_remove = []
        
        for file_path in nii_files:
            filename = file_path.name
            # Check if filename starts with a letter
            if filename[0].isalpha():
                files_to_keep.append(file_path)
            else:
                files_to_remove.append(file_path)
        
        print(f"Keeping {len(files_to_keep)} files, removing {len(files_to_remove)} files")
        
        # Backup files to be removed
        for file_path in files_to_remove:
            backup_file = backup_dir / file_path.name
            shutil.copy2(file_path, backup_file)
            print(f"Backed up: {file_path.name}")
        
        # Remove files starting with numbers
        for file_path in files_to_remove:
            file_path.unlink()
            print(f"Removed: {file_path.name}")
        
        print(f"Finished processing {dir_path}\n")
    
    print("All directories processed successfully!")
    print(f"Backup files saved to: {backup_dir}")

if __name__ == "__main__":
    main()