#!/usr/bin/env python3
"""
Compare multiple model results from JSON files
对比多个模型的评估结果
"""

import json
import pandas as pd
from pathlib import Path
import argparse
from openpyxl.styles import PatternFill, Font
from openpyxl.utils import get_column_letter


def load_json_data(json_path):
    """
    Load JSON file and return cases data
    Supports two formats:
    1. Format 1: {"cases": [{"patient_name": "...", "dice_score": ..., "miou_score": ...}]}
    2. Format 2: {"per_case": [{"case_id": "...", "dice": "...", "iou": "..."}]}
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Try Format 1 first (cases with patient_name, dice_score, miou_score)
    if 'cases' in data:
        return data.get('cases', []), data.get('overall_metrics', {})
    
    # Try Format 2 (per_case with case_id, dice, iou)
    elif 'per_case' in data:
        # Convert Format 2 to Format 1
        cases = []
        for case in data.get('per_case', []):
            cases.append({
                'patient_name': case.get('case_id', ''),
                'dice_score': float(case.get('dice', 0)),
                'miou_score': float(case.get('iou', 0))
            })
        
        # Build overall_metrics from top-level fields if available
        overall_metrics = {}
        if 'dice_mean' in data:
            overall_metrics['average_dice'] = float(data.get('dice_mean', 0))
        if 'iou_mean' in data:
            overall_metrics['average_miou'] = float(data.get('iou_mean', 0))
        
        return cases, overall_metrics
    
    else:
        print(f"⚠️  Warning: Unknown JSON format in {json_path}")
        return [], {}


def compare_models(json_paths, model_names=None, output_path='model_comparison.xlsx', existing_excel=None):
    """
    Compare multiple models' results
    
    Args:
        json_paths: List of paths to JSON files
        model_names: List of model names (optional, will use filenames if not provided)
        output_path: Path to save the comparison Excel file
        existing_excel: Path to existing Excel file for incremental update (optional)
    """
    if not json_paths:
        print("❌ No JSON files provided")
        return
    
    # Generate model names if not provided
    if model_names is None:
        model_names = [Path(p).stem for p in json_paths]
    
    if len(model_names) != len(json_paths):
        print("⚠️  Number of model names doesn't match number of JSON files")
        model_names = [f"Model_{i+1}" for i in range(len(json_paths))]
    
    # Load existing data if provided
    existing_dice_df = None
    existing_miou_df = None
    existing_model_names = []
    
    if existing_excel and Path(existing_excel).exists():
        print(f"📂 Loading existing Excel file: {existing_excel}")
        try:
            existing_dice_df = pd.read_excel(existing_excel, sheet_name='Dice Comparison')
            existing_miou_df = pd.read_excel(existing_excel, sheet_name='mIoU Comparison')
            
            # Get existing model names (exclude metadata columns)
            metadata_cols = ['patient_name', 'Best_Model', 'Best_Score', 'Worst_Score', 'Max_Diff', 'Avg_Score']
            existing_model_names = [col for col in existing_dice_df.columns if col not in metadata_cols]
            
            print(f"✅ Found {len(existing_model_names)} existing models: {', '.join(existing_model_names)}")
            
            # Check for duplicate model names
            duplicates = set(model_names) & set(existing_model_names)
            if duplicates:
                print(f"⚠️  Warning: The following models already exist and will be overwritten: {', '.join(duplicates)}")
            
        except Exception as e:
            print(f"⚠️  Could not load existing Excel file: {e}")
            print("Creating new comparison file...")
            existing_dice_df = None
            existing_miou_df = None
    
    print(f"📊 Adding {len(json_paths)} model(s):")
    for name, path in zip(model_names, json_paths):
        print(f"  - {name}: {path}")
    
    # Load all data
    all_data = {}
    overall_metrics = {}
    
    for model_name, json_path in zip(model_names, json_paths):
        cases, metrics = load_json_data(json_path)
        # Convert to dict with patient_name as key
        all_data[model_name] = {case['patient_name']: case for case in cases}
        overall_metrics[model_name] = metrics
    
    # Get all unique patient names (including from existing data)
    all_patients = set()
    for model_data in all_data.values():
        all_patients.update(model_data.keys())
    
    # Add patients from existing data if available
    if existing_dice_df is not None and 'patient_name' in existing_dice_df.columns:
        all_patients.update(existing_dice_df['patient_name'].tolist())
    
    all_patients = sorted(all_patients)
    
    print(f"📋 Found {len(all_patients)} unique patients")
    
    # Build separate dataframes for Dice and mIoU
    dice_rows = []
    miou_rows = []
    
    # Combine existing and new model names
    all_model_names = existing_model_names + model_names
    
    for patient in all_patients:
        dice_row = {'patient_name': patient}
        miou_row = {'patient_name': patient}
        
        dice_scores = []
        miou_scores = []
        dice_model_map = {}
        miou_model_map = {}
        
        # First, add data from existing models
        for model_name in existing_model_names:
            if existing_dice_df is not None and model_name in existing_dice_df.columns:
                patient_data = existing_dice_df[existing_dice_df['patient_name'] == patient]
                if not patient_data.empty:
                    dice = patient_data[model_name].values[0]
                    if pd.notna(dice):
                        dice_row[model_name] = dice
                        dice_scores.append(dice)
                        dice_model_map[dice] = model_name
                    else:
                        dice_row[model_name] = None
                else:
                    dice_row[model_name] = None
            
            if existing_miou_df is not None and model_name in existing_miou_df.columns:
                patient_data = existing_miou_df[existing_miou_df['patient_name'] == patient]
                if not patient_data.empty:
                    miou = patient_data[model_name].values[0]
                    if pd.notna(miou):
                        miou_row[model_name] = miou
                        miou_scores.append(miou)
                        miou_model_map[miou] = model_name
                    else:
                        miou_row[model_name] = None
                else:
                    miou_row[model_name] = None
        
        # Then, add/update data from new models
        for model_name in model_names:
            if patient in all_data[model_name]:
                dice = all_data[model_name][patient]['dice_score']
                miou = all_data[model_name][patient]['miou_score']
            else:
                dice = None
                miou = None
            
            dice_row[model_name] = dice
            miou_row[model_name] = miou
            
            if dice is not None:
                dice_scores.append(dice)
                dice_model_map[dice] = model_name
            if miou is not None:
                miou_scores.append(miou)
                miou_model_map[miou] = model_name
        
        dice_rows.append(dice_row)
        miou_rows.append(miou_row)
    
    df_dice = pd.DataFrame(dice_rows)
    df_miou = pd.DataFrame(miou_rows)
    
    # Reorder columns: patient_name first, then models in order
    dice_col_order = ['patient_name'] + [col for col in df_dice.columns if col != 'patient_name']
    miou_col_order = ['patient_name'] + [col for col in df_miou.columns if col != 'patient_name']
    
    df_dice = df_dice[dice_col_order]
    df_miou = df_miou[miou_col_order]
    
    # Write to Excel
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Sheet 1: Dice Comparison
        df_dice.to_excel(writer, sheet_name='Dice Comparison', index=False)
        
        # Sheet 2: mIoU Comparison
        df_miou.to_excel(writer, sheet_name='mIoU Comparison', index=False)
        
        # Apply formatting to Dice sheet
        ws_dice = writer.sheets['Dice Comparison']
        
        # Auto-adjust column widths
        for column in ws_dice.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 20)
            ws_dice.column_dimensions[column_letter].width = adjusted_width
        
        # Apply formatting to mIoU sheet
        ws_miou = writer.sheets['mIoU Comparison']
        
        # Auto-adjust column widths
        for column in ws_miou.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 20)
            ws_miou.column_dimensions[column_letter].width = adjusted_width
    
    print(f"\n✅ Comparison saved to: {output_path}")
    print(f"📊 Total patients: {len(df_dice)}")
    print(f"📊 Total models: {len(all_model_names)}")
    
    return output_path


def compare_from_directory(directory, pattern="*Summary.json", output_path=None):
    """
    Compare all JSON files in a directory
    
    Args:
        directory: Directory containing JSON files
        pattern: File pattern to match
        output_path: Output Excel file path
    """
    dir_path = Path(directory)
    json_files = sorted(dir_path.glob(pattern))
    
    if not json_files:
        print(f"❌ No JSON files found in {directory} matching pattern {pattern}")
        return
    
    json_paths = [str(f) for f in json_files]
    model_names = [f.stem for f in json_files]
    
    if output_path is None:
        output_path = dir_path / "model_comparison.xlsx"
    
    return compare_models(json_paths, model_names, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Compare multiple model results from JSON files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 1. Create new comparison from multiple JSON files
  python compare_models.py --inputs model1.json model2.json model3.json --names "UNET" "CDSA_Net" "Transformer"
  
  # 2. Add a single new model to existing comparison (incremental update)
  python compare_models.py --inputs new_model.json --names "NewModel" --existing comparison.xlsx --output comparison.xlsx
  
  # 3. Add multiple new models to existing comparison
  python compare_models.py --inputs model4.json model5.json --names "Model4" "Model5" --existing comparison.xlsx --output comparison.xlsx
  
  # 4. Compare all JSON files in a directory (batch mode)
  python compare_models.py --dir infer/results --pattern "*Summary.json"
  
  # 5. Specify custom output path
  python compare_models.py --inputs model1.json model2.json --output my_comparison.xlsx
        """
    )
    
    parser.add_argument('--inputs', nargs='+', type=str,
                       help='Paths to JSON files to compare')
    parser.add_argument('--names', nargs='+', type=str,
                       help='Model names (optional, must match number of inputs)')
    parser.add_argument('--existing', type=str,
                       help='Path to existing Excel file for incremental update')
    parser.add_argument('--dir', type=str,
                       help='Directory containing JSON files for batch comparison')
    parser.add_argument('--pattern', type=str, default='*Summary.json',
                       help='File pattern for directory mode (default: *Summary.json)')
    parser.add_argument('--output', type=str, default='model_comparison.xlsx',
                       help='Output Excel file path')
    
    args = parser.parse_args()
    
    if args.dir:
        # Directory mode
        compare_from_directory(args.dir, args.pattern, args.output)
    elif args.inputs:
        # Individual files mode
        compare_models(args.inputs, args.names, args.output, args.existing)
    else:
        parser.print_help()
        print("\n❌ Please specify either --inputs or --dir")
