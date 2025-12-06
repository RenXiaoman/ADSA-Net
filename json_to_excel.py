#!/usr/bin/env python3
"""
Convert JSON summary files to Excel format
将 JSON 格式的评估结果转换为 Excel 文件
"""

import json
import pandas as pd
from pathlib import Path
import argparse


def json_to_excel(json_path, output_path=None, sort_by='dice_score', ascending=False):
    """
    Convert JSON summary to Excel file
    
    Args:
        json_path: Path to the JSON file
        output_path: Path to save the Excel file (optional)
        sort_by: Column to sort by ('dice_score', 'miou_score', 'patient_name')
        ascending: Sort in ascending order (default: False for descending)
    """
    # Read JSON file
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract case data
    cases = data.get('cases', [])
    df_cases = pd.DataFrame(cases)
    
    # Keep only required columns
    df_cases = df_cases[['patient_name', 'dice_score', 'miou_score']]
    
    # Sort if specified
    if sort_by in df_cases.columns:
        df_cases = df_cases.sort_values(by=sort_by, ascending=ascending)
    
    # Reset index after sorting
    df_cases = df_cases.reset_index(drop=True)
    
    # Determine output path
    if output_path is None:
        json_path = Path(json_path)
        output_path = json_path.parent / f"{json_path.stem}.xlsx"
    
    # Write to Excel with single sheet
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_cases.to_excel(writer, sheet_name='Cases', index=False)
        
        # Auto-adjust column widths
        worksheet = writer.sheets['Cases']
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column_letter].width = adjusted_width
    
    # Calculate statistics for display
    avg_dice = df_cases['dice_score'].mean()
    avg_miou = df_cases['miou_score'].mean()
    
    print(f"✅ Excel file saved to: {output_path}")
    print(f"📊 Total cases: {len(df_cases)}")
    print(f"📈 Average Dice: {avg_dice:.4f}")
    print(f"📈 Average mIoU: {avg_miou:.4f}")
    
    return output_path


def batch_convert(input_dir, output_dir=None, pattern="*Summary.json"):
    """
    Batch convert multiple JSON files to Excel
    
    Args:
        input_dir: Directory containing JSON files
        output_dir: Directory to save Excel files (optional, defaults to same as input)
        pattern: File pattern to match (default: *Summary.json)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir) if output_dir else input_path
    output_path.mkdir(parents=True, exist_ok=True)
    
    json_files = list(input_path.rglob(pattern))
    
    if not json_files:
        print(f"⚠️  No JSON files found matching pattern: {pattern}")
        return
    
    print(f"📁 Found {len(json_files)} JSON file(s)")
    
    for json_file in json_files:
        print(f"\n🔄 Processing: {json_file.name}")
        try:
            # Create output path maintaining directory structure
            relative_path = json_file.relative_to(input_path)
            excel_output = output_path / relative_path.parent / f"{json_file.stem}.xlsx"
            excel_output.parent.mkdir(parents=True, exist_ok=True)
            
            json_to_excel(json_file, excel_output)
        except Exception as e:
            print(f"❌ Error processing {json_file.name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert JSON summary files to Excel format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single file
  python json_to_excel.py --input infer/SegTumor_ALIEN_chengda_infer/val_results/A_Summary.json
  
  # Convert single file with custom output
  python json_to_excel.py --input A_Summary.json --output results.xlsx
  
  # Batch convert all JSON files in a directory
  python json_to_excel.py --batch_dir infer/SegTumor_ALIEN_chengda_infer/val_results
  
  # Sort by mIoU score
  python json_to_excel.py --input A_Summary.json --sort_by miou_score
        """
    )
    
    parser.add_argument('--input', type=str, 
                       help='Path to input JSON file')
    parser.add_argument('--output', type=str, 
                       help='Path to output Excel file (optional)')
    parser.add_argument('--batch_dir', type=str,
                       help='Directory for batch conversion')
    parser.add_argument('--pattern', type=str, default='*Summary.json',
                       help='File pattern for batch conversion (default: *Summary.json)')
    parser.add_argument('--sort_by', type=str, default='dice_score',
                       choices=['dice_score', 'miou_score', 'patient_name'],
                       help='Column to sort by (default: dice_score)')
    parser.add_argument('--ascending', action='store_true',
                       help='Sort in ascending order (default: descending)')
    
    args = parser.parse_args()
    
    if args.batch_dir:
        # Batch conversion mode
        batch_convert(args.batch_dir, pattern=args.pattern)
    elif args.input:
        # Single file conversion mode
        json_to_excel(args.input, args.output, args.sort_by, args.ascending)
    else:
        # Default: convert the example file
        default_json = "infer/SegTumor_ALIEN_chengda_infer/val_results/A_Summary.json"
        if Path(default_json).exists():
            json_to_excel(default_json, sort_by=args.sort_by, ascending=args.ascending)
        else:
            parser.print_help()
            print(f"\n❌ Default file not found: {default_json}")
            print("Please specify --input or --batch_dir")
