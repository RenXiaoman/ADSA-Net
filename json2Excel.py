#!/usr/bin/env python3
"""
Convert JSON segmentation results to a simple Excel with patient names and UNet miou scores.
"""

import json
import pandas as pd
import argparse
from pathlib import Path

def json_to_excel(json_path, output_path=None):
    """
    Convert JSON to Excel with two columns: patient_name | UNet (miou_score)
    """
    json_path = Path(json_path)
    
    # 读取 JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 从 case_metrics 获取 patient_name 和 miou_score
    case_metrics = data.get('case_metrics', [])
    
    # 构建 DataFrame
    df = pd.DataFrame({
        'patient_name': [case['patient_name'] for case in case_metrics],
        'UNet': [case['miou_score'] for case in case_metrics]
    })
    
    # 默认输出路径
    if output_path is None:
        output_path = json_path.parent / f"{json_path.stem}.xlsx"
    
    # 保存 Excel
    df.to_excel(output_path, index=False)
    print(f"✅ Excel file saved to: {output_path}")
    
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSON miou results to Excel")
    parser.add_argument('--json', type=str, required=True, help='Path to JSON file')
    parser.add_argument('--output', type=str, default=None, help='Output Excel path (optional)')
    args = parser.parse_args()
    
    json_to_excel(args.json, args.output)