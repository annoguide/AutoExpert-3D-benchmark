import json
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Filter detection results by score threshold and add file names')
    parser.add_argument('--r_path', type=str, default='data/nuscenes/outputs/result_2D_val.json',
                       help='Input result file path')
    parser.add_argument('--output_path', type=str, default='data/nuscenes/outputs/result_2D_val.json',
                       help='Output filtered result file path')
    parser.add_argument('--annos_path', type=str, default='data/nuscenes/test_2D_val.json',
                       help='Annotations file path')
    parser.add_argument('--score_thre', type=float, default=0.1,
                       help='Score threshold for filtering (default: 0.1)')
    
    args = parser.parse_args()
    
    r_path = args.r_path
    output_path = args.output_path
    annos_path = args.annos_path
    score_thre = args.score_thre
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(annos_path, "r") as f:
        annos = json.load(f)
    images_name_list = annos["images"]
    
    with open(r_path, 'r') as file:
        r = json.load(file)
    
    r_filter = []
    for index in range(len(r)):
        result = r[index]
        image_id = result['image_id']
        score = result['score']
        if score < score_thre:
            continue 
        for temp in images_name_list:
            if image_id == temp['id']:
                image_name = temp['file_name']
                break
        result['file_name'] = image_name
        r_filter.append(result)
    
    with open(output_path, 'w') as file:
        json.dump(r_filter, file)
    
    print(f"Filtering completed! Input: {len(r)} results, Output: {len(r_filter)} results (score >= {score_thre})")
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main()