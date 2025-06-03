import json
import os
from pathlib import Path

def create_submission():
    # Paths
    image_ids_file = "./image_ids.json"
    predictions_dir = "./results_pred/preds"
    output_file = "results_pred/submission.json"
    
    # Load image IDs mapping
    with open(image_ids_file, 'r') as f:
        image_data = json.load(f)
    
    # Create filename to ID mapping
    filename_to_id = {}
    for image_info in image_data['images']:
        filename_to_id[image_info['file_name']] = image_info['id']
    
    # List all prediction files
    pred_files = list(Path(predictions_dir).glob("*.json"))
    
    # Process predictions
    submission_data = []
    
    for pred_file in pred_files:
        # Extract filename from prediction file name
        pred_filename = pred_file.stem + ".jpg"  # Remove .json and add .jpg
        
        # Find corresponding image ID
        if pred_filename not in filename_to_id:
            print(f"Warning: {pred_filename} not found in image_ids.json")
            continue
            
        image_id = filename_to_id[pred_filename]
        
        # Load prediction data
        with open(pred_file, 'r') as f:
            pred_data = json.load(f)
        
        # Convert each detection to submission format
        labels = pred_data['labels']
        scores = pred_data['scores']
        bboxes = pred_data['bboxes']
        
        for label, score, bbox in zip(labels, scores, bboxes):
            # Convert bbox from [x1, y1, x2, y2] to [x, y, width, height]
            x1, y1, x2, y2 = bbox
            x = x1
            y = y1
            width = x2 - x1
            height = y2 - y1
            
            detection = {
                "image_id": image_id,
                "category_id": label + 1,
                "bbox": [x, y, width, height],
                "score": score
            }
            submission_data.append(detection)
    
    # Save submission file
    with open(output_file, 'w') as f:
        json.dump(submission_data, f, indent=2)
    
    print(f"Submission file created: {output_file}")
    print(f"Total detections: {len(submission_data)}")
    print(f"Images processed: {len([f for f in pred_files if (f.stem + '.jpg') in filename_to_id])}")
    
    # Print a sample to verify format
    if submission_data:
        print("\nSample detection:")
        print(json.dumps(submission_data[0], indent=2))
    
    return submission_data

if __name__ == "__main__":
    submission = create_submission()