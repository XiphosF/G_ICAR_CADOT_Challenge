import json
import argparse
from collections import defaultdict
import numpy as np
from ensemble_boxes import weighted_boxes_fusion

def load_predictions(file_path):
    """Charge les prédictions depuis un fichier JSON"""
    with open(file_path, 'r') as f:
        predictions = json.load(f)
    return predictions

def coco_to_wbf_format(bbox, img_width, img_height):
    """
    Convertit bbox COCO [x, y, width, height] vers WBF [x1, y1, x2, y2] normalisé
    """
    x, y, w, h = bbox
    x1 = x / img_width
    y1 = y / img_height
    x2 = (x + w) / img_width
    y2 = (y + h) / img_height
    return [x1, y1, x2, y2]

def wbf_to_coco_format(bbox, img_width, img_height):
    """
    Convertit bbox WBF [x1, y1, x2, y2] normalisé vers COCO [x, y, width, height]
    """
    x1, y1, x2, y2 = bbox
    x = x1 * img_width
    y = y1 * img_height
    w = (x2 - x1) * img_width
    h = (y2 - y1) * img_height
    return [x, y, w, h]

def group_predictions_by_image(predictions):
    """Groupe les prédictions par image_id"""
    grouped = defaultdict(list)
    for pred in predictions:
        grouped[pred['image_id']].append(pred)
    return grouped

def prepare_wbf_inputs(grouped_predictions, img_width, img_height):
    """
    Prépare les inputs pour WBF depuis les prédictions groupées
    """
    boxes_list = []
    scores_list = []
    labels_list = []
    
    for predictions in grouped_predictions:
        boxes = []
        scores = []
        labels = []
        
        for pred in predictions:
            # Convertir bbox COCO vers format WBF
            wbf_bbox = coco_to_wbf_format(pred['bbox'], img_width, img_height)
            boxes.append(wbf_bbox)
            scores.append(pred['score'])
            labels.append(pred['category_id'])
        
        boxes_list.append(boxes)
        scores_list.append(scores)
        labels_list.append(labels)
    
    return boxes_list, scores_list, labels_list

def ensemble_single_image(predictions_per_model, img_width, img_height, 
                         weights, iou_thr=0.5, skip_box_thr=0.0001):
    """
    Applique WBF sur les prédictions d'une seule image provenant de plusieurs modèles
    """
    boxes_list, scores_list, labels_list = prepare_wbf_inputs(
        predictions_per_model, img_width, img_height
    )
    
    # Appliquer WBF
    ensemble_boxes, ensemble_scores, ensemble_labels = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list,
        weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr
    )
    
    return ensemble_boxes, ensemble_scores, ensemble_labels

def main():
    parser = argparse.ArgumentParser(description='Ensemble 5 prediction files using WBF')
    parser.add_argument('--pred_files', nargs=5, required=True, 
                       help='5 prediction JSON files')
    parser.add_argument('--output_file', required=True, 
                       help='Output submission file')
    parser.add_argument('--img_width', type=int, default=500, 
                       help='Image width for normalization')
    parser.add_argument('--img_height', type=int, default=500, 
                       help='Image height for normalization')
    parser.add_argument('--weights', nargs=5, type=float, 
                       default=[1.0, 1.0, 1.0, 1.0, 1.0],
                       help='Weights for each model')
    parser.add_argument('--iou_thr', type=float, default=0.5,
                       help='IoU threshold for WBF')
    parser.add_argument('--skip_box_thr', type=float, default=0.0001,
                       help='Skip box threshold for WBF')
    
    args = parser.parse_args()
    
    print("Loading prediction files...")
    all_predictions = []
    for i, pred_file in enumerate(args.pred_files):
        print(f"Loading {pred_file}...")
        predictions = load_predictions(pred_file)
        all_predictions.append(predictions)
        print(f"Model {i+1}: {len(predictions)} predictions")
    
    # Grouper les prédictions par image_id pour chaque modèle
    print("Grouping predictions by image...")
    grouped_by_model = []
    all_image_ids = set()
    
    for predictions in all_predictions:
        grouped = group_predictions_by_image(predictions)
        grouped_by_model.append(grouped)
        all_image_ids.update(grouped.keys())
    
    print(f"Found {len(all_image_ids)} unique images")
    
    # Ensemble pour chaque image
    final_predictions = []
    
    for img_id in sorted(all_image_ids):
        print(f"Processing image {img_id}...")
        
        # Récupérer les prédictions de chaque modèle pour cette image
        predictions_per_model = []
        for grouped in grouped_by_model:
            if img_id in grouped:
                predictions_per_model.append(grouped[img_id])
            else:
                # Si un modèle n'a pas de prédictions pour cette image
                predictions_per_model.append([])
        
        # Ignorer les images sans prédictions
        if all(len(preds) == 0 for preds in predictions_per_model):
            continue
        
        # Appliquer WBF
        ensemble_boxes, ensemble_scores, ensemble_labels = ensemble_single_image(
            predictions_per_model, args.img_width, args.img_height,
            args.weights, args.iou_thr, args.skip_box_thr
        )
        
        # Convertir les résultats vers le format COCO final
        for box, score, label in zip(ensemble_boxes, ensemble_scores, ensemble_labels):
            coco_bbox = wbf_to_coco_format(box, args.img_width, args.img_height)
            
            final_prediction = {
                "image_id": int(img_id),
                "category_id": int(label),
                "bbox": coco_bbox,
                "score": float(score)
            }
            final_predictions.append(final_prediction)
    
    print(f"Final ensemble: {len(final_predictions)} predictions")
    
    # Sauvegarder le fichier final
    print(f"Saving ensemble predictions to {args.output_file}...")
    with open(args.output_file, 'w') as f:
        json.dump(final_predictions, f, indent=2)
    
    print("Ensemble completed successfully!")
    
    # Statistiques finales
    print("\n=== ENSEMBLE STATISTICS ===")
    print(f"Input files: {len(args.pred_files)}")
    print(f"Model weights: {args.weights}")
    print(f"IoU threshold: {args.iou_thr}")
    print(f"Skip box threshold: {args.skip_box_thr}")
    print(f"Total images processed: {len(all_image_ids)}")
    print(f"Final predictions: {len(final_predictions)}")
    
    # Distribution par classe
    class_counts = defaultdict(int)
    for pred in final_predictions:
        class_counts[pred['category_id']] += 1
    
    print("\nPredictions per class:")
    for class_id in sorted(class_counts.keys()):
        print(f"  Class {class_id}: {class_counts[class_id]} predictions")

if __name__ == "__main__":
    main()

# # Usage basique avec poids égaux
# python ensemble_predictions.py \
#     --pred_files model1_pred.json model2_pred.json model3_pred.json model4_pred.json model5_pred.json \
#     --output_file final_submission.json

# # Usage avec poids différents pour chaque modèle
# python ensemble_predictions.py \
#     --pred_files model1_pred.json model2_pred.json model3_pred.json model4_pred.json model5_pred.json \
#     --output_file final_submission.json \
#     --weights 1.2 1.0 1.1 0.9 1.0 \
#     --iou_thr 0.6 \
#     --skip_box_thr 0.001

# # Si tes images ne font pas 800x800
# python ensemble_predictions.py \
#     --pred_files model1_pred.json model2_pred.json model3_pred.json model4_pred.json model5_pred.json \
#     --output_file final_submission.json \
#     --img_width 1000 --img_height 1000