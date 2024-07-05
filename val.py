import argparse

from ultralytics import YOLO
import numpy as np
import os
import json


def _format_results(val_results, class_names):
    confusion_matrix = val_results.confusion_matrix
    detection_precision_metrics = val_results.results_dict
    speed_metrics = val_results.speed
    return {
        "num_classes": confusion_matrix.nc,
        "general_metrics": {
            "confidence_threshold": confusion_matrix.conf,
            "speed_metrics": {
                "inference_time": speed_metrics['inference'],
                "preprocess_time": speed_metrics['preprocess'],
                "postprocess_time": speed_metrics['postprocess']
            },
            "classification_metrics": {
                "accuracy_top1": val_results.top1,
                "accuracy_top5": val_results.top5,
                "fitness": detection_precision_metrics['fitness'],
            }
        },
        "class_specific_metrics": _get_class_metrics(val_results, class_names)
    }


def _save_to_json_file(data, save_path, file_name='results.json'):
    os.makedirs(save_path, exist_ok=True)
    full_save_path = os.path.join(save_path, file_name)
    with open(full_save_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)


def _get_class_metrics(validation_values, class_names):
    confusion_matrix_obj = validation_values.confusion_matrix
    conf_matrix = confusion_matrix_obj.matrix
    num_classes = confusion_matrix_obj.nc
    class_metrics = {}
    for i in range(num_classes):
        TP = conf_matrix[i, i]
        FN = np.sum(conf_matrix[:, i]) - TP
        FP = np.sum(conf_matrix[i, :]) - TP
        TN = np.sum(conf_matrix) - (TP + FP + FN)
        total_samples = np.sum(conf_matrix)

        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0

        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        misclassification_rate = (FP + FN) / total_samples if total_samples > 0 else 0

        class_metrics[class_names[i]] = {
            "tp": int(TP),
            "tn": int(TN),
            "fp": int(FP),
            "fn": int(FN),
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "misclassification_rate": misclassification_rate * 100  # in percentage
        }
    return class_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validate trained YOLO.')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model file.')
    # /Users/arthi/Documents/UALR/projects/skin-lesion-classification/code/runs/classify/train/weights/best.pt
    args = parser.parse_args()
    model = YOLO(args.model)  # load a custom model

    # Validate the model
    metrics = model.val(
        save_json=True,
        data="C:/Users/arthi/Documents/projects/skin-lesion-classification/dataset",
        plots=True,
        device='mps'
    )
    val_save_path = metrics.save_dir
    class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    formatted_results = _format_results(metrics, class_names)
    _save_to_json_file(formatted_results, val_save_path)

