# Dataset Datasheet


This datasheet is generated only from `reports/snapshot/` and tracked project configuration. Missing evidence is stated as missing; no dataset history or license terms are inferred.

## Saved evaluation data

Source file: `reports/snapshot/eval_summary.json`
Source date: date not recorded in snapshot

- `accuracy`: `0.993`
- `calibration`: `{"ece_after": 0.005, "ece_before": 0.002, "reliability_diagram": "reports/figures/fig_05_calibration.png", "temperature": 1.3199}`
- `checkpoint_path`: `models/best_model.pth`
- `headline`: `{"accuracy": 0.993, "bootstrap_ci_95": {"accuracy": {"high": 0.9947, "low": 0.9913, "mean": 0.993}, "macro_f1": {"high": 0.9931, "low": 0.9871, "mean": 0.9902}, "weighted_f1": {"high": 0.9947, "low": 0.9912, "mean": 0.993}}, "macro_f1": 0.9903, "n_test": 8134, "weighted_f1": 0.993}`
- `macro_f1`: `0.9903`
- `mlflow_run_id`: `<redacted-identifier>`
- `per_crop`: `{"apple": {"accuracy": 1.0, "is_single_class": false, "macro_f1": 1.0, "support": 474}, "blueberry": {"accuracy": 1.0, "is_single_class": true, "macro_f1": 1.0, "support": 225}, "cherry_(including_sour)": {"accuracy": 1.0, "is_single_class": false, "macro_f1": 1.0, "support": 286}, "corn_(maize)": {"accuracy": 0.9706, "is_single_class": false, "macro_f1": 0.9593, "support": 578}, "grape": {"accuracy": 1.0, "is_single_class": false, "macro_f1": 1.0, "support": 608}, "orange": {"accuracy": 1.0, "is_single_class": true, "macro_f1": 1.0, "support": 826}, "peach": {"accuracy": 1.0, "is_single_class": false, "macro_f1": 1.0, "support": 399}, "pepper,_bell": {"accuracy": 1.0, "is_single_class": false, "macro_f1": 1.0, "support": 372}, "potato": {"accuracy": 0.9907, "is_single_class": false, "macro_f1": 0.7371, "support": 323}, "raspberry": {"accuracy": 1.0, "is_single_class": true, "macro_f1": 1.0, "support": 56}, "soybean": {"accuracy": 1.0, "is_single_class": true, "macro_f1": 1.0, "support": 764}, "squash": {"accuracy": 1.0, "is_single_class": true, "macro_f1": 1.0, "support": 275}, "strawberry": {"accuracy": 1.0, "is_single_class": false, "macro_f1": 1.0, "support": 234}, "tomato": {"accuracy": 0.9864, "is_single_class": false, "macro_f1": 0.821, "support": 2714}}`
- `single_class_crops_excluded`: `["blueberry", "orange", "raspberry", "soybean", "squash"]`
- `top_10_confused_pairs`: `[{"count": 10, "pred": "Corn_(maize)___Northern_Leaf_Blight", "true": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot"}, {"count": 6, "pred": "Tomato___Late_blight", "true": "Tomato___Early_blight"}, {"count": 5, "pred": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "true": "Corn_(maize)___Northern_Leaf_Blight"}, {"count": 4, "pred": "Tomato___Spider_mites Two-spotted_spider_mite", "true": "Tomato___Tomato_Yellow_Leaf_Curl_Virus"}, {"count": 3, "pred": "Tomato___Early_blight", "true": "Tomato___Late_blight"}, {"count": 3, "pred": "Tomato___healthy", "true": "Tomato___Target_Spot"}, {"count": 3, "pred": "Tomato___Bacterial_spot", "true": "Tomato___Tomato_Yellow_Leaf_Curl_Virus"}, {"count": 2, "pred": "Potato___healthy", "true": "Potato___Late_blight"}, {"count": 2, "pred": "Tomato___Target_Spot", "true": "Tomato___Septoria_leaf_spot"}, {"count": 2, "pred": "Tomato___Target_Spot", "true": "Tomato___Spider_mites Two-spotted_spider_mite"}]`
- `weakest_5_classes`: `[{"class_idx": 7, "class_name": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "f1": 0.8993, "precision": 0.9306, "recall": 0.8701, "support": 77}, {"class_idx": 9, "class_name": "Corn_(maize)___Northern_Leaf_Blight", "f1": 0.9435, "precision": 0.9281, "recall": 0.9595, "support": 148}, {"class_idx": 22, "class_name": "Potato___healthy", "f1": 0.9583, "precision": 0.92, "recall": 1.0, "support": 23}, {"class_idx": 29, "class_name": "Tomato___Early_blight", "f1": 0.9592, "precision": 0.9792, "recall": 0.94, "support": 150}, {"class_idx": 30, "class_name": "Tomato___Late_blight", "f1": 0.9748, "precision": 0.9748, "recall": 0.9748, "support": 278}]`
- `weighted_f1`: `0.993`
- `within_crop_macro_f1_excluding_single_class_crops`: `0.9888`

## Seed summary

Source file: `reports/snapshot/seed_variance.json`
Source date: date not recorded in snapshot

- `driving_class_counts`: `{"champion": {"correct": 70, "total": 77}, "seed_43": {"correct": 68, "total": 77}, "seed_44": {"correct": 69, "total": 77}}`
- `metrics`: `{"bg_drop_black": {"champion": 0.1843, "mean": 0.17213333333333333, "range": 0.031899999999999984, "seed_43": 0.1797, "seed_44": 0.1524}, "bg_drop_mid_grey": {"champion": 0.0938, "mean": 0.08326666666666667, "range": 0.043399999999999994, "seed_43": 0.0563, "seed_44": 0.0997}, "bg_drop_per_image_mean": {"champion": 0.0364, "mean": 0.03656666666666667, "range": 0.0407, "seed_43": 0.0163, "seed_44": 0.057}, "clean_macro_f1": {"champion": 0.991907, "mean": 0.9916876666666666, "range": 0.002794000000000074, "seed_43": 0.990181, "seed_44": 0.992975}, "ece": {"champion": 0.001097, "mean": 0.0013586666666666667, "range": 0.000495, "seed_43": 0.001387, "seed_44": 0.001592}, "min_recall": {"champion": 0.909091, "mean": 0.8961039999999999, "range": 0.02597399999999994, "seed_43": 0.883117, "seed_44": 0.896104}, "within_crop_macro_f1": {"champion": 0.990758, "mean": 0.991863, "range": 0.0021240000000000148, "seed_43": 0.992882, "seed_44": 0.991949}}`
- `minimum_recall_driving_class`: `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot`
- `models`: `{"champion": {"bg_drop_black": 0.1843, "bg_drop_mid_grey": 0.0938, "bg_drop_per_image_mean": 0.0364, "clean_macro_f1": 0.991907, "ece": 0.001097, "min_recall": 0.909091, "run_id": "<redacted-identifier>", "within_crop_macro_f1": 0.990758}, "seed_43": {"bg_drop_black": 0.1797, "bg_drop_mid_grey": 0.0563, "bg_drop_per_image_mean": 0.0163, "clean_macro_f1": 0.990181, "ece": 0.001387, "min_recall": 0.883117, "run_id": "<redacted-identifier>", "within_crop_macro_f1": 0.992882}, "seed_44": {"bg_drop_black": 0.1524, "bg_drop_mid_grey": 0.0997, "bg_drop_per_image_mean": 0.057, "clean_macro_f1": 0.992975, "ece": 0.001592, "min_recall": 0.896104, "run_id": "<redacted-identifier>", "within_crop_macro_f1": 0.991949}}`
- `notes`: `["Three seeds are a small sample; summaries are descriptive, not a stable variance estimate.", "The earlier 0.8750 (70/80) uses a different denominator from supplied VAL n=77. Current counts are derived from each report; seed 43's supplied 0.883117 is 68/77. Reconcile 70/80 against its source before comparing."]`
- `split`: `VAL`

## Data characteristics and limitations

The project evaluates plant-leaf image classes using a PlantVillage-derived dataset as named in the project configuration and documentation. The available data are lab-style images; field representativeness is not established. A missing evaluation snapshot means counts and class distributions are not generated in this card.

## Collection, consent, and license

Collection methods, consent details, and redistribution terms are not established by the committed snapshot inputs. Verify the dataset license and permitted uses before public release.
