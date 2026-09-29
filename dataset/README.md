# Veyraxis Sentinel Dataset Directory

This directory contains the annotated dataset organized into train, val, and test partitions:

```
dataset/
├── images/
│   ├── train/       # Training input images
│   ├── val/         # Validation split images
│   └── test/        # Isolated test split images
├── labels/
│   ├── train/       # YOLO format bounding box annotations
│   ├── val/         # YOLO format bounding box annotations
│   └── test/        # YOLO format bounding box annotations
├── data.yaml        # Ultralytics dataset configuration file
├── dataset_card.md  # Detailed dataset documentation & governance
└── README.md        # This guide
```

### Validation Command
Run the Veyraxis dataset validator to audit dataset integrity:
```bash
python scripts/validate_dataset.py --data dataset/data.yaml --output-dir outputs/dataset_reports
```
