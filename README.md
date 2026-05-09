# CV Classification Pipeline

A modular, configurable pipeline for Computer Vision classification tasks. Built following the same architecture principles as tabular data pipelines.

## Project Structure

```
MNIST/
├── config/
│   ├── config.py          # Central configuration (hyperparameters, paths)
│   └── paths.py           # Path management
├── src/
│   ├── data_loader.py     # Image data loading and preprocessing
│   ├── modeling/
│   │   └── models.py      # Model architectures (CNN, ResNet, etc.)
│   ├── training/
│   │   └── trainer.py     # Training loop and cross-validation
│   └── utils.py           # Utility functions
├── data/
│   ├── raw/               # Raw dataset files
│   └── processed/         # Processed data
├── outputs/
│   ├── models/            # Saved model weights
│   ├── logs/              # Training logs
│   ├── oof/               # Out-of-fold predictions
│   └── submissions/       # Submission files
├── eda/                   # Exploratory data analysis
├── main.py               # Entry point
└── requirements.txt      # Dependencies
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download MNIST Dataset

Download the MNIST dataset from Kaggle:
- Go to https://www.kaggle.com/c/digit-recognizer/data
- Download `train.csv` and `test.csv`
- Place them in `data/raw/`

Or use the Kaggle API:
```bash
kaggle competitions download -c digit-recognizer -p data/raw/
```

### 3. Run the Pipeline

#### Debug Mode (quick test)
```bash
python main.py --running_mode debug --model cnn --fold 3
```

#### Full Training
```bash
python main.py --running_mode train --model cnn --fold 5 --epochs 30
```

#### With Test Predictions
```bash
python main.py --running_mode train --model resnet18 --use_test --submit
```

## Usage Examples

### Train with Custom Parameters

```bash
python main.py \
    --running_mode train \
    --model resnet18 \
    --fold 5 \
    --epochs 50 \
    --batch_size 256 \
    --lr 0.001 \
    --device cuda
```

### Available Models

- `cnn`: Simple 4-layer CNN (default, fast training)
- `cnn_deep`: Deeper CNN with more layers
- `resnet18`: ResNet-18 (transfer learning)
- `resnet34`: ResNet-34 (transfer learning)
- `efficientnet_b0`: EfficientNet-B0 (transfer learning)

### Cross-Validation Strategies

- `stratified_kfold`: Maintains label distribution (default)
- `kfold`: Standard K-Fold

## Configuration

Edit `config/config.py` to customize:

- **Model Parameters**: `model_name`, `num_classes`
- **Training**: `num_epochs`, `batch_size`, `learning_rate`
- **Data**: `image_size`, `num_channels`
- **Augmentation**: `use_augmentation`, `augmentation_params`
- **Paths**: All output directories

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--running_mode` | `debug` | Mode: debug/train/predict |
| `--model` | `cnn` | Model architecture |
| `--fold` | `5` | Number of CV folds |
| `--epochs` | `20` | Training epochs |
| `--batch_size` | `128` | Batch size |
| `--lr` | `0.001` | Learning rate |
| `--cv_strategy` | `stratified_kfold` | CV strategy |
| `--use_test` | `False` | Generate test predictions |
| `--submit` | `False` | Create submission file |
| `--device` | `auto` | Device: auto/cuda/cpu |

## Architecture Principles

This pipeline follows the same modular design as tabular pipelines:

1. **Config-driven**: All parameters in `config.py`, no hardcoding
2. **Modular**: Separate modules for data, models, training
3. **Reproducible**: Seed setting, deterministic operations
4. **Extensible**: Easy to add new models, datasets, augmentations
5. **Production-ready**: Logging, checkpointing, proper error handling

## Extending for Other Datasets

To adapt for other image classification tasks:

1. Update `config.py`:
   - `num_classes`: Number of target classes
   - `image_size`: Your image dimensions
   - `num_channels`: 1 for grayscale, 3 for RGB

2. Modify `data_loader.py`:
   - Add dataset class for your data format
   - Update normalization stats if needed

3. Add new models in `modeling/models.py`:
   - Define architecture
   - Register in `get_model()` factory

## Output Files

- **Models**: `outputs/models/model_fold_{N}.pth`
- **Logs**: `outputs/logs/run_{timestamp}.log`
- **OOF Predictions**: `outputs/oof/oof_predictions.csv`
- **Submissions**: `outputs/submissions/submission_{model}_{timestamp}.csv`

## License

MIT License
