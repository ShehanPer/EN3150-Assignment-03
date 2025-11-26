# EN3150-Assignment-03

This repository contains notebooks and artifacts for training, fine-tuning, and evaluating convolutional neural networks (CNNs) on the RealWaste dataset. It includes custom CNN implementations, ResNet50 fine-tuning, AlexNet fine-tuning, and saved model checkpoints, along with scripts to prepare a consistent test set.

## Repository Contents


- `Model_01.ipynb`: Early experiment notebook exploring baseline models and data pipeline variants.
- `Model_02_and_ALEXNET.ipynb`: Experiments with a second baseline and AlexNet fine-tuning; compares performance and training behavior.
- `Model_03.ipynb`: Another tried version of the custom CNN experiments; builds and trains a CNN from scratch with data loading, augmentation, training loops, and evaluation utilities.
- `Fine_tuning_Resnet50.ipynb`: Fine-tunes a pretrained ResNet50 on the RealWaste dataset; includes layer freezing, head replacement, training, and evaluation.
- `model_testing.ipynb`: Focused notebook for loading saved checkpoints and running evaluation (accuracy, confusion matrix, classification report).

- `models/`:
	- `best_model_Adam.pth`, `last_model_Adam.pth`: Best and last checkpoints for the Adam run.
	- `best_model_SGD.pth`, `last_model_SGD.pth`: Best and last checkpoints for the SGD run.
	- `best_model_SGDM.pth`, `last_model_SGDM.pth`: Best and last checkpoints for the SGD with Momentum run.
	- `best_model_AlexNet_FT.pth`, `last_model_AlexNet_FT.pth`: Best and last checkpoints for AlexNet fine-tuning.
	- `best_model.pth`, `last_model.pth`: Generic/latest model artifacts from earlier experiments.
	- `training_histories.json` (optional): Serialized training curves if exported.

- `realwaste-main/`:
	- `RealWaste/`: The dataset root with class subfolders:
		- `Cardboard/`, `Food Organics/`, `Glass/`, `Metal/`, `Miscellaneous Trash/`, `Paper/`, `Plastic/`, `Textile Trash/`, `Vegetation/`
	- `test_dataset/`: Curated test split (by copying images per class) created by the notebook utility.
	- `README.md`: Dataset-specific documentation.

- `.gitignore`: Excludes dataset folders, models, `.vscode/`, and large archives (e.g., `*.zip`). Add `model_test.ipynb` here if needed.

## Data Pipeline Overview

- **Transforms**: Standard resize to `128x128`, normalization to ImageNet means/std. Augmentations include flips, color jitter, rotations (90/180/270), affine, and center crop.
- **BalancedAugmentedDataset**: Adds augmentations only to underrepresented classes until classes reach a target count (mean by default). Ensures original samples are preserved.
- **Splits**: 70% train, 15% validation, 15% test with reproducible shuffling.
- **Dataloaders**: Configured with `batch_size=32`, shuffling for train, pin memory enabled.


## Models Trained

- **Custom CNN + Adam** (`best_model_Adam.pth`)
- **Custom CNN + SGD** (`best_model_SGD.pth`)
- **Custom CNN + SGD with Momentum (Nesterov)** (`best_model_SGDM.pth`)
- **AlexNet Fine-Tuning** (`best_model_AlexNet_FT.pth`)

## Evaluation

- `evaluate_model(model, test_loader, classes)`: Prints accuracy, classification report, and plots confusion matrix; shows sample predictions.
- Notebooks provide quick reload-and-evaluate cells for each saved checkpoint.


