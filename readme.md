### MVMammo-Density Public

### Authors
Nak-Jun Sung, Donghyun Lee, Eun-Gyeong Lee, Bo Hwa Choi, Chae Jung Park
National Cancer Center Korea

### Description
- This repository provides code for training and inference of deep learning-based models that classify four breast tissue density categories (BI-RADS: A, B, C, D) from mammography images.  
It is built on PyTorch and the timm library, and includes a custom classification head that supports a variety of image classification backbones (such as efficientnet, resnet, swin, vit, etc.).

- Main features:
    - Easy selection and extensibility of diverse backbones (supports timm and torchvision)
    - Sampling options to mitigate class imbalance in the dataset
    - Automatic application of standard image preprocessing (augmentation and normalization)
    - Automatic train/validation split, and per-class weight calculation
    - Provides a custom loss function based on CrossEntropy (DistanceWeightedCrossEntropyLoss), and major evaluation metrics (accuracy, sensitivity, specificity, F1, ROC-AUC, etc.)
    - Automatic saving of best/last checkpoints and logging of major metrics during training

- Code structure:
    - model.py: Defines datasets (Dataset) and deep learning models, supporting various backbones

- Pretrained weight download link:

### Evaluation Results
The following is an example confusion matrix from our internal results:
![Confusion Matrix](./resources/confusion_matrix_internal.png)