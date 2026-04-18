import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import models
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class MammoDataset(Dataset):
    def __init__(self, samples, image_size=224, is_train=True):        
        self.samples = samples
        self.image_size = image_size
        self.is_train = is_train
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(degrees=4),
            transforms.RandomHorizontalFlip(),            
            transforms.ColorJitter(brightness=(0.9,1.2), contrast=(0.9,1.2), saturation=(0.9,1.2), hue=(-0.1,0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.42629118, 0.42629118, 0.42629118], std=[0.28209133, 0.28209133, 0.28209133]),            
        ])
        self.transform_test = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.42629118, 0.42629118, 0.42629118], std=[0.28209133, 0.28209133, 0.28209133]),            
        ])
        self.density_mapping = {
            'A': 0,
            'B': 1,
            'C': 2,
            'D': 3
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        img = Image.open(sample['IMAGE_PATH']).convert('RGB')
        if self.is_train:
            img = self.transform(img)
        else:
            img = self.transform_test(img)

        # Get label
        _type = sample['BREAST_DENSITY']
        label = self.density_mapping.get(_type, -1)  # -1 for unknown labels
        
        if label == -1:
            return img, -1
            # raise ValueError(f"Unknown type label: {_type}")
            
        return img, label

    def get_labels(self):
        return [self.density_mapping.get(s['BREAST_DENSITY'], -1) for s in self.samples]

class MammoDatasetVinDr(Dataset):
    def __init__(self, samples, image_size=224):        
        self.samples = samples
        self.image_size = image_size
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.42629118, 0.42629118, 0.42629118], std=[0.28209133, 0.28209133, 0.28209133]),            
        ])
        self.density_mapping = {
            'A': 0,
            'B': 1,
            'C': 2,
            'D': 3
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        img = Image.open(sample['IMAGE_PATH']).convert('RGB')        
        img = self.transform(img)

        # Get label
        _type = sample['breast_density']
        label = self.density_mapping.get(_type, -1)  # -1 for unknown labels
        
        if label == -1:
            raise ValueError(f"Unknown type label: {_type}")
            
        return img, label

    def get_labels(self):
        return [self.density_mapping.get(s['breast_density'], -1) for s in self.samples]



def _build_backbone(model_name, pretrained=True):
    # torchvision models
    if model_name.startswith('resnet'):
        base_model = getattr(models, model_name)(pretrained=pretrained)
        num_features = base_model.fc.in_features
        base_model.fc = nn.Identity()
    else:
        # timm models
        base_model = timm.create_model(model_name, pretrained=pretrained)
        
        if model_name.startswith('efficientnet_v2'):
            num_features = base_model.head.in_features
            base_model.head = nn.Identity()
        elif model_name.startswith('efficientnet'):
            num_features = base_model.classifier.in_features
            base_model.classifier = nn.Identity()
        elif model_name.startswith('convnext'):
            num_features = base_model.head.fc.in_features
            base_model.head.fc = nn.Identity()
        elif model_name.startswith('densenet'):
            num_features = base_model.classifier.in_features
            base_model.classifier = nn.Identity()
        elif model_name.startswith('swin'):
            num_features = base_model.head.fc.in_features
            base_model.head = nn.Identity()
            base_model.reset_classifier(0)
        elif model_name.startswith('vit'):
            num_features = base_model.head.in_features
            base_model.head = nn.Identity()
        elif model_name.startswith('regnet'):
            num_features = base_model.head.fc.in_features
            base_model.head.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported model architecture: {model_name}")

    return base_model, num_features

class MammoCNN(nn.Module):
    def __init__(self, model_name, num_classes=4, pretrained=True):
        super().__init__()

        self.model_name = model_name
        
        if model_name.startswith('resnet'):
            # ResNet models from torchvision
            self.base_model = getattr(models, model_name)(pretrained=pretrained)
            num_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()
            
        else:
            # All other models from timm
            self.base_model = timm.create_model(model_name, pretrained=pretrained)

            print(self.base_model)
            
            # Handle different model architectures
            if model_name.startswith('efficientnet_v2'):
                num_features = self.base_model.head.in_features
                self.base_model.head = nn.Identity()
                
            elif model_name.startswith('efficientnet'):
                num_features = self.base_model.classifier.in_features
                self.base_model.classifier = nn.Identity()
                
            elif model_name.startswith('convnext'):
                num_features = self.base_model.head.fc.in_features
                self.base_model.head.fc = nn.Identity()
                
            elif model_name.startswith('densenet'):
                num_features = self.base_model.classifier.in_features
                self.base_model.classifier = nn.Identity()
                
            elif model_name.startswith('swin'):
                num_features = self.base_model.head.fc.in_features                
                self.base_model.head = nn.Identity()                
                
                
            elif model_name.startswith('vit'):
                num_features = self.base_model.head.in_features
                self.base_model.head = nn.Identity()
                
            elif model_name.startswith('regnet'):
                num_features = self.base_model.head.fc.in_features
                self.base_model.head.fc = nn.Identity()

            else:
                raise ValueError(f"Unsupported model architecture: {model_name}")
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            
            nn.Linear(128, num_classes)
        )


    def forward(self, x):        
        features = self.base_model(x)

        if self.model_name.startswith('swin'):
            features = features.mean(dim=[1, 2])

        output = self.classifier(features)
        return output, features

def create_model(model_name, num_classes=4, pretrained=True):    
    return MammoCNN(model_name, num_classes, pretrained)
