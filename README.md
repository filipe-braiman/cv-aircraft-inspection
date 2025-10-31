# Aircraft Damage Classification & Captioning

A deep learning project that classifies aircraft damage into "dent" or "crack" categories using a VGG16-based model and generates descriptive captions using a BLIP transformer model for automated aircraft inspection.

*This project represents an enhanced version of coursework from the IBM AI Engineering Professional Certificate, which I'm currently enrolled in.*

## Project Overview

This project addresses the critical need for automated aircraft damage detection in the aviation industry. Traditional manual inspection methods are time-consuming and prone to human error. This solution leverages computer vision and natural language processing to:

- **Classify** aircraft damage types (dent vs. crack) using transfer learning with VGG16
- **Generate captions** and summaries of damage using BLIP transformer model
- Provide an end-to-end automated damage assessment system

## Project Structure

```
cv-aircraft-inspection/
├── Aircraf_DentCrack_Classifier.h5          # Trained classification model
├── Aircraft_Damage_Classification+Captioning.ipynb  # Main Jupyter notebook
├── requirements.txt                         # Python dependencies
├── Dataset/
│   └── aircraft_damage_dataset_v1.tar      # Original dataset archive
└── aircraft_damage_dataset_v1/
    ├── train/                              # Training images
    │   ├── dent/
    │   └── crack/
    ├── valid/                              # Validation images
    │   ├── dent/
    │   └── crack/
    └── test/                               # Test images
        ├── dent/
        └── crack/
```

## Features

### Classification Module
- **Model Architecture**: VGG16 base with custom classification head
- **Input Size**: 224×224×3 RGB images
- **Custom Layers**: Two dense layers (512 units) with Dropout (0.3)
- **Output**: Binary classification (sigmoid activation)
- **Performance**: Achieved 84.38% test accuracy in dent vs. crack classification

### Captioning Module
- **Model**: BLIP (Bootstrapping Language-Image Pre-training) transformer
- **Capabilities**: Generates descriptive captions and summaries for aircraft damage images
- **Applications**: Automated damage reporting and documentation

## Dataset

The project uses the [Aircraft Damage Dataset](https://universe.roboflow.com/youssef-donia-fhktl/aircraft-damage-detection-1j9qk) from Roboflow with CC BY 4.0 license.

**Dataset Statistics:**
- Training: 300 images (2 classes)
- Validation: 96 images (2 classes)
- Testing: 50 images (2 classes)
- Classes: Dent, Crack

## Installation & Setup

### Prerequisites
- Python 3.7+
- TensorFlow 2.x
- Jupyter Notebook

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/filipe-braiman/cv-aircraft-inspection.git
cd cv-aircraft-inspection
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Extract the dataset**
```bash
# The dataset will be automatically extracted when running the notebook
# Or manually extract:
tar -xf Dataset/aircraft_damage_dataset_v1.tar
```

## Usage

### Running the Classification Model

1. **Open the Jupyter notebook:**
```bash
jupyter notebook Aircraft_Damage_Classification+Captioning.ipynb
```

2. **Execute cells in order:**
   - Data preprocessing and augmentation
   - Model definition and training
   - Performance evaluation
   - Prediction visualization

### Model Training Parameters
- **Batch Size**: 32
- **Epochs**: 10
- **Image Size**: 224×224
- **Optimizer**: Adam (lr=0.0001)
- **Loss Function**: Binary Crossentropy

### Generating Captions

The BLIP model automatically generates captions for classified images, providing detailed descriptions of the detected damage.

## Results

### Classification Performance
- Training Accuracy: 96.33%
- Validation Accuracy: 76.04%
- **Test Accuracy: 84.38%**
- Robust performance on dent vs. crack classification

### Sample Outputs
- **Classification**: "Dent"
- **Caption**: "Aircraft surface showing circular dent with light reflection"

## Technical Details

### Model Architecture

The classification model is built using transfer learning with the following architecture:

**Architecture Summary:**
1. **Feature Extraction**: Pre-trained VGG16 (frozen weights)
2. **Flatten Layer**: Convert feature maps to 1D vector
3. **Dense Layer**: 512 units with ReLU activation
4. **Dropout**: 0.3 regularization
5. **Dense Layer**: 512 units with ReLU activation  
6. **Dropout**: 0.3 regularization
7. **Output Layer**: 1 unit with Sigmoid activation for binary classification

### Data Preprocessing
- Image resizing to 224×224
- Pixel normalization (0-1 scaling)
- Data augmentation for improved generalization

## Applications

- **Aviation Maintenance**: Automated damage inspection
- **Insurance Claims**: Objective damage assessment
- **Quality Control**: Manufacturing defect detection
- **Safety Compliance**: Regular maintenance documentation

## Author
**Filipe B. Carvalho**

**Email:** [filipebraiman@gmail.com](mailto:filipebraiman@gmail.com)  
**LinkedIn:** [linkedin.com/in/filipe-b-carvalho](https://www.linkedin.com/in/filipe-b-carvalho)  
**GitHub:** [github.com/filipe-braiman](https://github.com/filipe-braiman)  

### About Me  
Data and AI professional with experience in **AI model evaluation, data annotation, and NLP projects**, currently contributing to AI initiatives at **Huawei**. Skilled in **Python, SQL, data visualization, machine learning, AI, and dataset building**, and certified through the **IBM Data Science Professional Certificate**. Multilingual in **Portuguese, English, Spanish, Turkish, and French**, bringing a linguistic and analytical perspective to data-driven problem solving. Passionate about leveraging data and AI to create practical, high-impact solutions.

---

### Version History

| Version | Date       | Changes                         |
|:--------:|:-----------|:--------------------------------|
| 1.0      | 2025-10-29 | First publication of the notebook |