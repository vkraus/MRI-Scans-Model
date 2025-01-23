# MRI-Scans-Model
This repository demonstrates a simple Convolutional Neural Network (CNN) model for classifying Alzheimer's disease from MRI scans. It also includes a Grad-CAM visualization to highlight the most important regions in the MRI scans contributing to the model's classification.

![Obrázek gradcam](https://github.com/user-attachments/assets/78fc49ef-90c8-45aa-b49c-ab0953793cd8)

## Maintainers
Lucie Semeráková (seml04@vse.cz)  
Martina Maslíková (masm35@vse.cz)  
Štěpán Fojtík (fojs00@vse.cz)  
Ivana Žůrková (zuri00@vse.cz)  
Vojta Kraus (xkrav00@vse.cz)  
Tomáš Hejl (hejt02@vse.cz)  


## Project
This project consists of 2 primary notebooks
- **main.ipynb** is used to:
    - Load
    - Create model
    - Train
    - Evaluate
    - Create GradCAM and apply it on existing MRI image.

- **models_compare.ipynb** is primarly used to evaluate existing models that we as a group tried to implement and rank them, how well they scored.

## Data 
MRI scans images are in the parquet format and we had to change it into other forms, that are usable and understandable for CNN to successfully train. Parquet data example:

![parquetData](https://github.com/user-attachments/assets/f410115c-1a9e-4473-9146-01910bb251cc)

Classes
  - Mild Demented
  - Moderate Demented
  - Non Demented
  - Very Demented

## Preprocessing
Based on the type of dataset we got, the first step is to preprocess it... **tbc**

### Parquet Loading
- The script uses pandas.read_parquet to load the training and testing data.
### Dictionary to Image
- A custom function, dict_to_image, extracts the raw bytes from each record, decodes them using OpenCV, and converts them to grayscale arrays.
### DataFrame Transformation
- After decoding, the script drops the original byte-column and replaces it with a new column named img_arr.
- Sample rows and images are visualized for sanity checks.

## Models

### Model Descriptions

**BaselineCNN**:  
A baseline CNN (`BaselineCNN`) with two convolutional layers, batch normalization, max pooling, and fully connected layers, designed for general image classification tasks.

---

**BaselineCNN_WeightedRandomSample**:  
An extended CNN (`BaselineCNN_WeightedRandomSample`) that incorporates weighted random sampling to prioritize specific classes during training.

---

**BaselineCNN_HEM**:  
A variant of the baseline CNN (`BaselineCNN_HEM`) designed to emphasize hard example mining (HEM), allowing the model to focus on more challenging samples.

---

**BaselineCNN_CE**:  
An enhanced CNN (`BaselineCNN_CE`) with added dropout layers for improved generalization and reduced overfitting during training.

---

**BaselineCNN_CE_Dropped**:  
A modified version of `BaselineCNN_CE` (`BaselineCNN_CE_Dropped`) that removes certain dropout layers, aiming to retain feature learning capacity and improve accuracy.

---

**BaselineCNN_CE_Drop01**:  
A refined version of the dropout-enhanced CNN (`BaselineCNN_CE_Drop01`) with minimal dropout probability (`p=0.1`), balancing generalization and overfitting.

---

**BaselineCNN_CE_ConvAdd**:  
An advanced CNN (`BaselineCNN_CE_ConvAdd`) that extends `BaselineCNN_CE` with an additional convolutional layer for better feature extraction and classification performance.

---

**BaselineCNN_CE_ConvAdded_DropMinus**:  
The most advanced CNN (`BaselineCNN_CE_ConvAdded_DropMinus`) in the baseline family, featuring multiple convolutional layers and no dropout, achieving superior accuracy.

---

**DifferentOdBaseline**:  
A simplified CNN (`DifferentOdBaseline`) with fewer parameters, making it ideal for smaller datasets or faster training needs.

---

**StudyModel_1**:  
A dual-path architecture (`StudyModel_1`) combining two parallel convolutional networks that merge their outputs for advanced feature extraction and classification.

---

**KerasLikeModel**:  
A CNN (`KerasLikeModel`) inspired by Keras-style sequential networks, featuring multiple convolutional layers, batch normalization, and dropout layers for robust and flexible classification.

---

**ModelOdGPT**:  
A moderately complex CNN (`ModelOdGPT`) with three convolutional layers and dropout, designed with GPT assistance for general-purpose tasks.

---

**AdvancedCNN**:  
A ResNet-50-based model (`AdvancedCNN`) fine-tuned for grayscale image classification, with a modified first convolutional layer and a frozen backbone for transfer learning.


## Training the Models

This code defines a training pipeline for a neural network model using a custom dataset and dataloader. It implements a simple training loop with support for tracking losses and updating model parameters using backpropagation.

---

### Key Steps:
1. **Dataset and DataLoader**:  
   A custom dataset (`ImageDataset`) is wrapped into a DataLoader for batch processing.
   
2. **Training Loop**:  
   - Computes the loss using `CrossEntropyLoss`.
   - Updates the model parameters using an optimizer (`AdamW` in this case).
   - Tracks and averages the loss over epochs.

3. **Model Initialization**:  
   A baseline CNN (`BaselineCNN`) is instantiated and trained over the specified number of epochs.

---

### Example Code:
```python
# Initialize dataset and dataloader
train_dataset = ImageDataset(df_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Train the model
model = BaselineCNN()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
model, train_losses = train_model(model, train_loader, optimizer)
```

## Grad-CAM Implementation for Visualizing Neural Networks

This implementation demonstrates the use of **Grad-CAM** (Gradient-weighted Class Activation Mapping) to interpret the decision-making process of convolutional neural networks during image classification. Grad-CAM highlights the regions in an input image that are most relevant for predicting a specific class, making it particularly useful for medical imaging applications, such as classifying the stages of dementia using MRI scans.

---

### How It Works

1. **Layer Hooking**:
   The `GradCAM` class hooks into a specified layer of the model (e.g., a convolutional layer) to capture both the activations and gradients during forward and backward passes.
   
2. **Generating Grad-CAM Heatmaps**:
   - The target class output is used to compute gradients with respect to the activations of the selected layer.
   - A weighted average of the gradients is applied to the activations to produce a class-specific activation map.
   - The resulting heatmap is normalized and optionally adjusted for class-specific weighting.

3. **Visualizing Grad-CAM Results**:
   - The Grad-CAM heatmap is overlaid onto the original image using a color map, providing a clear visual representation of the regions critical for the model's classification.

---

### Class-Specific Weighting

Each class (e.g., *Very Demented*, *Mild Demented*, etc.) is assigned a weight to enhance or dampen the importance of the regions associated with that class. These weights help tailor the visualization to specific use cases and emphasize relevant features.

---

### Example Usage

This implementation includes a demonstration of Grad-CAM applied to real-world data:

1. **Single Image Example**
   - Generate a Grad-CAM heatmap for a single input image and overlay it onto the original image.

2. **Batch Visualization**
   - Test Grad-CAM on multiple randomly selected images from the dataset and visualize their results in a grid format.

---

### Output Examples

- **Original Image with Grad-CAM Heatmap**:
  The code overlays the Grad-CAM activation map (in a `jet` colormap) on top of the grayscale input image.

- **Batch Heatmap Visualization**:
  A set of images from the test dataset, each showing its corresponding Grad-CAM overlay with the predicted class label.

---

### Key Features

- Modular implementation with the `GradCAM` class, allowing it to be reused with any PyTorch model.
- Support for class-specific enhancements through configurable weights.
- Easy visualization of Grad-CAM overlays for both individual and batch inputs.

---

### Example Code Snippet

```python
# Initialize Grad-CAM for a specific convolutional layer
grad_cam = GradCAM(model, model.conv2)

# Generate Grad-CAM for a single test image
input_image, label = test_dataset[0]
input_tensor = input_image.unsqueeze(0).to(device)
true_class = label.item()
cam_map = grad_cam.generate_cam(input_tensor, true_class)

# Visualize the heatmap overlay
plt.imshow(input_image.squeeze(0).cpu(), cmap='gray')
plt.imshow(cam_map, cmap='jet', alpha=0.5)
plt.title('Grad-CAM Overlay')
plt.show()
```

## Evaluation

