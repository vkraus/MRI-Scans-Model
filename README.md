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

## Model


## Training


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


