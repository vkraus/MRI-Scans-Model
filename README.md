# MRI-Scans-Model
This repository demonstrates a simple Convolutional Neural Network (CNN) model for classifying Alzheimer's disease from MRI scans. The model predicts one of four classes:

- Mild Demented
- Moderate Demented
- Non Demented
- Very Demented
It also includes a Grad-CAM visualization to highlight the most important regions in the MRI scans contributing to the model's classification.

![Obrázek gradcam](https://github.com/user-attachments/assets/78fc49ef-90c8-45aa-b49c-ab0953793cd8)


## Data 


## Preprocessing
### Parquet Loading
- The script uses pandas.read_parquet to load the training and testing data.
### Dictionary to Image
- A custom function, dict_to_image, extracts the raw bytes from each record, decodes them using OpenCV, and converts them to grayscale arrays.
### DataFrame Transformation
- After decoding, the script drops the original byte-column and replaces it with a new column named img_arr.
- Sample rows and images are visualized for sanity checks.

## Model


## Training


## GradCAM
A Grad-CAM class (GradCAM) is implemented to visualize which regions of the MRI scan the model focuses on. Key steps:

### CAM Generation: Forward pass on a chosen input and target class.
- Backpropagate gradients for that class.
- Compute a weighted average of the gradients to create a heatmap.
- Resize and overlay the heatmap onto the original image.
This helps in interpreting the model’s decision-making process by highlighting relevant regions in MRI scans.


