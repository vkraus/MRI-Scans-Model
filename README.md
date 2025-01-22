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
- **main.ipynb** is used to Load - Create model - Train - Evaluate - Apply GradCAM and apply it on existing MRI image.
- **models_compare.ipynb** is used to evaluate existing models that we as a group tried to implement and rank them, how well they scored in this classification task.

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


## GradCAM
A Grad-CAM class (GradCAM) is implemented to visualize which regions of the MRI scan the model focuses on. Key steps:

### CAM Generation: Forward pass on a chosen input and target class.
- Backpropagate gradients for that class.
- Compute a weighted average of the gradients to create a heatmap.
- Resize and overlay the heatmap onto the original image.
This helps in interpreting the model’s decision-making process by highlighting relevant regions in MRI scans.


