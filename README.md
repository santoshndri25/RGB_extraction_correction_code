# Automated RGB Extraction and Correction for Camera Calibration

## Introduction
This repository provides a Python-based implementation of a single-step RGB mapping method for color calibration of digital cameras. The code automates the process of extracting RGB values from an image of color checker chart (a device widely used in industries to solve the problem of color correction), segmenting color patches, and generating correction equations using regression models (Linear, Quadratic, and Cubic). Traditional color calibration methods using standard color checker chart requires multiple steps such as image capturing of individual shades, validation of extracted color data with the colorimeter, application of regression models or other suitable calibration models for identification of correction factor, generation of corrected color values based on coefficients of the models, and visualization of results.  
The project introduces a novel single-step calibration method designed to enhance color measurement performance of cameras. Unlike conventional multi-step methods, this approach streamlines the calibration process into a single step, making it faster and more practical for real-world applications.
The method generates and evaluates different calibration models (Linear, Quadratic, and Cubic) for RGB values, allowing flexibility in selecting the most suitable model based on accuracy. The method has been validated using different cameras with varying resolutions (2, 2, and 3 megapixels).

## Key Features
1. Segmentation of color patches for precise color measurement.  
2. Automated RGB extraction: Extracts RGB values from an image of a ColorChecker (target image).  
3. Flexible calibration models: Supports three regression models — Linear, Quadratic, and Cubic.  
4. Automatic selection of the best-performing model based on R² and RMSE.  
5. Accurate calibration: Enhances the accuracy of lightness (CIE L*) and chromatic components (CIE a*b*).  
6. Fast processing: Calibration can be completed in a very short time (1.86 to 2.08 seconds).  
7. Visual output: Displays the calibration model fit and segmented patches for visual verification.  

## Purpose
This code was developed as part of a research study to streamline the color calibration process for camera systems, ensuring accurate color measurement while reducing complexity and execution time. The method suits diverse applications, including food imaging, industrial inspection, and scientific research.

## How It Works

### Running the Code

#### Prerequisites
- Ensure you have Python installed.
- Navigate to the code directory in Bash:
  $ cd ~/RGB_extraction_correction_code
- Run the script:
  python src/single_step_chart_segmentation_rgb_extraction_and_correction.py

### 1. Image Processing
- Loads an image of a ColorChecker chart.  
- Detects the largest contour, assuming it is the ColorChecker.  
- Crops the image and segments it into multiple color patches.  
- Extracts the RGB values for each color patch and normalizes them.  

### 2. Calibration Modeling
- Loads the measured RGB values and the corresponding reference values.  
- Applies three regression models (Linear, Quadratic, and Cubic).  
- Selects the best model based on R² and RMSE.  
- Generates corrected RGB values using the best model.  

### 3. Visualization and Output
- Displays the segmented patches for verification.  
- Exports the measured and corrected RGB values to an Excel file.  

## Prerequisites
- Python 3.11.7 (Anaconda, Inc.)  
- Required Libraries:  
  1. **Standard Libraries** (No installation required):  
     - os  
     - time  
  2. **Third-Party Libraries** (Install using pip):  
     - opencv-python==4.7.0.72  
     - numpy==1.25.0  
     - pandas==2.0.1  
     - scipy==1.11.1  
     - scikit-learn==1.3.0  
     - matplotlib==3.8.0  
     - scikit-image==0.21.0  
     - openpyxl==3.1.2  

## Usage
### 1. Setup
- Clone the repository.  
- Ensure the required libraries are installed.  
- Prepare the color checker chart images for calibration.  

### 2. Run the Code
- Specify the color checker chart image path, measured RGB values file (for saving and retrieving during further operations), and reference RGB values file in the script.
- While executing code in the Bash, the generated images need to be closed to maintain code flow. These images allow the user to verify whether the code is correctly extracting the 
colors from the patches. Once confirmed, the user should close the image. In Python, these images are generated in a single step and displayed in the console. There is no need to 
close the images manually.
- Execute the script to extract RGB values and perform camera calibration.  
- The corrected RGB values will be saved as an Excel file.  

### 3. Model Selection
- The script automatically evaluates Linear, Quadratic, and Cubic models for each RGB channel.
- While executing the code in Bash, the generated graphs/ plots must be closed to maintain the code flow. These plots allow the user to verify whether the modeling is accurate. Once confirmed, the user should close the plots. 2. In Python, these plots are generated in a single step and displayed in the console. There is no need to close the plots manually 
- The model with the highest R² and lowest RMSE is selected for correction.  

## Example
- **Input:** Image of a color calibration chart and a reference file of standard RGB values.  
- **Output:** Corrected RGB values in an Excel file with a graphical display of model fits.  

## Performance
- Calibration time: 1.86 to 2.08 seconds.  
- Tested on cameras with 2, 2, and 3 megapixels.  

## License
This project is open-source and is provided under the MIT License.
