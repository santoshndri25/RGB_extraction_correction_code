Automated RGB Extraction and Correction for Camera Calibration

Introduction
This repository provides a Python-based implementation of a single-step RGB mapping method for color calibration of digital cameras. The code is designed to automate the process of extracting RGB values from an image, segmenting color patches, and generating correction equations using regression models (Linear, Quadratic, and Cubic).
This project provides a novel single-step calibration method designed to enhance the color measurement performance of cameras. Unlike conventional multi-step methods, this approach streamlines the calibration process into a single step, making it faster and more practical for real-world applications.
The method generates and evaluates different calibration models (Linear, Quadratic, and Cubic) for RGB values, allowing for flexibility in selecting the most suitable model based on accuracy. The method has been validated using different cameras with varying resolutions (2, 2, and 3 megapixels).

Key Features
1.Segmentation of color patches for precise color measurement.
2.Automated RGB extraction: Extracts RGB values from an image of a ColorChecker (target image).
3.Flexible calibration models: Supports three regression models Linear, Quadratic, and Cubic models for correction.
4.Automatic selection of the best-performing model based on R² and RMSE.
5.Accurate calibration: Enhances the accuracy of lightness (CIE L*) and chromatic components (CIE a*b*).
6.Fast Processing: Calibration can be completed in a very short time (1.86 to 2.08 seconds).
7.Visual Output: Displays the calibration model fit and segmented patches for visual verification.

Purpose:
This code was developed as part of a research study to streamline the color calibration process for camera systems, ensuring accurate color measurement while reducing complexity and execution time. The method is suitable for diverse applications, including food imaging, industrial inspection, and scientific research.

How It Works
1.Image Processing:
	a)Loads an image of a ColorChecker chart.
	b)Detects the largest contour, assuming it is the ColorChecker.
	c)Crops the image and segments it into multiple color patches.
	d)Extracts the RGB values for each color patch and normalizes them.
2.Calibration Modeling:
	a)Loads the measured RGB values and the corresponding reference values.
	b)Applies three regression models (Linear, Quadratic, and Cubic).
	c)Selects the best model based on R² and RMSE.
	d)Generates corrected RGB values using the best model.
3.Visualization and Output:
	a)Displays the segmented patches for verification.
	b)Exports the measured and corrected RGB values to an Excel file.

Prerequisites
- Python 3.11.7 (Anaconda, Inc.)
- Required Libraries:
  1. Standard Libraries (No installation required):
    a) os
    b) time
  2. Third-Party Libraries (Install using pip):
    a) opencv-python==4.7.0.72
    b) numpy==1.25.0
    c) pandas==2.0.1
    d) scipy==1.11.1
    e) scikit-learn==1.3.0
    f) matplotlib==3.8.0
    g) scikit-image==0.21.0
Usage
1. Setup
	•Clone the repository.
	•Ensure the required libraries are installed.
	•Prepare the color checker chart images for calibration.
2. Run the Code
	•Specify the color checker chart image path, measured RGB values file (for saving and retrieving during the further operations), and reference RGB 
	  values file in the script.
	•Execute the script to extract RGB values and perform camera calibration.
	•The corrected RGB values will be saved as an Excel file.
3. Model Selection
	• The script automatically evaluates Linear, Quadratic, and Cubic models for each RGB channel.
	• The model with the highest R² and lowest RMSE is selected for correction.
	Example
	• Input: Image of a color calibration chart and a reference file of standard RGB values.
	• Output: Corrected RGB values in an Excel file with a graphical display of model fits.
Performance
	• Calibration time: 1.86 to 2.08 seconds.
	• Tested on cameras with 2, 2, and 3 megapixels. 

License
This project is open-source and is provided under the MIT License.

