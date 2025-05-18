## Automated RGB Extraction and Correction for Camera Calibration

This repository offers a practical solution for color calibration of cameras. It automates the entire process— segmenting the image into distinct color patches from an image of a ColorChecker chart, extracting RGB values from segmented color patches, and applying correction equations using three regression models namely linear, quadratic, and cubic.
Unlike traditional multi-step approaches, this approach simplifies calibration into a single, efficient process. The tool is validated across various camera resolutions (2, 2, and 3 megapixels), achieving calibration within 1.86 to 2.08 seconds. The method ensures accurate color measurement, particularly in lightness (CIE L*) and chromatic components (CIE a*b*), making it valuable for diverse applications like food imaging, industrial inspection, and scientific research.
The script evaluates each calibration model and selects the one with the best performance (highest R² and lowest RMSE) without manual intervnetion. This means you always get the most accurate results without any manual tuning. Visual outputs for segmented patches and model fits further enhance transparency and verification.

## Main Features:
• Automatically extracts RGB values from a ColorChecker chart image.
• Supports three regression models (linear, quadratic, and cubic) for calibration.
• Automatically selects the best model using R² and RMSE.
• Provides fast calibration (1.86 to 2.08 seconds) with clear visual outputs.
• Supports flexible use across different cameras (2, 2, and 3 megapixels).


This project is open-source and licensed under the MIT License.
