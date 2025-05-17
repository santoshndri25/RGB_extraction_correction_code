"""
Created on Mon Dec 16 10:17:16 2024

@author: santosh chopde
"""

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from skimage.color import rgb2lab
import time
start_time = time.time() 
# Part 1: Extract RGB values from the image and display segmented patches
def extract_rgb_and_save(image_path, output_file):
    # Load the original image
    image = cv2.imread(image_path)

    # Check if the image was loaded correctly
    if image is None:
        print("Error: Image not loaded correctly. Please check the file path.")
        return

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use GaussianBlur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Use morphological operations to close gaps between edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Assume the largest contour is the color checker
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Crop the image using the bounding box
        cropped_image = image[y:y + h, x:x + w]

        # Define parameters for patch segmentation
        rows, cols = 4, 6
        rebgp, ribgp, cebgp, cibgp = 75, 75, 110, 80
        patch_height = (cropped_image.shape[0] - 2 * rebgp - 3 * ribgp) // rows
        patch_width = (cropped_image.shape[1] - 2 * cebgp - 5 * cibgp) // cols

        rgb_values = []
        patches = []  # Store patches for visualization

        for i in range(rows):
            for j in range(cols):
                top_left_x = j * patch_width + cebgp + j * cibgp
                top_left_y = i * patch_height + rebgp + i * ribgp
                bottom_right_x = (j + 1) * patch_width + cebgp + j * cibgp
                bottom_right_y = (i + 1) * patch_height + rebgp + i * ribgp

                top_left_x = max(0, top_left_x)
                top_left_y = max(0, top_left_y)
                bottom_right_x = min(cropped_image.shape[1], bottom_right_x)
                bottom_right_y = min(cropped_image.shape[0], bottom_right_y)

                patch = cropped_image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
                avg_color = np.mean(patch, axis=(0, 1))[::-1]  # Convert to RGB
                rgb_values.append(avg_color)
                patches.append(patch)

        rgb_values = np.array(rgb_values).astype(int)
        # Normalize RGB values for conversion to Lab
        normalized_rgb = np.clip(rgb_values / 255.0, 0, 1)  # Normalize to [0, 1]
        lab_values = rgb2lab(normalized_rgb)
        # Create a DataFrame to store RGB and Lab values
        df = pd.DataFrame({
            'Measured R': rgb_values[:, 0],
            'Measured G': rgb_values[:, 1],
            'Measured B': rgb_values[:, 2],
            'Measured L*': lab_values[:, 0],
            'Measured a*': lab_values[:, 1],
            'Measured b*': lab_values[:, 2]
        })
        df.to_excel(output_file, index=False)
        print(f"RGB values extracted and saved to {output_file}")

        # Display the cropped image and segmented patches
        display_cropped_and_patches(cropped_image, patches, rows, cols)
    else:
        print("No contours found in the image.")

# Function to display the cropped image and segmented patches
def display_cropped_and_patches(cropped_image, patches, rows, cols):
    fig, axes = plt.subplots(rows + 1, cols, figsize=(12, 8))
    fig.suptitle("Cropped Image and Segmented Color Patches", fontsize=16)

    # Display the cropped image
    axes[0, 0].imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Cropped Image")
    axes[0, 0].axis('off')

    # Display the segmented patches
    for idx, patch in enumerate(patches):
        row = (idx // cols) + 1
        col = idx % cols
        axes[row, col].imshow(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
        axes[row, col].set_title(f"Patch {idx + 1}")
        axes[row, col].axis('off')

    # Hide any remaining axes
    for ax in axes.flatten():
        if not ax.images:
            ax.axis('off')

    plt.tight_layout()
    plt.show()

# Part 2: Perform modeling and data correction
def model_and_correct_data(measured_file_path, reference_file_path):
    measured_df = pd.read_excel(measured_file_path)
    reference_df = pd.read_excel(reference_file_path)

    parameters = ['R','G','B']
    corrected_values = {f'Corrected {param}': [] for param in parameters}

    for param in parameters:
        x_data = measured_df[f'Measured {param}'].values
        y_data = reference_df[f'Reference {param}'].values

        def Linear(x, m, c): return m * x + c
        def Quadratic(x, a, b, c): return a * x**2 + b * x + c
        def Cubic(x, a, b, c, d): return a * x**3 + b * x**2 + c * x + d

        function_generators = [Linear, Quadratic, Cubic]

        best_model = None
        best_r2 = float('-inf')
        best_rmse = float('inf')
        best_params = None
        best_func_name = None

        for func in function_generators:
            try:
                params, _ = curve_fit(func, x_data, y_data)
                y_pred = func(x_data, *params)

                # Calculate R² and RMSE
                r2 = r2_score(y_data, y_pred)
                rmse = np.sqrt(mean_squared_error(y_data, y_pred))

                # Print results
                if func == Linear:
                    print(f"{param} - Linear model: m={params[0]:.4f}, c={params[1]:.4f}")
                elif func == Quadratic:
                    print(f"{param} - Quadratic model: a={params[0]:.4f}, b={params[1]:.4f}, c={params[2]:.4f}")
                elif func == Cubic:
                    print(f"{param} - Cubic model: a={params[0]:.4f}, b={params[1]:.4f}, c={params[2]:.4f}, d={params[3]:.4f}")

                print(f"R²={r2:.4f}, RMSE={rmse:.4f}")

                # Check if this model is the best so far
                if r2 > best_r2 or (r2 == best_r2 and rmse < best_rmse):
                    best_model = func
                    best_r2 = r2
                    best_rmse = rmse
                    best_params = params
                    best_func_name = func.__name__

                # Plot the fit
                plt.scatter(x_data, y_data, label='Original Data')
                plt.plot(x_data, y_pred, label=f"{func.__name__} Fit", color='red')
                plt.xlabel('Measured')
                plt.ylabel('Reference')
                plt.legend()
                plt.show()

            except RuntimeError as e:
                print(f"Error fitting {func.__name__} for {param}: {e}")

        # Generate corrected values using the best model
        if best_model is not None:
            corrected_values[f'Corrected {param}'] = np.round(best_model(x_data, *best_params), 2)
            print(f"Best model for {param}: {best_func_name}")
            print(f"Parameters: {best_params}")
            print(f"R²={best_r2:.4f}, RMSE={best_rmse:.4f}")

    # Create a DataFrame for corrected values and save to an Excel file
    corrected_df = pd.DataFrame(corrected_values)
    corrected_output_file = measured_file_path.replace("Measured_values", "Corrected_values")
    corrected_df.to_excel(corrected_output_file, index=False)
    print(f"Corrected RGB values saved to {corrected_output_file}")


# Execution
image_path = "data/input/test_image.jpg"
output_measured_file = "data/output/test_measured_rgb_values.xlsx"
reference_file_path = "data/input/test_reference_rgb_values.xlsx"

# Run the functions
extract_rgb_and_save(image_path, output_measured_file)
model_and_correct_data(output_measured_file, reference_file_path)
end_time = time.time()
# Calculate and display the elapsed time
elapsed_time = end_time - start_time
print(f"Total execution time: {elapsed_time:.2f} seconds")
