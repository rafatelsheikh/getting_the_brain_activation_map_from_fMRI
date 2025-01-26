import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn.plotting import plot_stat_map
from nilearn.image import mean_img, resample_to_img
import matplotlib.pyplot as plt

# Define task details: onsets, durations, and conditions (excluding white noise)
task_data = pd.DataFrame({
    "onset": [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360, 390, 420, 450, 480, 510, 540, 570],
    "duration": [30] * 20,
    "condition": ["calm", "white_noise", "afraid", "white_noise", "delighted", "white_noise",
                  "depressed", "white_noise", "excited", "white_noise", "delighted", "white_noise",
                  "depressed", "white_noise", "calm", "white_noise", "excited", "white_noise",
                  "afraid", "white_noise"]
})

# Define paths (update these paths to your local dataset)
data_dir = r"E:\EECE\2nd year\1st term\Neuro-Notes\fMRI\Dataset3"
subject_id = "02"
func_file = r"E:\EECE\2nd year\1st term\Neuro-Notes\fMRI\Dataset3\sub-22\func\sub-22_task-fe_bold.nii"
resting_state_file = r"E:\EECE\2nd year\1st term\Neuro-Notes\fMRI\Dataset3\sub-22\func\sub-22_task-rest_bold.nii"
anat_file = r"E:\EECE\2nd year\1st term\Neuro-Notes\fMRI\Dataset3\sub-22\anat\sub-22_T1w.nii"

# Load fMRI data
func_img = nib.load(func_file)
resting_state_img = nib.load(resting_state_file)

# Define repetition time (TR) and high-pass filter cutoff
TR = 2.0  # Adjust based on your data's TR
high_pass = 1 / 128  # Standard high-pass filter cutoff for fMRI (Hz)


# Create a design matrix (excluding white noise)
def create_design_matrix(task_data, n_scans, TR):
    frame_times = np.arange(n_scans) * TR  # Time for each frame
    events = task_data[task_data["condition"] != "white_noise"]  # Exclude white_noise

    # Rename the 'condition' column to 'trial_type' to match expected input
    events = events.rename(columns={"condition": "trial_type"})

    # Create the design matrix
    design_matrix = make_first_level_design_matrix(
        frame_times, events=events, hrf_model='glover', drift_model='cosine'
    )
    return design_matrix


# Fit the GLM with the functional image
def fit_glm(func_img, task_data, TR, high_pass):
    n_scans = func_img.shape[-1]
    design_matrix = create_design_matrix(task_data, n_scans, TR)

    # GLM Model
    glm = FirstLevelModel(t_r=TR, slice_time_ref=0.5, high_pass=high_pass)
    glm = glm.fit(func_img, design_matrices=design_matrix)

    return glm, design_matrix


# Plot the activation map
def plot_activation_map(z_map, anat_file, title):
    if np.sum(np.isnan(z_map.get_fdata())) > 0:
        print(f"Warning: The contrast map for {title} contains NaN values!")
    else:
        print(f"Plotting activation map for {title}.")
        plot_stat_map(z_map, bg_img=anat_file, title=title, display_mode='ortho', draw_cross=True)
        plt.show()


# Main processing loop
glm, design_matrix = fit_glm(func_img, task_data, TR, high_pass)

# Loop through conditions and compute contrasts
for condition in task_data["condition"].unique():
    if condition != "white_noise":  # Skip white noise
        print(f"Processing condition: {condition}")

        # Check if the condition exists in the design matrix columns
        if condition in design_matrix.columns:
            # Compute contrast directly for the condition
            try:
                z_map = glm.compute_contrast(condition, output_type='z_score')

                if z_map is None or np.isnan(z_map.get_fdata()).all():
                    print(f"No meaningful signal detected for condition '{condition}'. Skipping.")
                    continue

                # Plot the activation map
                plot_activation_map(z_map, anat_file, f"Activation Map - {condition}")
            except Exception as e:
                print(f"Error processing condition '{condition}': {e}")

# Plot the resting-state contrast
print("Processing resting-state data...")

# Load the mean resting-state image for the baseline
mean_rest_img = mean_img(resting_state_img)

# Resample to match the functional data dimensions and plot
resampled_rest_img = resample_to_img(mean_rest_img, func_img, interpolation='nearest')

# Plot the resting-state activation map (it should be baseline)
plot_activation_map(resampled_rest_img, anat_file, "Resting-State Activation Map")