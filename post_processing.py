import pandas as pd
import matplotlib.pyplot as plt

# When reloading, fill NaN with a default value or drop again
ped_data_loaded = pd.read_csv('pedestrian_data.csv')
ped_data_loaded.fillna(0, inplace=True)  # Or use another appropriate value

# Load and print the first 5 rows (with all columns)
ped_data = pd.read_csv('pedestrian_data.csv')
print(ped_data.head())

# Get unique values of 'num_of_classes' and 'num_peds_in_class'
unique_classes = ped_data['num_of_classes'].unique()
unique_peds = ped_data['num_peds_in_class'].unique()

# Create subplots with rows for each 'num_of_classes' and columns for 'num_peds_in_class'
fig, axes = plt.subplots(len(unique_classes), len(unique_peds), figsize=(15, 10))

# Iterate over each 'num_of_classes' and 'num_peds_in_class' to create histograms
for i, num_classes in enumerate(unique_classes):
    for j, num_peds in enumerate(unique_peds):
        ax = axes[i, j]  # Current subplot

        # Filter data for the current 'num_of_classes' and 'num_peds_in_class'
        class_peds_data = ped_data[(ped_data['num_of_classes'] == num_classes) &
                                   (ped_data['num_peds_in_class'] == num_peds)]

        # Plot histogram for simulation time if there is data
        ax.hist(class_peds_data['simulation_time'], bins=10, edgecolor='black', color='blue')

        # Set title for the top row and labels for the left column
        ax.set_title(f'{num_peds} peds / {num_classes} classes')
        ax.set_ylabel(f'Classes: {num_classes}')

        # Set common x and y labels
        ax.set_xlabel('Simulation Time')
        ax.set_ylabel('Frequency')

# Add a global title to the entire figure
fig.suptitle('Evacuation Time Histograms by Number of Classes and Pedestrians', fontsize=16)

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0, 1, 0.97])  # Add extra space for the global title

plt.savefig('evacuation_histogram.png')

# Show the plot
plt.show()

# Load your waiting data
waiting_data = pd.read_csv('waiting_data.csv')

# Get unique values of 'num_of_classes' and 'num_peds_in_class'
unique_classes = waiting_data['num_of_classes'].unique()
unique_peds = waiting_data['num_peds_in_class'].unique()

# Create subplots with rows for each 'num_of_classes' and columns for 'num_peds_in_class'
fig, axes = plt.subplots(len(unique_classes), len(unique_peds), figsize=(15, 10))

# Iterate over each 'num_of_classes' and 'num_peds_in_class' to create histograms
for i, num_classes in enumerate(unique_classes):
    for j, num_peds in enumerate(unique_peds):
        ax = axes[i, j]  # Current subplot

        # Filter data for the current 'num_of_classes' and 'num_peds_in_class'
        class_peds_data = waiting_data[(waiting_data['num_of_classes'] == num_classes) &
                                       (waiting_data['num_peds_in_class'] == num_peds)]

        # Plot histogram for waiting peds if there is data
        ax.hist(class_peds_data['waiting_peds'], bins=10, edgecolor='black', color='green')

        # Set title for the top row and labels for the left column
        ax.set_title(f'Peds: {num_peds}')
        ax.set_ylabel(f'Classes: {num_classes}')

        # Set common x and y labels
        ax.set_xlabel('Waiting Peds')
        ax.set_ylabel('Frequency')

# Add a global title to the entire figure
fig.suptitle('Waiting Pedestrians Histograms by Number of Classes and Pedestrians', fontsize=16)

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0, 1, 0.97])  # Add extra space for the global title

# Save the plot as a file (e.g., .png or .jpg)
fig.savefig('waiting_peds_histograms.png', bbox_inches='tight')

# Show the plot
plt.show()