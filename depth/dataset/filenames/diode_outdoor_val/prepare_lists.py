import argparse
import glob

# Create an ArgumentParser object.
parser = argparse.ArgumentParser(description="Search for files with a specific pattern in a directory tree.")

# Add a positional argument for the directory path.
parser.add_argument('directory_path', help="Path to the directory to search in.")

# Parse the command-line arguments.
args = parser.parse_args()

# Use the provided directory path.
directory_path = args.directory_path
file_pattern_depth = '*_depth.npy'  # Specify your fixed pattern here

# Use the glob.glob() function to find files that match the fixed pattern.
file_list_depth = glob.glob(f'{directory_path}/*/*/{file_pattern_depth}', recursive=True)

# The 'recursive=True' argument makes sure the search is done in all subdirectories.

depth_files = []
# Print the list of matching files.
for file_name in file_list_depth:
	file_name = file_name.replace(directory_path, "")
	depth_files.append(file_name)

file_pattern_png = '*.png'  # Specify your fixed pattern here

# Use the glob.glob() function to find files that match the fixed pattern.
file_list_png = glob.glob(f'{directory_path}/*/*/{file_pattern_png}', recursive=True)

# The 'recursive=True' argument makes sure the search is done in all subdirectories.

png_files = []
# Print the list of matching files.
for file_name in file_list_png:
	file_name = file_name.replace(directory_path, "")
	png_files.append(file_name)


########################### WRITE TXT FILE ###########################
# Define the output file name
output_file = "val_list.txt"

png_files.sort()
depth_files.sort()

# Open the file for writing
with open(output_file, "w") as file:
	# Iterate through both lists and write elements in the desired format
	for png, depth in zip(png_files, depth_files):
		file.write(f"{png} {depth}\n")
