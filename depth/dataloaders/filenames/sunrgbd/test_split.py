import argparse
import os
# Create an ArgumentParser object.
#parser = argparse.ArgumentParser(description="Search for files with a specific pattern in a directory tree.")

# Add a positional argument for the directory path.
#parser.add_argument('directory_path', help="Path to the directory to search in.")

# Parse the command-line arguments.
#args = parser.parse_args()
# Initialize an empty list to store the paths
img_paths = []
depth_paths = []

root_dir = "/home/aradhye/kstyles/MDP_CVPR/datasets/sunrgbd"

# Open the text file for reading
with open('sunrgbd_pixelformer.txt', 'r') as file:
    # Iterate through each line in the file
    for line in file:
        # Split each line by space and take the first part as the path
        parts = line.strip().split()
        if len(parts) >= 1:
            img_path = parts[0]
            img_path = "downloaded/" + img_path
            img_path = img_path.replace("//", "/")
            img_path_actual = root_dir + "/" + img_path
            img_paths.append(img_path)
        depth_path = img_path
        depth_path_actual = img_path_actual
        depth_path = depth_path.replace("image", "depth")
        depth_path_actual = depth_path_actual.replace("image", "depth")
        if not os.path.exists(depth_path_actual):
            depth_path = depth_path.replace("jpg", "png")
            depth_path_actual = depth_path_actual.replace("jpg", "png")
            if not os.path.exists(depth_path_actual):
                depth_path = depth_path.replace(".png", "_abs.png")
                depth_path_actual = depth_path_actual.replace(".png", "_abs.png")
                if not os.path.exists(depth_path_actual):
                    directory_actual = depth_path_actual.rsplit('/', 1)[0]
                    diff_named_file = os.listdir(directory_actual)[0]
                    directory_path = depth_path.rsplit('/', 1)[0]
                    depth_path = directory_path + '/' + diff_named_file

        depth_paths.append(depth_path)

########################### WRITE TXT FILE ###########################
# Define the output file name
output_file = "sunrgbd2.txt"

depth_paths.sort()
img_paths.sort()

# Open the file for writing
with open(output_file, "w") as file:
	# Iterate through both lists and write elements in the desired format
	for png, depth in zip(img_paths, depth_paths):
		file.write(f"{png} {depth}\n")
