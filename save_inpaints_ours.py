import os
from PIL import Image

# Create the directory '../inpaint_selected_results' if it doesn't exist
selected_results_dir = '../inpaint_selected_results'
os.makedirs(selected_results_dir, exist_ok=True)

# Iterate over the directories in '../inpaint_results/'
inpaint_results_dir = '../inpaint_results/'
for dir_name in os.listdir(inpaint_results_dir):
    dir_path = os.path.join(inpaint_results_dir, dir_name)
    
    # Check if the current item in the directory is a directory itself
    if os.path.isdir(dir_path):
        # Find 'inpainted_with_mask_2.png' inside the current directory
        file_to_rename = os.path.join(dir_path, 'inpainted_with_mask_2.png')
        
        # Rename the file to 'X.jpg' and move it to '../inpaint_selected_results'
        if os.path.isfile(file_to_rename):
            new_file_name = os.path.join(selected_results_dir, dir_name + '.jpg')
            # Open the PNG image
            image = Image.open(file_to_rename)
            # Convert and save it as JPEG
            image.convert('RGB').save(new_file_name, 'JPEG')
            
            # Remove the original PNG file
            os.remove(file_to_rename)
            
            print(f"Converted and moved {file_to_rename} to {new_file_name}")