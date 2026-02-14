import os
import shutil

# Short script that organises the dataset given to us by our client into the below strucure:
# datasets
#   /Like
#       /like_1.pcap
#       /like_2.pcap
#   /Comment
#       /comment_1.pcap
#   ...

base_path = "datasets/"
folders = ["Like", "Play", "Subscribe", "Comment", "Search"]

for folder in folders:
    folder_root = os.path.join(base_path, folder)
    
    if not os.path.exists(folder_root):
        print(f"Directory not found: {folder_root}")
        continue

    print(f"Organizing {folder}")
    # Initialise a counter that will be used to rename files in each folder
    global_count = 0

    # Iterate through the subfolders
    for root, dirs, files in os.walk(folder_root):
        # Ensure only pcap files are processed
        pcap_files = []
        for f in files: 
            if f.endswith(".pcap"): 
                pcap_files.append(f) 
        pcap_files.sort()

        for filename in pcap_files:
            old_path = os.path.join(root, filename)
            
            # Create a new name
            new_name = f"{folder}_{global_count}.pcap"
            new_path = os.path.join(folder_root, new_name)

            if old_path != new_path:
                shutil.move(old_path, new_path)
                global_count += 1
    
    print(f"Successfully moved {global_count} files into /{folder}")