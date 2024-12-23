import numpy as np
from astropy.io import fits
from sklearn.cluster import KMeans
from astropy.visualization import ZScaleInterval
from scipy import ndimage
import os
import csv

def findBoundingBox(filePath):
    with fits.open(filePath) as hdul:
        # Assuming the image data is in the primary HDU
        img_data = hdul[0].data
    
        # Replace NaNs and infinite values with zeros
        img_data = np.nan_to_num(img_data, nan=0.0, posinf=0.0, neginf=0.0)
    
        zscale = ZScaleInterval()
        z1, z2 = zscale.get_limits(img_data)
    
        # Clip the data to the ZScale range
        clipped_data = np.clip(img_data, z1, z2)
    
        normalized_data = (clipped_data - z1) / (z2 - z1)

    pixels = normalized_data.reshape((-1, 1))
    
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=10, random_state=42)
    kmeans.fit(pixels)

    centers = kmeans.cluster_centers_.flatten()
    labels = kmeans.labels_  
    labeled_image = labels.reshape(img_data.shape)
    
    # Find the darkest cluster (lowest center value)
    darkest_cluster = np.argmax(centers)
    dark_mask = labeled_image == darkest_cluster
    labeled_mask, num_features = ndimage.label(dark_mask)
    
    # Find the largest connected region
    if num_features > 0:
        region_sizes = ndimage.sum(dark_mask, labeled_mask, range(1, num_features + 1))
        largest_region_label = np.argmax(region_sizes) + 1
        largest_dark_region = labeled_mask == largest_region_label
    else:
        print("num features less 0")
        largest_dark_region = np.zeros_like(dark_mask)
    
    rows, cols = np.where(largest_dark_region)
    
    # Find bounding box of the largest dark region
    box_noise = 50
    
    if np.any(largest_dark_region):
        top = max(0, np.min(rows) - box_noise)
        bottom = min(img_data.shape[0] - 1, np.max(rows) + box_noise)
        left = max(0, np.min(cols) - box_noise)
        right = min(img_data.shape[1] - 1, np.max(cols) + box_noise)
        bbox = (int(top), int(bottom), int(left), int(right))
    else:
        print("Empty box")
        top, bottom, left, right = 0, 0, 0, 0
        bbox = (int(top), int(bottom), int(left), int(right))

    return img_data, bbox




def process_fits_in_subfolders(root_folder):

    # Loop through each sub-folder in the root folder
    for subfolder in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder)
        files = [""]
        if os.path.isdir(subfolder_path):
            files = []
            i = 0
            for filename in os.listdir(subfolder_path):
                if filename.endswith('.fits'):
                    filepath = os.path.join(subfolder_path, filename)
                    files.append(filepath)
                    i+=1
            if len(files) == 4:
                # Get bounding boxes for band 3 and 4
                band3 = ""
                band4 = ""
                for i in range(len(files)):
                    if ('-w3-' in files[i]):
                        band3 = files[i]
                    if ('-w4-' in files[i]):
                        band4 = files[i]
                img_data3, bbox_band3 = findBoundingBox(band3)
                img_data4, bbox_band4 = findBoundingBox(band4)
                # Calculate the total bounding box that encompasses both band 3 and band 4 bounding boxes
                total_bbox = (
                    min(bbox_band3[0], bbox_band4[0]),
                    max(bbox_band3[1], bbox_band4[1]),
                    min(bbox_band3[2], bbox_band4[2]),
                    max(bbox_band3[3], bbox_band4[3])
                )
                csv_filepath = os.path.join(subfolder_path, "bounding_box.csv")
                temp_bbox = [total_bbox[2], total_bbox[0], total_bbox[3], total_bbox[1]]
                # Write the bounding box to the CSV file
                with open(csv_filepath, mode="w", newline="") as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(["xmin", "ymin", "xmax", "ymax"])  # Header row
                    writer.writerow(temp_bbox)
                    print("Wrote box to" + csv_filepath)
                
            
if __name__ == "__main__":
    root_folder = "/Volumes/NODE NAME/aFile-part1/organized/testing"  # REPLACE WITH CORRECT PATH
    process_fits_in_subfolders(root_folder)
    print("finished processing")


