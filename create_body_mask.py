import os
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import binary_fill_holes, binary_closing
from skimage.morphology import ball
from multiprocessing import Pool
import argparse

# Function to process a single eid
def process_eid(eid):
    try:
        body_mask_path = os.path.join(body_mask_total_dir, str(eid), "body_mask.nii.gz")
        if os.path.exists(body_mask_path):
            print(f"Already exists: {eid}")
            return

        os.makedirs(os.path.dirname(body_mask_path), exist_ok=True)
        print(f"Processing: {eid}")

        # Load nifti files
        nifti_img = nib.load(os.path.join(nifti_dir, str(eid), "wat.nii.gz"))
        nifti_seg = nib.load(os.path.join(total_seg_dir, str(eid), f"{eid}_total_seg.nii.gz"))

        # Convert to numpy arrays
        nifti_img_data = nifti_img.get_fdata()
        nifti_seg_data = nifti_seg.get_fdata()

        # All non-zero values to 1 in segmentation data
        nifti_seg_data[nifti_seg_data != 0] = 1

        # Create body mask based on intensity thresholding
        air_threshold = 100
        body_mask_intensity = (nifti_img_data > air_threshold).astype(np.uint8)

        # Combine the masks
        total_body_mask = np.logical_or(nifti_seg_data, body_mask_intensity).astype(np.uint8)

        # Morphological operations
        structuring_element = ball(5)
        closed_body_mask = binary_closing(total_body_mask, structure=structuring_element).astype(np.uint8)
        closed_body_mask = np.logical_or(closed_body_mask, total_body_mask).astype(np.uint8)
        closed_body_mask = binary_fill_holes(closed_body_mask).astype(np.uint8)

        # Save the result as a NIfTI file
        nifti_seg_data_mask = nib.Nifti1Image(closed_body_mask, nifti_seg.affine)
        nib.save(nifti_seg_data_mask, body_mask_path)

        print(f"Completed: {eid}")
    except Exception as e:
        print(f"Error processing {eid}: {e}")

# Main entry point
if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Process body masks for UKBB.")
    parser.add_argument("-c", "--csv", required=True, help="Path to CSV file containing 'eid' column.")
    args = parser.parse_args()

    # Constants
    total_seg_dir = "/vol/aimspace/projects/ukbb/data/whole_body/total_segmentator"
    nifti_dir = "/vol/aimspace/projects/ukbb/data/whole_body/nifti"
    body_mask_total_dir = "/vol/aimspace/projects/ukbb/data/whole_body/body_masks"

    # Read EIDs from CSV
    csv_path = args.csv
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        exit(1)

    eids_df = pd.read_csv(csv_path)
    if "eid" not in eids_df.columns:
        print(f"'eid' column not found in CSV file: {csv_path}")
        exit(1)

    eids = eids_df["eid"].tolist()

    # Create a pool of workers
    with Pool(processes=128) as pool:
        pool.map(process_eid, eids)
