import os
import sys
sys.path.append("./MAMA-MIA/")

from src.visualization import *
from src.preprocessing import read_mri_phase_from_patient_id, read_segmentation_from_patient_id

preprocessed_dir = "./nii_preprocessed"
predicted_dir = "./predicted"

out_dir = "./tumour_extracted"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    
padding = 50

patients = [f for f in os.listdir(predicted_dir) if f != ".DS_Store"]

for patient in patients:
    print(f"--- Processing patient: {patient}.")
    files = sorted([f for f in os.listdir(os.path.join(preprocessed_dir, patient)) if f.endswith(".nii.gz")])
    
    for file in files:
        stem = file.split(".")[0]
        print(f"-- Processing phase file: {file}.")
        img_sitk = sitk.ReadImage(os.path.join(preprocessed_dir, patient, file), sitk.sitkFloat32)
        mask_sitk = sitk.ReadImage(os.path.join(predicted_dir, patient, file), sitk.sitkUInt8)
        
        [x_min, y_min, z_min, x_max, y_max, z_max] = get_segmentation_bounding_box(mask_sitk, margin=0)
        print('- Bounding box from predicted segmentation:')
        print(f'  X: [{x_min}, {x_max}]')
        print(f'  Y: [{y_min}, {y_max}]')
        print(f'  Z: [{z_min}, {z_max}]')
        
        # Extract cubic voxel centering on tumor
        x_c = (x_min + x_max) // 2
        y_c = (y_min + y_max) // 2
        z_c = (z_min + z_max) // 2
        x_min_new, x_max_new = x_c - padding, x_c + padding
        y_min_new, y_max_new = y_c - padding, y_c + padding
        z_min_new, z_max_new = z_c - padding, z_c + padding
        
        tumor_box = img_sitk[x_min_new:x_max_new, y_min_new:y_max_new, z_min_new:z_max_new]
        print('- New bounding box:')
        print(f'  X: [{x_min_new}, {x_max_new}]')
        print(f'  Y: [{y_min_new}, {y_max_new}]')
        print(f'  Z: [{z_min_new}, {z_max_new}]')
        
        # save plot of the extracted img for inspection
        display_slices = [tumor_box.GetSize()[0] // 2, 
                          tumor_box.GetSize()[1] // 2, 
                          tumor_box.GetSize()[2] // 2]
        image_array = sitk.GetArrayFromImage(tumor_box)
        slice_xy = image_array[:, :, display_slices[0]]
        slice_xz = image_array[:, display_slices[1], :]
        slice_yz = image_array[display_slices[2], :, :]
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Patient ID: BC_100", fontsize=12)
        ax[0].imshow(slice_xy, cmap='gray')
        ax[1].imshow(slice_xz, cmap='gray')
        ax[2].imshow(slice_yz, cmap='gray')
        plt.savefig(os.path.join(out_dir, stem+".png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # save tumor box img
        tumor_box.SetSpacing(img_sitk.GetSpacing())
        tumor_box.SetOrigin(img_sitk.GetOrigin())
        tumor_box.SetDirection(img_sitk.GetDirection())
        #TODO: copy timestamp
        sitk.WriteImage(tumor_box, os.path.join(out_dir, stem+".nii.gz"))
