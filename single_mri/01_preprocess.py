import os
import re
import sys
import pydicom

sys.path.append('./MAMA-MIA/')
from src.visualization import *
from src.preprocessing import *

import SimpleITK as sitk
import numpy as np

from collections import defaultdict


class PatientMriDcm:
""" One patient's MRI images of all phases. """
    def __init__(self, dcm_folder_dir, patient_id, source):
        self.dcm_folder_dir = dcm_folder_dir
        self.patient_id = str(patient_id)
        self.source = source
        
    def traverse_dcm(self):
    """ Check all .dcm files and organize by phase. """
        image_groups = defaultdict(list)
        self.dcm_paths = [os.path.join(self.dcm_folder_dir, f)
                          for f in os.listdir(self.dcm_folder_dir) 
                          if f.endswith((".DCM", ".dcm"))]
        
        for path in self.dcm_paths:
            dcm = pydicom.dcmread(path)
            series_desc = dcm.get("SeriesDescription", "Unknown")
            instance_num = dcm.get("InstanceNumber", 0)
            image_groups[series_desc].append(
                (instance_num, path))
        self.dcm_dict = image_groups
    
    def convert_to_nii(self, out_dir):
    """ Convert .dcm files of the same phase to one .nii file. """
        nii_dict = {}
        for k, v in self.dcm_dict.items():
            # keep only the phase images
            if re.match("(Ph\d+\/)?Dyn Ax Vibrant\+C", k):
                if k.startswith("Ph"):
                    phase = re.search(r'Ph(\d+)/', k).group(1)
                else:
                    phase = "0"
                print(f"--- Phase {phase}; {len(v)} DCM files.")
                
                # this will reorder v by instance_num
                v.sort(key = lambda x: x[0])
                reader = sitk.ImageSeriesReader()
                reader.SetFileNames([x[1] for x in v])
                image = reader.Execute()
                out_nii_folder = (f"{out_dir}/{self.source}"
                                  f"{self.patient_id.zfill(3)}")
                os.makedirs(out_nii_folder, exist_ok = True)
                out_nii_path = os.path.join(
                    out_nii_folder,
                    (f"{self.source}{self.patient_id.zfill(3)}"
                     f"_{phase.zfill(4)}.nii.gz"))
                sitk.WriteImage(image, out_nii_path)
                print(f"Saved: {out_nii_path}.\n")
                nii_dict[int(phase)] = out_nii_path
        self.nii_path_list = [nii_path for _, nii_path in sorted(nii_dict.items())]
        
    def preprocess(self, out_dir=None):
    """ Preprocess specific to DCE-MRI. Note co-registration may pop warnings. """
        print("--- Loading nii images.")
        images = [sitk.ReadImage(path, sitk.sitkFloat32) for path in self.nii_path_list]
        print("--- Co-registering all phases.")
        coregistered_images = PatientMriDcm.coregister_images(images)
        print("--- Applying z-score normalization across all phases.")
        normalized_images = PatientMriDcm.zscore_normalization(coregistered_images)
        print("--- Resampling images.\n")
        resampled_images = PatientMriDcm.resample_images(normalized_images)
        
        self.preprocessed_nii = resampled_images
        
        if out_dir is not None:
            out_nii_folder = (f"{out_dir}/{self.source}"
                              f"{self.patient_id.zfill(3)}")
            os.makedirs(out_nii_folder, exist_ok = True)
            for phase, img in enumerate(self.preprocessed_nii):
                out_nii_path = os.path.join(
                    out_nii_folder,
                    (f"{self.source}{self.patient_id.zfill(3)}"
                     f"_{str(phase).zfill(4)}.nii.gz"))
                print(f"--- Saving preprocessed phase image to {out_nii_path}.")
                sitk.WriteImage(img, out_nii_path)
            
    @staticmethod
    def coregister_images(phase_images):
  	""" Co-register images from different phases using phase-0 as reference image,
  	and the rest as moving images. 
  	"""
        return [phase_images[0]] + [
            PatientMriDcm.register_image(
                phase_images[0], 
                img) 
            for img in phase_images[1:]
        ]
    
    @staticmethod
    def register_image(fixed, moving):
        registration = sitk.ImageRegistrationMethod()
        registration.SetMetricAsMeanSquares()
        registration.SetInterpolator(sitk.sitkLinear)
        registration.SetOptimizerAsGradientDescent(learningRate=0.1, 
                                                   numberOfIterations=300)
        registration.SetInitialTransform(sitk.AffineTransform(3))
        transform = registration.Execute(fixed, moving)
        
        registration.SetInitialTransform(transform, inPlace=False)
        registration.SetInterpolator(sitk.sitkLinear)
        final_transform = registration.Execute(fixed, moving)
        registered_image = sitk.Resample(moving, 
                                         fixed, 
                                         final_transform, 
                                         sitk.sitkLinear, 
                                         0.0, 
                                         moving.GetPixelID())
        return registered_image
    
    @staticmethod
    def zscore_normalization(phase_images):
    """ Z-score normalization across all phases instead each phase independently. """
        arrays = [sitk.GetArrayFromImage(img) for img in phase_images]
        stacked_array = np.stack(arrays, axis=0)
        
        global_mean = np.mean(stacked_array)
        global_std = np.std(stacked_array)
        
        normalized_arrays = [(arr-global_mean) / global_std 
                                 for arr in arrays]
        normalized_images = [sitk.GetImageFromArray(arr) 
                                 for arr in normalized_arrays]
        
        # Preserve metadata
        for i, img in enumerate(images):
            normalized_images[i].CopyInformation(img)

        return normalized_images
    
    @staticmethod
    def resample_images(phase_images):
    """ Resample voxels to [1, 1, 1] spacing. """
        return [resample_sitk(img, 
                              new_spacing=[1, 1, 1], 
                              interpolator=sitk.sitkBSpline)
                for img in phase_images
               ]


class Dataset:
""" Collection of MRI patient images from the same project under the same path. """
    def __init__(self, dataset_root, name="BC_"):
        self.root = dataset_root
        self.name = name
        self.scan()
    
    def scan(self):
    """ Scan dataset_root folder to collect and organize patient MRI images. """
        self.patients = [
            (
                os.path.join(self.root, folder), 
                re.match("(\d+).*MR", folder).group(1)
            ) 
            for folder in os.listdir(self.root) 
            if re.match("\d+.*MR.*", folder)
        ]
        
        # Folder with MR data for each patient
        self.mr_dcm_dir = {
            id_: os.path.join(
                dir_, 
                [i for i in os.listdir(dir_) 
                    if i.startswith("MR")
                ][0])
            for dir_, id_ in self.patients
        }
        
        self.patient_mri_dcm_list = [
            PatientMriDcm(final_dir, id_, self.name)
            for id_, dir_ in self.mr_dcm_dir.items()
        ]
    
    def convert_to_nii(self, out_dir="./nii"):
        for dcm in self.patient_mri_dcm_list:
            dcm.traverse_dcm()
            dcm.convert_to_nii(out_dir)
        
    def preprocess(self, out_dir="./nii_preprocessed"):
        for dcm in self.patient_mri_dcm_list:
            print(f"Preprocessing for patient {dcm.patient_id}.")
            dcm.preprocess(out_dir=out_dir)
        

# Test
if __name__ == "__main__":
	root = "./test_data/"
	ds = Dataset(root)
	ds.convert_to_nii(out_dir="./nii")
	ds.preprocess(out_dir="./nii_preprocessed")





