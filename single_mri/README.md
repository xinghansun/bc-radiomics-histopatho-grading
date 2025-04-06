# Env setup
- Conda environment setup: `conda env create -f ./env/conda/nnUNetv2.yaml`
- nnUNetv2 with fixed GradScaler import problem: `pip install -e ./MAMA-MIA/nnUNet`


# Pre-trained weights
To make prediction on a new dataset, download and unzip the MAMA-MIA pre-trained weights from https://www.synapse.org/Synapse:syn61247992 and save the everything in the **full_image_dce_mri_tumor_segmentation** folder under **./MAMA-MIA/nnUNet/nnunetv2/nnUNet_results/Dataset105_full_image/nnUNetTrainer__nnUNetPlans__3d_fullres**

# Prediction 
*still having trouble on my macbook pro; not sure if there was a problem with cpu only mode*

`export nnUNet_results=./MAMA-MIA/nnUNet/nnunetv2/nnUNet_results; nnUNetv2_predict -i ./your_image_folder -o ./your_output_folder -d 105 -c 3d_fullres`
