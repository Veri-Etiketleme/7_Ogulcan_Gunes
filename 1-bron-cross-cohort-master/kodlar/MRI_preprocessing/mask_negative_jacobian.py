#!/bin/env python
#
#
import os
import nibabel as nib
import numpy as np
import argparse

def mask_negative_jacobian(spatial_jacobian,mask,output_file):
    
    spatial_jacobian_nii = nib.load(spatial_jacobian)
    spatial_jacobian_data = spatial_jacobian_nii.get_data()
    
    mask_nii = nib.load(mask)
    mask_data = mask_nii.get_data()       

    mask_pos = mask_data * (spatial_jacobian_data >= 0) + 0  
  
    # Save GM modulated map
    if not os.path.exists(output_file):
        affine = mask_nii.affine
        new_nii = nib.Nifti1Image(mask_pos, affine, header=mask_nii.header.copy())
        nib.save(new_nii,output_file)
			
def main( spatial_jacobian, mask, output_file):	
        
#    spatial_jacobian = 'Y:\emc16367\ADNI\Template_space/002_S_0295_bl/Brain_image_in_template_space/spatialJacobian.nii.gz'
#    gm_density_map = 'Y:\emc16367\ADNI\Template_space/002_S_0295_bl/wc1w002_S_0295_bl_in_template_space/result.nii.gz'
#    mask = 'Y:\emc16367\ADNI\Template_space/gm_vote_in_template_space.nii.gz'
#    brain_mask = 'Y:\emc16367\ADNI\Hammers/atlas_work_temp/002_S_0295_bl/brain_mask.nii.gz'
#    
    mask_negative_jacobian(spatial_jacobian, mask, output_file)

        
			
if __name__ == '__main__':
    #Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--jac', type=unicode)
    parser.add_argument('--mask', type=unicode)
 
    parser.add_argument('--out', type=unicode, help='output file')
    
    
    args = parser.parse_args() 
    spatial_jacobian = args.jac
    mask = args.mask
    output_file = args.out
    
    main( spatial_jacobian, mask, output_file)
  