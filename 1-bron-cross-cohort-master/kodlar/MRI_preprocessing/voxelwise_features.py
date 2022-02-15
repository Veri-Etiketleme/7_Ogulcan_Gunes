#!/bin/env python
#
# Output: voxel-based morphometry feature map = 
# Multiplication of
# 1. spatial Jacobian map of brain
# 2. GM density map
# 3. GM vote mask (all voxels that have GM in any template subject)
# 4. Divided by intracranial volume
#
import os
import nibabel as nib
import numpy as np
import argparse
from scipy.stats import zscore

def compute_icv(brain_mask):
    #Computes ICV in mm3
    
    brain_mask_nii = nib.load(brain_mask)
    brain_data = brain_mask_nii.get_data()
    brain_voxels=np.sum(brain_data)
    voxel_size=np.prod(np.asarray(brain_mask_nii.header.get_zooms()))
    icv=np.multiply(brain_voxels,voxel_size)
    
    return icv

def compute_vbm(spatial_jacobian,gm_density_map,mask,icv,output_dir):
    
    spatial_jacobian_nii = nib.load(spatial_jacobian)
    spatial_jacobian_data = spatial_jacobian_nii.get_data()
    
    spatial_jacobian_pos = spatial_jacobian_data * (spatial_jacobian_data >= 0) + 0 
    
    gm_density_map_nii = nib.load(gm_density_map)
    gm_density_map_data = gm_density_map_nii.get_data()   
 
    mask_nii = nib.load(mask)
    mask_data = mask_nii.get_data()       
    
    icv_map = np.ones(np.shape(spatial_jacobian_data))/icv
		
    vbm_map=np.prod([spatial_jacobian_pos, gm_density_map_data, mask_data, icv_map],axis=0)  + 0

    vbm_map_file=os.path.join(output_dir, 'gmModulatedJacobian.nii.gz')
    

    # Save GM modulated map
    if not os.path.exists(vbm_map_file):
        affine = spatial_jacobian_nii.affine
        vbm_nii = nib.Nifti1Image(vbm_map, affine, header=spatial_jacobian_nii.header.copy())
        nib.save(vbm_nii,vbm_map_file)

def compute_t1wmodulated(spatial_jacobian,t1w_image,mask,output_dir):
    
    spatial_jacobian_nii = nib.load(spatial_jacobian)
    spatial_jacobian_data = spatial_jacobian_nii.get_data()
    
    spatial_jacobian_pos = spatial_jacobian_data * (spatial_jacobian_data >= 0) + 0 
    
    t1w_nii = nib.load(t1w_image)
    t1w_data = t1w_nii.get_data()   
 
    mask_nii = nib.load(mask)
    mask_data = mask_nii.get_data()     
    
    t1w_data_masked = np.ma.masked_array(t1w_data, mask=np.logical_not(mask_data).astype(int))
    t1w_data_scaled = zscore(t1w_data_masked).data
	
    t1_scaled_file = os.path.join(output_dir, 'T1w_scaled.nii.gz')	
    
    t1_mod = np.prod([spatial_jacobian_pos, t1w_data_scaled, mask_data],axis=0)  + 0

    t1_mod_file = os.path.join(output_dir, 'T1wModulatedJacobian.nii.gz')

    # Save T1w modulated image & T1w scaled image
    if not os.path.exists(t1_mod_file):
        affine = spatial_jacobian_nii.affine
        t1_mod_nii = nib.Nifti1Image(t1_mod, affine, header=spatial_jacobian_nii.header.copy())
        nib.save(t1_mod_nii,t1_mod_file)
        
        t1_scaled_nii = nib.Nifti1Image(t1w_data_scaled, affine, header=spatial_jacobian_nii.header.copy())
        nib.save(t1_scaled_nii,t1_scaled_file)	
			
def main( brain_mask, spatial_jacobian, t1w_image, gm_density_map, mask, output_dir):	
            
    icv = compute_icv(brain_mask)
    compute_vbm(spatial_jacobian,gm_density_map, mask, icv,output_dir)
    compute_t1wmodulated(spatial_jacobian, t1w_image, mask, output_dir)
      
			
if __name__ == '__main__':
    #Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--brain', type=unicode)
    parser.add_argument('--jac', type=unicode)
    parser.add_argument('--t1w', type=unicode)
    parser.add_argument('--gm', type=unicode)
    parser.add_argument('--mask', type=unicode)
 
    parser.add_argument('--out', type=unicode, help='output directory')
    
    
    args = parser.parse_args() 
    brain_mask = args.brain
    spatial_jacobian = args.jac
    t1w_image = args.t1w
    gm_density_map = args.gm
    mask = args.mask
    output_dir = args.out
    
    main( brain_mask, spatial_jacobian, t1w_image, gm_density_map, mask, output_dir)
  