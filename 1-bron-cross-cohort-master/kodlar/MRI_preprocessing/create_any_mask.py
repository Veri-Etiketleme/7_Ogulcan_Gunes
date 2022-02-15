#!/bin/env python
# transform_cbf_t1wspace.py

import os
import argparse
import nibabel as nib
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser(
            description='Make OR-mask: a voxel is True if it is True in any of the inputs.',
    )
    parser.add_argument('--input', type=str, nargs='+', required=True, help='Input images')
    parser.add_argument('--out', type=str, required=True, help='Output filename')
    parser.add_argument('--value', type=int, required=False, default=1, help='Value used for masking')
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    input_files = args.input
    target_file = args.out
    value = args.value
	
    if os.path.exists(target_file):
        print('Output file exists already, nothing done')
        return
        
    
    for input_file in input_files:	
        print input_file
        loaded_file = nib.load(input_file)	
        data = loaded_file.get_data()

        if not'mask' in locals():
            mask = data
        else: 
            # add new data to mask
            mask = np.array([mask, data==value])
            # threshold mask
            mask=np.any(mask,0).astype(int)
            
    print mask
    
    # Save mask
    mask_image = nib.Nifti1Image(mask, loaded_file.affine, header=loaded_file.header.copy())
    nib.save(mask_image,target_file)


if __name__ == '__main__':
    main()

