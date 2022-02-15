#!/bin/env python
# subject_to_MNI.py
#
# subject_to_MNI.process
# 1. Subject rough to MNI
# 2. Update transform file
# 3. Transform scan

import os
import re
import argparse
from errno import ENOENT

def process(input_image, input_brain_mask, mni_brain_mask, output_dir, affine_transform, threads):
    # Get input subject name
    [input_dir,subject_image]=os.path.split(input_image)
        
    # Set binaries
    bin_elastix       = 'elastix -threads' 
    bin_transformix   = 'transformix -threads'
	
    # Set Elastix parameter files for registration
    dir_params      = '/home/ebron/params'
	
    par_file_sim  = 'par_atlas_sim_checknrfalse.txt'
    par_file_aff  = 'par_atlas_aff_checknrfalse.txt'
#    par_file_aff  = 'par_atlas_aff_checknrfalse_output.txt'
    par_file_sim_mask  = 'par_atlas_sim_nn_checknrfalse.txt'
    par_file_aff_mask  = 'par_atlas_aff_nn_checknrfalse.txt'

	
    # Create absolute path names
    par_file_sim = os.path.join( dir_params, par_file_sim )
    par_file_aff = os.path.join( dir_params, par_file_aff )
    par_file_sim_mask = os.path.join( dir_params, par_file_sim_mask )
    par_file_aff_mask = os.path.join( dir_params, par_file_aff_mask )
	       
    # Step 1: Register brain mask of templates with MNI
    print('# Step 1: Register brain mask of templates with MNI')
    if affine_transform:
        output_transform = os.path.join(output_dir, 'TransformParameters.1.txt')
    else:
        output_transform = os.path.join(output_dir, 'TransformParameters.0.txt')
  
    if not os.path.exists( output_transform ) :      
        if affine_transform:
            elx_command = '%s %i -f %s -m %s -p %s -p %s -out %s'% (bin_elastix, threads, mni_brain_mask, input_brain_mask, par_file_sim_mask, par_file_aff_mask, output_dir)
        else:
            elx_command = '%s %i -f %s -m %s -p %s -out %s'% (bin_elastix, threads, mni_brain_mask, input_brain_mask, par_file_sim_mask, output_dir)
        print elx_command
        os.popen(elx_command)	
        
    # Clean-up  
    if os.path.exists(output_transform):
        filenames=['elastix.log','IterationInfo.0.R0.txt','IterationInfo.0.R1.txt','IterationInfo.0.R2.txt','IterationInfo.1.R0.txt','IterationInfo.1.R1.txt','IterationInfo.1.R2.txt','IterationInfo.2.R0.txt','IterationInfo.2.R1.txt','IterationInfo.2.R2.txt']
        for filename in [os.path.join(output_dir, f) for f in filenames]:
            try:
                os.remove(filename)
            except OSError:
                pass				
 
    # Step 2: Change interpolation in transform
    print('# Step 2: Change interpolation in transform')
    if affine_transform:
        output_transform = os.path.join(output_dir, 'TransformParameters.1.txt')
    else:
        output_transform = os.path.join(output_dir, 'TransformParameters.0.txt')
 
    image_transform = os.path.join(output_dir, 'ImageTransformParameters.txt')

    if not os.path.exists( image_transform) :
        o = open( image_transform ,"w")
        data = open( output_transform ).read()    
        data = re.sub('ResampleInterpolator "FinalNearestNeighborInterpolator"','ResampleInterpolator "FinalBSplineInterpolator"', data)
        o.write( data )
        o.close()
           		    
    # Step 3: Transform the image
    print('# Step 3: Transform the image')
    transformed_image = os.path.join( output_dir, 'result.nii.gz' ) 
    if not os.path.exists( transformed_image ):
        trans_command = '%s %i -in %s -tp %s -out %s' % (bin_transformix, threads, input_image, image_transform, output_dir)
        print trans_command
        os.popen(trans_command)
        
    # Clean-up
    if os.path.exists( output_dir ):
        filenames=['transformix.log']
        for filename in [os.path.join(output_dir, fn) for fn in filenames]:
            try:
                os.remove(filename)
            except OSError:
                pass			
            
if __name__ == '__main__':
#    from bigr.environmentmodules import EnvironmentModules
#
#    #Environment modules
#    elastix='elastix/4.8' 
#    matlab='matlab/R2013b'
#    mcr='mcr/R2013b'
#    itktools='itktools'
#
#    env = EnvironmentModules()
#   
#    env.unload(matlab)
#    env.unload(mcr)
#    env.load(elastix)
#    env.load(itktools)	
    
    #Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=unicode, help='input image file')
    parser.add_argument('--brain', type=unicode, help='brain mask file for input image')
    parser.add_argument('--mni', type=unicode, help='mni brain mask, e.g. /cm/shared/apps/fsl/5.0.2.2/data/standard/MNI152_T1_1mm_brain_mask.nii.gz')
    parser.add_argument('--out', type=unicode, help='output directory')
    parser.add_argument('--aff', dest='affine_transform', help='Affine transformation (default)', action='store_true')
    parser.add_argument('--sim', dest='affine_transform', help='Similarity transformation', action='store_false')
    parser.add_argument('--threads', type=unicode, help='number of threads for elastix')
    parser.set_defaults(affine_transform=True)
    
    
    args = parser.parse_args() 

    input_image=args.input
    output_dir=args.out
    input_brain_mask=args.brain
    threads=int(args.threads)
    mni_brain_mask = args.mni
    affine_transform = args.affine_transform
    
    process(input_image, input_brain_mask, mni_brain_mask, output_dir, affine_transform, threads)