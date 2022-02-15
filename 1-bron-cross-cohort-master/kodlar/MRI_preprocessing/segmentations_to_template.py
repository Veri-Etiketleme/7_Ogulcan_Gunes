#!/bin/env python
# subject_to_template.py
#

import os
import re
import argparse

def process(segmentation_image, subject_name, transformfile, nearest_neighbor_transform_needed, output_dir, threads):	   
    # Get input subject name
    [input_dir,segmentation]=os.path.split(segmentation_image)
    segmentation_name=os.path.splitext(os.path.splitext(segmentation)[0])[0]  
    segmentation_name=segmentation_name.split(subject_name + '_')[-1]

    # Set output directories
    output_template_dir=os.path.join(output_dir, 'Template_space')
    output_dirs=[output_template_dir]
 
    subject_output_dir=os.path.join(output_template_dir,subject_name)
    output_dirs.append(subject_output_dir)
    
    segmentation_mean_dir=os.path.join(subject_output_dir, segmentation_name + '_in_template_space')    
    output_dirs.append(segmentation_mean_dir)
    
    
    for new_dir in output_dirs:
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)
        
    # Enable continuation of jobs       				
    force = False
    force_next = False
	
    # Set binaries
    bin_transformix   = 'transformix -threads'

    # Step 1: Change output image settings in inverse transform file for Nearest Neighbor Transform
    print('# Step 1: Change output image settings in inverse transform file for Nearest Neighbor Transform')
    if force_next:
        force = True   
    
    
    if not os.path.exists(transformfile):
        print('Error: ' + transformfile + 'does not exist' )
        return 0
    
    print nearest_neighbor_transform_needed
    
    transformfile_nn=''
    if nearest_neighbor_transform_needed: 
        print nearest_neighbor_transform_needed
        transformfile_nn = re.sub('InverseTransformParameters.txt', 'InverseTransformParameters.nn' + str((segmentation_image + subject_name).__hash__()) + '.txt', transformfile )	              
#        if (not os.path.exists( transformfile_nn ) ):
        with open( transformfile_nn,"w") as o:
            data = open( transformfile ).read()
            data = re.sub('ResampleInterpolator "FinalBSplineInterpolator"','ResampleInterpolator "FinalNearestNeighborInterpolator"', data)
            data = re.sub('ResultImagePixelType "float"','ResultImagePixelType "unsigned_char"', data)
 						                
            o.write( data )  

        transformfile = transformfile_nn

    # Step 2: Transform the images
    print('# Step 2: Transform the images')
    if force_next:
        force = True
	   
    print transformfile    
    segmentation_mean_space = os.path.join( segmentation_mean_dir, 'result.nii.gz' )   
    if force or (not os.path.exists( segmentation_mean_space ) ):
        force_next = True
        trans_command = '%s %i -in %s -tp %s -out %s' % (bin_transformix, threads, segmentation_image, transformfile, segmentation_mean_dir)
        print trans_command
        os.popen(trans_command)

    # Step 3: Clean-up
    print('# Step 3: Clean-up')
    if os.path.exists( segmentation_mean_space ):
        for filename in [os.path.join(segmentation_mean_dir, 'transformix.log'), transformfile_nn]:
            try:
                os.remove(filename)
            except OSError:
                pass	  
    
if __name__ == '__main__':
    #Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=unicode, help='segmentation to be transformed to mean space')
    parser.add_argument('--subject', type=unicode, help='subject name')
    parser.add_argument('--transform', type=unicode, help='transformfile, typically: /Template_space/${subject_name}/Template_registration/invert/InverseTransformParameters.txt')
    parser.add_argument('--nn', dest='nn', help='Nearest neighbor transform needed True', action='store_true')
    parser.add_argument('--no-nn', dest='nn', help='Nearest neighbor transform needed False', action='store_false')
    parser.set_defaults(nn=True)
    
    parser.add_argument('--threads', type=unicode, help='number of threads for elastix')
    parser.add_argument('--out', type=unicode, help='output directory')
    
    
    args = parser.parse_args() 

    segmentation_image=args.input
    subject_name=args.subject
    output_dir=args.out
    transformfile=args.transform
    threads=int(args.threads)
    nearest_neighbor_transform_needed=args.nn
    

    
    process(segmentation_image, subject_name, transformfile, nearest_neighbor_transform_needed, output_dir, threads)
