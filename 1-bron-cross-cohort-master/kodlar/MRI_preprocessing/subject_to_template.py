#!/bin/env python
# subject_to_template.py
#
# subject_to_template.process
# 0. Subjects roughly to MNI
# 1. Subjects to templates pairwise registration
# 2. Get mean
# 3. Get deformation field
# 4. Get inverse transformation
# 5. Update transform file
# 6. Transform brain and scan
# 7. Transform segmentations

import os
import subprocess
import re
import math
import inspect
import argparse
from errno import ENOENT

def get_boundingbox( bin_bbox, filename ):
    command = [bin_bbox, '-in', filename]
    proc = subprocess.Popen( command, stdout=subprocess.PIPE )
    result = proc.communicate()[0]

    min_point = re.search( "MinimumPoint = \[(-?[\d\.]*), (-?[\d\.]*), (-?[\d\.]*)\]", result).groups()
    max_point = re.search( "MaximumPoint = \[(-?[\d\.]*), (-?[\d\.]*), (-?[\d\.]*)\]", result).groups()

    min_point = map(lambda x: float(x), min_point )
    max_point = map(lambda x: float(x), max_point )

    if min_point[0] > max_point[0]:
        t = min_point[0]
        min_point[0] = max_point[0]
        max_point[0] = t
    if min_point[1] > max_point[1]:
        t = min_point[1]
        min_point[1] = max_point[1]
        max_point[1] = t
    if min_point[2] > max_point[2]:
        t = min_point[2]
        min_point[2] = max_point[2]
        max_point[2] = t

    return (min_point, max_point)

def get_all_mixtures( in1, in2 ):
    if len(in1) is not len(in2):
        raise ValueError('Length of arrays to mix must be equal')

    N = len( in1 )
    paired = (in1, in2)
    
    output = []
    for k in range(0, int( math.pow(2, N) ) ):
        suboutput = []
        
        for m in range(0, N):
            suboutput.append( paired[ (k >> m) & 1 ][m] )

        output.append( suboutput )

    return output

def get_image_corners( bin_bbox, filename ):
    command = [bin_bbox, '-in', filename]
    proc = subprocess.Popen( command, stdout=subprocess.PIPE )
    result = proc.communicate()[0]

    min_index = re.search( "MinimumIndex = \[(-?[\d]*), (-?[\d]*), (-?[\d]*)\]", result).groups()
    max_index = re.search( "MaximumIndex = \[(-?[\d]*), (-?[\d]*), (-?[\d]*)\]", result).groups()

    min_index = map(lambda x: float(x), min_index )
    max_index = map(lambda x: float(x), max_index )
        
    return get_all_mixtures( min_index, max_index )

def get_image_info( bin_iinfo, filename ):
    # Run the pxgetimageinformation command
	command = [bin_iinfo, '-all', '-in', filename]
	proc = subprocess.Popen( command, stdout=subprocess.PIPE )
	
	result = proc.communicate()[0] 
		    
    # Run regular expressions
	size      = re.search( "size:\s*\(([\d]+), ([\d]+), ([\d]+)\)", result ).groups()
	spacing   = re.search( "spacing:\s*\(([\d\.]*), ([\d\.]*), ([\d\.]*)\)", result ).groups()
	origin    = re.search( "origin:\s*\((-?[\d\.]*), (-?[\d\.]*), (-?[\d\.]*)\)", result ).groups()
	direction = re.search( "direction:\s*\((-?[\d\.]*e?-?[\d]?[\d]?[\d]?), (-?[\d\.]*e?-?[\d]?[\d]?[\d]?), (-?[\d\.]*e?-?[\d]?[\d]?[\d]?), (-?[\d\.]*e?-?[\d]?[\d]?[\d]?), (-?[\d\.]*e?-?[\d]?[\d]?[\d]?), (-?[\d\.]*e?-?[\d]?[\d]?[\d]?), (-?[\d\.]*e?-?[\d]?[\d]?[\d]?), (-?[\d\.]*e?-?[\d]?[\d]?[\d]?), (-?[\d\.]*e?-?[\d]?[\d]?[\d]?)\)", result ).groups()
	#direction = re.search( "direction:\s*\((-?[\d\.]*), (-?[\d\.]*), (-?[\d\.]*), (-?[\d\.]*), (-?[\d\.]*), (-?[\d\.]*), (-?[\d\.]*), (-?[\d\.]*), (-?[\d\.]*)\)", result ).groups()

    # Convert to numerics
	size      = map(lambda x: int(x), size )
	spacing   = map(lambda x: float(x), spacing )
	origin    = map(lambda x: float(x), origin )
	direction = map(lambda x: float(x), direction )
    
	return (size, spacing, origin, direction)

def process(input_image, input_brain_mask, list_of_template_images, template_brain_masks, mni_brain_mask, output_dir, threads, register_templates_mni):
    # Get input subject name
    [input_dir,subject_image]=os.path.split(input_image)
    subject_name=os.path.splitext(os.path.splitext(subject_image)[0])[0]
    
    # Get names template subjects
    template_names = [];
    for template_image in list_of_template_images:
        [template_dir,template_image]=os.path.split(template_image)
        template_name=os.path.splitext(os.path.splitext(template_image)[0])[0]
        template_names.append(template_name)

    if not input_brain_mask:
        hammers_dir=os.path.join( output_dir, 'Hammers', 'atlas_work_temp') 
        input_brain_mask = os.path.join( hammers_dir, subject_name, 'brain_mask.nii.gz' )
        template_brain_masks = []
        for l in template_names: 
            template_brain_masks.append(os.path.join( hammers_dir, l, 'brain_mask.nii.gz'))            
    
    # Set output directories
    output_template_dir=os.path.join(output_dir, 'Template_space')
    output_dirs=[output_template_dir]
 
    subject_output_dir=os.path.join(output_template_dir,subject_name)
    output_dirs.append(subject_output_dir)
    
    subject_registration_dir=os.path.join(subject_output_dir, 'Template_registration')
    output_dirs.append(subject_registration_dir)
    subject_subject_dir=os.path.join(subject_registration_dir,subject_name)
    output_dirs.append(subject_subject_dir)
    subject_invert_dir = os.path.join( subject_registration_dir, 'invert' )
    output_dirs.append(subject_invert_dir)
    subject_mean_dir=os.path.join(subject_output_dir, 'Image_in_template_space')    
    output_dirs.append(subject_mean_dir)
    brain_mean_dir=os.path.join(subject_output_dir, 'Brain_image_in_template_space')    
    output_dirs.append(brain_mean_dir)      
    
    template_settings_dir=os.path.join(output_template_dir,'Template_settings')
    output_dirs.append(template_settings_dir)
        
    for l in template_names:
        template_subject_output_dir=os.path.join(output_template_dir,l)
        output_dirs.append(template_subject_output_dir)
        template_mni_dir=os.path.join(template_subject_output_dir,'MNI')
        output_dirs.append(template_mni_dir)
        
#        template_registration_dir=os.path.join(template_subject_output_dir, 'Template_registration')
#        output_dirs.append(template_registration_dir)
#        template_invert_dir = os.path.join( template_registration_dir, 'invert' )
#        output_dirs.append(template_invert_dir)        
        
#        template_settings_template_dir=os.path.join(template_settings_dir, l)
#        output_dirs.append(template_settings_template_dir)  
        subject_template_dir=os.path.join(subject_registration_dir,l)
        output_dirs.append(subject_template_dir)      
      
    for new_dir in output_dirs:
        try:
            os.mkdir(new_dir)
        except OSError:
            pass
        
    # Enable continuation of jobs       				
    force = False
    force_next = False
	
    # Set binaries
    bin_iinfo         = 'pxgetimageinformation'
    bin_bbox          = 'pxcomputeboundingbox'
    bin_elastix       = 'elastix -threads' 
    bin_transformix   = 'transformix -threads'
    bin_im			  = 'pxbinaryimageoperator'
	
    # Set Elastix parameter files for registration
    dir_params      = '/home/ebron/params'
	
    par_file_sim  = 'par_atlas_sim_checknrfalse.txt'
    par_file_aff  = 'par_atlas_aff_checknrfalse.txt'
    par_file_sim_mask  = 'par_atlas_sim_nn_checknrfalse.txt'
    par_file_aff_mask  = 'par_atlas_aff_nn_checknrfalse.txt'
    #par_file_bsp  = 'par_atlas_bsp.txt'
    par_file_bsp  = 'par_atlas_bsp_grid_checknrfalse.txt'
    #par_file_inv  = 'par_atlas_invert.txt'
    par_file_inv  = 'par_atlas_invert_grid.txt'
    par_file_dum  = 'par_dummy.txt'
    par_file_rig  = 'par_atlas_rigid_checknrfalse.txt'
    par_file_rig_mask  = 'par_atlas_rigid_nn_checknrfalse.txt'
	
	
    # Create absolute path names
    par_file_sim = os.path.join( dir_params, par_file_sim )
    par_file_aff = os.path.join( dir_params, par_file_aff )
    par_file_bsp = os.path.join( dir_params, par_file_bsp )
    par_file_dum = os.path.join( dir_params, par_file_dum )
    par_file_inv = os.path.join( dir_params, par_file_inv )
    par_file_rig = os.path.join( dir_params, par_file_rig )
    par_file_rig_mask = os.path.join( dir_params, par_file_rig_mask )
    par_file_sim_mask = os.path.join( dir_params, par_file_sim_mask )
    par_file_aff_mask = os.path.join( dir_params, par_file_aff_mask )
	
    trans_def_field_tpl = os.path.join( dir_params, 'deformation_transform.tpl.txt' )

    # Final output
    image_mean_space = os.path.join( subject_mean_dir, 'result.nii.gz' ) 
    brain_mean_space = os.path.join( brain_mean_dir, 'result.nii.gz' ) 
    inverse_transform = os.path.join(subject_registration_dir,'invert','InverseTransformParameters.txt')
          
    if os.path.exists(image_mean_space) and os.path.exists(brain_mean_space) and os.path.exists(inverse_transform):
        print( 'Not processed: output images exist already: ' + image_mean_space + ' and ' + brain_mean_space)
        return
    
#    invert_trans_file = os.path.join( template_invert_dir, 'TransformParameters.0.txt' )   
#    if not (subject_name in template_names or os.path.exists(invert_trans_file)):
#        raise IOError(ENOENT, 'Not a file: Run pipeline for template subjects first', invert_trans_file)

    # Step 0: Register brain mask of templates with MNI
    print('# Step 0: Register brain mask of templates with MNI')
    if register_templates_mni:
        for j in range(0, len( template_names )):
            l = template_names[j]
            template_brain_mask = template_brain_masks[j]
            template_subject_output_dir=os.path.join(output_template_dir,l)
            template_mni_dir=os.path.join(template_subject_output_dir,'MNI')
            template_mni_transform = os.path.join(template_mni_dir, 'TransformParameters.1.txt')
        
            if force or (not os.path.exists( template_mni_transform) ) :      
                elx_command = '%s %i -f %s -m %s -p %s -p %s -out %s'% (bin_elastix, threads, template_brain_mask, mni_brain_mask, par_file_sim_mask, par_file_aff_mask, template_mni_dir)
                print elx_command
                os.popen(elx_command)	
            
        if os.path.exists(template_mni_transform):
            filenames=['elastix.log','IterationInfo.0.R0.txt','IterationInfo.0.R1.txt','IterationInfo.0.R2.txt','IterationInfo.1.R0.txt','IterationInfo.1.R1.txt','IterationInfo.1.R2.txt','IterationInfo.2.R0.txt','IterationInfo.2.R1.txt','IterationInfo.2.R2.txt']
            for filename in [os.path.join(template_mni_dir, f) for f in filenames]:
                try:
                    os.remove(filename)
                except OSError:
                    pass				
    
    # Step 1: Register input images with templates
    print('# Step 1: Register input images with templates')
    if force_next:
        force = True
    
    subject_skull_stripped = os.path.join( subject_output_dir , 'brain_image.nii.gz' )
    
    mean_trans_file = os.path.join( subject_registration_dir, 'MeanTransformParameters.txt' )	           
    if force or (not os.path.exists( mean_trans_file) ) :      
        template_names_temp=template_names[:]
        template_brain_masks_temp=template_brain_masks[:]
        list_of_template_images_temp=list_of_template_images[:] 
        
        if not subject_name in template_names:
            template_names.append(subject_name)
            template_brain_masks.append(input_brain_mask)	
            list_of_template_images.append(input_image)
            
        for j in range(0, len( template_names )):
            l = template_names[j]
            template_image = list_of_template_images[j]
            template_brain_mask = template_brain_masks[j]
            template_subject_output_dir=os.path.join(output_template_dir,l)
            subject_template_dir=os.path.join(subject_registration_dir,l)
				        
            template_skull_stripped = os.path.join( template_subject_output_dir, 'brain_image.nii.gz' )
    			
            # Apply brain_masks to T1w
            if not os.path.exists(subject_skull_stripped):
                mask_command = '%s -in %s %s -ops MASK -arg 0 -out %s' % (bin_im, input_image, input_brain_mask, subject_skull_stripped)
                print mask_command
                os.popen(mask_command)
            if not os.path.exists(template_skull_stripped):
                mask_command = '%s -in %s %s -ops MASK -arg 0 -out %s' % (bin_im, template_image, template_brain_mask, template_skull_stripped)
                print mask_command
                os.popen(mask_command)
     
            subject_template_transform=os.path.join( subject_template_dir, 'TransformParameters.2.txt' )
            if subject_name!=l and ( force or (not os.path.exists( subject_template_transform ) ) ):
                force_next = True			
    				
                elx_command = '%s %i -f %s -m %s -p %s -p %s -p %s -out %s'% (bin_elastix, threads, subject_skull_stripped, template_skull_stripped, par_file_sim, par_file_aff, par_file_bsp, subject_template_dir)
                print elx_command
                os.popen(elx_command)				
    				
            elif subject_name==l and ( force or (not os.path.exists( subject_template_transform) ) ):
                elx_command = '%s %i -f %s -m %s -p %s -p %s -p %s -out %s'% (bin_elastix, threads, subject_skull_stripped, subject_skull_stripped, par_file_dum, par_file_dum, par_file_dum, subject_template_dir)
                print elx_command
                os.popen(elx_command)
 
            if os.path.exists( os.path.join( subject_template_dir, 'TransformParameters.2.txt' )):
                filenames=['elastix.log','IterationInfo.0.R0.txt','IterationInfo.0.R1.txt','IterationInfo.0.R2.txt','IterationInfo.1.R0.txt','IterationInfo.1.R1.txt','IterationInfo.1.R2.txt','IterationInfo.2.R0.txt','IterationInfo.2.R1.txt','IterationInfo.2.R2.txt']
                for filename in [os.path.join(subject_template_dir, f) for f in filenames]:
                    try:
                        os.remove(filename)
                    except OSError:
                        pass		
               
            # Step 3: Combine subject-template and template-MNI transformations
            print('# Step 3: Combine subject-template and template-MNI transformations')
            template_mni_dir=os.path.join(template_subject_output_dir,'MNI')   
            template_mni_transform = os.path.join(template_mni_dir, 'TransformParameters.1.txt')
            template_mni_ini_transform = os.path.join(template_mni_dir, 'TransformParameters.0.txt')
            subject_template_mni_transform = os.path.join( subject_template_dir,'MNITemplateTransformParameters.txt')
            subject_template_mni_ini_transform = os.path.join( subject_template_dir,'InitialMNITemplateTransformParameters.txt')

            # Not needed for non-template subjects
            if l == subject_name and not subject_name in template_names_temp:
                continue
            if force or (not os.path.exists( subject_template_mni_transform) ) :
                # Get output spacing etc from MNI images
                info = get_image_info( bin_iinfo, subject_skull_stripped )
                size_voxels = info[0]
                best_spacing = info[1]  
                origin = info[2]
                direction = info[3] 
                
                o = open( subject_template_mni_ini_transform ,"w")
                data = open( template_mni_ini_transform ).read()    
                data = re.sub('InitialTransformParametersFileName "NoInitialTransform"','InitialTransformParametersFileName ' + subject_template_transform , data)   
                o.write( data )
                o.close()                       
 
                o = open( subject_template_mni_transform ,"w")
                data = open( template_mni_transform ).read()    
                data = re.sub('ResampleInterpolator "FinalNearestNeighborInterpolator"','ResampleInterpolator "FinalBSplineInterpolator"', data)
                data = re.sub('InitialTransformParametersFileName "' + template_mni_ini_transform + '"' ,'InitialTransformParametersFileName ' + subject_template_mni_ini_transform, data)   
                data = re.sub('\(Origin -?[\d\.]+ -?[\d\.]+ -?[\d\.]+\)','(Origin %f %f %f)' % ( origin[0], origin[1], origin[2] ), data)
                data = re.sub('\(Index -?[\d\.]+ -?[\d\.]+ -?[\d\.]+\)','(Index 0 0 0)', data)
                data = re.sub('\(Size [\d\.]+ [\d\.]+ [\d\.]+\)','(Size %f %f %f)' % ( size_voxels[0],  size_voxels[1],  size_voxels[2]  ), data)
                data = re.sub('\(Spacing [\d\.]+ [\d\.]+ [\d\.]+\)','(Spacing %f %f %f)' % ( best_spacing[0], best_spacing[1], best_spacing[2]  ), data)
                data = re.sub('\(Direction -?[\d\.]+ -?[\d\.]+ -?[\d\.]+ -?[\d\.]+ -?[\d\.]+ -?[\d\.]+ -?[\d\.]+ -?[\d\.]+ -?[\d\.]+\)','(Direction %f %f %f %f %f %f %f %f %f)' % ( direction[0], direction[1], direction[2], direction[3], direction[4], direction[5], direction[6], direction[7], direction[8]  ), data)
                o.write( data )
                o.close()  
           		
    
        logfile=os.path.join(output_template_dir, 'registration_log.txt')
        if not os.path.exists(logfile):
            with open(logfile, 'w') as f:
                f.write('Running: ' + inspect.getfile(inspect.currentframe()) + '\n')
                f.write('par_file 1  = ' + par_file_sim + '\n')
                f.write('par_file 2  = ' + par_file_aff + '\n')
                f.write('par_file 3  = ' + par_file_bsp + '\n')
                f.write('fixed: ' + subject_skull_stripped + '\n')
                f.write('moving: ' + template_skull_stripped + '\n')
                f.write('template:' + ' '.join(template_names))
                
        template_names=template_names_temp[:]
        template_brain_masks=template_brain_masks_temp[:]
        list_of_template_images=list_of_template_images_temp[:]        
	
    # Step 2: Create mean transformation
    print('# Step 2: Create mean transformation')
    if force_next:
        force = True
	
    trans_param_str = ''
    trans_param_val = 1.0 / len( template_names )
    print len( template_names )
    for j in range(0, len( template_names )):
        l = template_names[j]
        subject_template_dir=os.path.join(subject_registration_dir,l)
        trans_param_str += (' ' + str( trans_param_val ) ) 

    in_file  = os.path.join( subject_subject_dir, 'TransformParameters.0.txt' ) 
    
    mean_trans_file = os.path.join( subject_registration_dir, 'MeanTransformParameters.txt' )	        
			
    if force or (not os.path.exists( mean_trans_file) ) :
        o = open( mean_trans_file,"w")
        data = open( in_file ).read()
        data = re.sub('Transform "BSplineTransform"','Transform "WeightedCombinationTransform"', data)
        data = re.sub('TransformParameters.*','TransformParameters %s)' % trans_param_str, data)
        data = re.sub('NumberOfParameters \d*', 'NumberOfParameters %d' % len( template_names ), data )
        data = re.sub('InitialTransformParametersFileName ".*"','InitialTransformParametersFileName "NoInitialTransform"', data)
        data = re.sub('ResultImagePixelType "short"', 'ResultImagePixelType "float"', data )
        data = re.sub('ResultImageFormat "mhd"','ResultImageFormat "nii.gz"', data )
        data += '\n(SubTransforms'
    	    
        for j in range(0, len( template_names )):
            l = template_names[j]
            subject_template_dir=os.path.join(subject_registration_dir,l)
#            data += (' "' + os.path.join( subject_template_dir, 'TransformParameters.2.txt' ) + '"')
            data += (' "' + os.path.join( subject_template_dir, 'MNITemplateTransformParameters.txt' ) + '"')
   	
        data += ")"
    	        
        o.write( data )
        o.close()

    # Step 3: Change the mean transform to a deformation transform
    print('# Step 3: Change the mean transform to a deformation transform')
    if force_next:
        force = True

    trans_file = os.path.join( subject_registration_dir, 'MeanTransformParameters.txt' )
    def_trans_file   = os.path.join( subject_registration_dir, 'DefFieldTransformParameters.txt' )		
	 
    deformation_field=os.path.join( subject_registration_dir, 'deformationField.nii.gz' )       
    
    if force or (not os.path.exists( deformation_field) ) :
        force_next = True
        trans_command = '%s %i -tp %s -out %s -def all' % (bin_transformix, threads, trans_file, subject_registration_dir)
        os.popen(trans_command)
				
        o = open( def_trans_file, "w")
        data = open( trans_def_field_tpl ).read()
        data = re.sub('__inputfile__', deformation_field, data)
        o.write( data )
        o.close()

    if os.path.exists( deformation_field ):
        filenames=['transformix.log']
        for filename in [os.path.join(subject_registration_dir, fn) for fn in filenames]:
            try:
                os.remove(filename)
            except OSError:
                pass	

    # Step 4: Invert transforms
    print('# Step 4: Invert the transforms')
    if force_next:
        force = True
	
    def_trans_file   = os.path.join( subject_registration_dir, 'DefFieldTransformParameters.txt' )		
    invert_trans_file = os.path.join( subject_invert_dir, 'TransformParameters.0.txt' )
	
    if force or (not os.path.exists( invert_trans_file ) ) :
        force_next = True
        invert_command = '%s %i -m %s -f %s -p %s -t0 %s -out %s' % ( bin_elastix, threads, input_image, input_image, par_file_inv, def_trans_file, subject_invert_dir )
        os.popen(invert_command)

    if os.path.exists( invert_trans_file ):
        filenames=['elastix.log','IterationInfo.0.R0.txt','IterationInfo.0.R1.txt','IterationInfo.0.R2.txt','IterationInfo.1.R0.txt','IterationInfo.1.R1.txt','IterationInfo.1.R2.txt','IterationInfo.2.R0.txt','IterationInfo.2.R1.txt','IterationInfo.2.R2.txt']
        for filename in [os.path.join(subject_invert_dir, fn) for fn in filenames]:
            try:
                os.remove(filename)
            except OSError:
                pass		   
            
    # Get output spacing etc from MNI images
    info = get_image_info( bin_iinfo, mni_brain_mask )
    size_voxels = info[0]
    best_spacing = info[1]  
    origin = info[2]
    direction = info[3]        

    print ("Size (voxel):", size_voxels)  
    print ("Spacing:", best_spacing)
    print ("Origin:", origin)
    print ("Direction:", direction)
  
    # Step 5b: Change output image settings in inverse transform file
    print('# Step 5b: Change output image settings in inverse transform file')
    if force_next:
        force = True
	
    in_file = invert_trans_file
    inverse_transform_file = re.sub('TransformParameters.0.txt', 'InverseTransformParameters.txt', in_file )
	 
    if force or (not os.path.exists( inverse_transform_file ) ):
        o = open( inverse_transform_file,"w")
        data = open( in_file ).read()
        data = re.sub('\(Origin -?[\d\.]+ -?[\d\.]+ -?[\d\.]+\)','(Origin %f %f %f)' % ( origin[0], origin[1], origin[2] ), data)
        data = re.sub('\(Index -?[\d\.]+ -?[\d\.]+ -?[\d\.]+\)','(Index 0 0 0)', data)
        data = re.sub('\(Size [\d\.]+ [\d\.]+ [\d\.]+\)','(Size %f %f %f)' % ( size_voxels[0],  size_voxels[1],  size_voxels[2]  ), data)
        data = re.sub('\(Spacing [\d\.]+ [\d\.]+ [\d\.]+\)','(Spacing %f %f %f)' % ( best_spacing[0], best_spacing[1], best_spacing[2]  ), data)
        data = re.sub('\(Direction -?[\d\.]+ -?[\d\.]+ -?[\d\.]+ -?[\d\.]+ -?[\d\.]+ -?[\d\.]+ -?[\d\.]+ -?[\d\.]+ -?[\d\.]+\)','(Direction %f %f %f %f %f %f %f %f %f)' % ( direction[0], direction[1], direction[2], direction[3], direction[4], direction[5], direction[6], direction[7], direction[8]  ), data)
        data = re.sub('\(InitialTransformParametersFileName ".*"\)', '(InitialTransformParametersFileName "NoInitialTransform")', data)
    	        
        o.write( data )
        o.close()
     	    
    # Step 6a: Transform the images
    print('# Step 6a: Transform the images')
    if force_next:
        force = True
	
    image_mean_space = os.path.join( subject_mean_dir, 'result.nii.gz' ) 
    if force or (not os.path.exists( image_mean_space ) ):
        force_next = True
        trans_command = '%s %i -in %s -tp %s -out %s -jac all' % (bin_transformix, threads, input_image, inverse_transform_file, subject_mean_dir)
        print trans_command
        os.popen(trans_command)
        
    if os.path.exists( image_mean_space ):
        filenames=['transformix.log']
        for filename in [os.path.join(subject_mean_dir, fn) for fn in filenames]:
            try:
                os.remove(filename)
            except OSError:
                pass			
	
    
    # Step 6b: Transform the skull stripped images
    print('# Step 6b: Transform the skull stripped images')
    if force_next:
	    force = True
				
    brain_mean_space=os.path.join( brain_mean_dir, 'result.nii.gz' ) 
    if force or (not os.path.exists( brain_mean_space) ):
        force_next = True
        trans_command = '%s %i -in %s -tp %s -out %s -jac all' % (bin_transformix, threads, subject_skull_stripped, inverse_transform_file, brain_mean_dir)
        print trans_command
        os.popen(trans_command)	
        
    if os.path.exists( brain_mean_space ):
        filenames=['transformix.log']
        for filename in [os.path.join(brain_mean_dir, fn) for fn in filenames]:
            try:
                os.remove(filename)
            except OSError:
                pass	      
    
if __name__ == '__main__':
#    from bigr.environmentmodules import EnvironmentModules
#
#    #Environment modules
#    elastix='elastix/4.8' 
#    itktools='itktools'
#
#    env = EnvironmentModules()
#   
#    env.load(elastix)
#    env.load(itktools)	
#    
    #Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=unicode, help='input image file')
    parser.add_argument('--brain', type=unicode, help='brain mask file for input image')
    parser.add_argument('--templates', type=unicode, nargs='+', help='template files')
    parser.add_argument('--templatebrains', type=unicode, nargs='+', help='brain masks for template files')
    parser.add_argument('--mni', type=unicode, help='mni brain mask, e.g. /cm/shared/apps/fsl/5.0.2.2/data/standard/MNI152_T1_1mm_brain_mask.nii.gz')
    parser.add_argument('--registertemplatesmni', type=bool, help='do you want to register the templates with MNI space in this run?')
    parser.add_argument('--threads', type=unicode, help='number of threads for elastix')
    parser.add_argument('--out', type=unicode, help='output directory')
    args = parser.parse_args() 

    input_image=args.input
    list_of_template_images=args.templates
    output_dir=args.out
    input_brain_mask=args.brain
    template_brain_masks = args.templatebrains
    threads=int(args.threads)
    mni_brain_mask = args.mni
    register_templates_mni = args.registertemplatesmni

    process(input_image, input_brain_mask, list_of_template_images, template_brain_masks, mni_brain_mask, output_dir, threads, register_templates_mni)