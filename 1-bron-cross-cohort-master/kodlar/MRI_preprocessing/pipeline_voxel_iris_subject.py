# Revised version of the Voxel-wise iris pipeline
# Pipeline has subject/tool structure like Fastr
# Runs both on BIGR and Cartesius cluster
# Esther Bron - e.bron@erasmusmc.nl 

import os
import re
from errno import ENOENT

# Define cluster, modules and job submission functions
try:
    ## BIGR Cluster ##
    #Python module for job submission
    from bigr.clustercontrol import ClusterControl
    cc = ClusterControl()
    
    #Environment modules
    from bigr.environmentmodules import EnvironmentModules
    
    elastix='elastix/4.7'
    matlab='matlab/R2013b'
    mcr='mcr/R2013b'
    itktools='itktools'

    env = EnvironmentModules()
    env.unload(matlab)
    env.unload(mcr)
    env.load(elastix)
    env.load(itktools)	    
    
    #Set variables
    bigr=1
    
    threads=1  
    number_of_jobs_per_node=200    
    
    print('Submitting jobs with ClusterControl on bigr-cluster')
except ImportError:
    ## Cartesius Cluster ##
    #Function for job submission
    from slurm import sbatch
    
    #Set variables
    bigr=0
    
    threads=4  
    number_of_jobs_per_node=6 
    
    print('Submitting jobs with Slurm on Cartesius cluster')

# Set parameters

# Which ROI labelings should be transformed to template space: roi_list=['seg','lobe','hemisphere','brain', 'brain_mask']
# Seg is required to segment cerebellum
roi_list=['seg', 'brain_mask']

input_dir = os.getcwd()

# Set template directory
#template_dir = input_dir
template_dir = os.path.join(input_dir, os.pardir, 'ADNI')

## For testing
#templates=['007_S_0101_bl',
#           '123_S_0108_bl',
#           '018_S_0103_bl',
#           '130_S_0102_bl',
#           '068_S_0109_bl',
#           '136_S_0107_bl',
#           '123_S_0106_bl']

# For ADNI 50 random subjects
templates=['014_S_0169_bl',
           '006_S_5153_bl',
           '024_S_4169_bl',
           '073_S_2264_bl',
           '003_S_1122_bl',
           '005_S_0610_bl',
           '041_S_4200_bl',
           '027_S_2219_bl',
           '032_S_0400_bl',
           '067_S_1253_bl',
           '021_S_0337_bl',
           '016_S_0702_bl',
           '021_S_2125_bl',
           '013_S_4395_bl',
           '041_S_4629_bl',
           '099_S_0470_bl',
           '023_S_4502_bl',
           '037_S_4015_bl',
           '129_S_0778_bl',
           '005_S_4168_bl',
           '072_S_4539_bl',
           '005_S_0602_bl',
           '116_S_4199_bl',
           '027_S_0307_bl',
           '016_S_4951_bl',
           '109_S_2200_bl',
           '006_S_4515_bl',
           '021_S_0343_bl',
           '032_S_0677_bl',
           '041_S_1260_bl',
           '128_S_4607_bl',
           '136_S_0429_bl',
           '100_S_0296_bl',
           '035_S_0048_bl',
           '126_S_0605_bl',
           '130_S_4730_bl',
           '033_S_0723_bl',
           '127_S_4301_bl',
           '024_S_4905_bl',
           '100_S_1286_bl',
           '126_S_1077_bl',
           '041_S_4143_bl',
           '135_S_5113_bl',
           '137_S_4816_bl',
           '941_S_4365_bl',
           '014_S_4328_bl',
           '035_S_0997_bl',
           '023_S_0126_bl',
           '073_S_4311_bl',
           '022_S_5004_bl']


mni_brain_mask = "/cm/shared/apps/fsl/5.0.2.2/data/standard/MNI152_T1_1mm_brain_mask.nii.gz"
if not os.path.exists(mni_brain_mask):
    mni_brain_mask = "~/fsl/5.0.2.2/data/standard/MNI152_T1_1mm_brain_mask.nii.gz"

#binaries
bin_average = 'pxmeanstdimage'
bin_combine = 'pxcombinesegmentations'

image_dir=os.path.join( input_dir, 'T1w','nuc')
segmentation_dir=os.path.join(input_dir, 'Hammers', 'atlas_work_temp')
tissue_dir=os.path.join(input_dir, 'Tissue')
spm_dir=os.path.join(input_dir, 'T1w', 'spm' )

template_image_dir = os.path.join( template_dir, 'T1w','nuc')
template_segmentation_dir = os.path.join( template_dir, 'Hammers', 'atlas_work_temp')

template_images = [os.path.join( template_image_dir, t + '.nii.gz') for t in templates]
template_brains = [os.path.join( template_segmentation_dir, t, 'brain_mask.nii.gz') for t in templates]

job_file = 'vox_adni'

# Step 1: Subjects to atlas 
print '# Step 1: Subjects to atlas '
command_list=[]
job_id_list=[]
register_templates_mni=True
for input_image in os.listdir(image_dir):
    if re.match('.*.nii.gz', input_image ): # or re.match('.*m[01][0-9].nii.gz', input_image ):
        subject_name=os.path.splitext(os.path.splitext(input_image)[0])[0]        
        subject_image=os.path.join(image_dir, input_image)
       
#        if not subject_name in templates:
#            print ('Continue, templates first')
#            continue
        
        brain_mask=os.path.join(segmentation_dir, subject_name, 'brain_mask.nii.gz')
        if not os.path.exists(brain_mask):
            raise IOError(ENOENT, 'Not a file: Run pipeline_roi_iris.py to generate this', brain_mask)

        result_image=os.path.join(input_dir, 'Template_space', subject_name,'Brain_image_in_template_space','result.nii.gz')
        transform=os.path.join(input_dir, 'Template_space', subject_name,'Template_registration','invert','InverseTransformParameters.txt')
        if os.path.exists(result_image) and os.path.exists(transform):
            continue

        subject_to_template_function='~/scripts/iris_pipeline/subject_to_template.py'
        process_command = 'python %s %s --brain %s --templates %s --templatebrains %s --mni %s --registertemplatesmni %s --threads %i --out %s'% (subject_to_template_function, subject_image, brain_mask, ' '.join(template_images), ' '.join(template_brains), mni_brain_mask, register_templates_mni, threads, input_dir)
        
        command_list.append(process_command)
        
        register_templates_mni=False
    					
        job_file_def = job_file + '_subject2template_' + subject_name
        if len(command_list) == number_of_jobs_per_node:
            if bigr:
                job_id = cc.send_job_array(command_list, 'day', '10:00:00', job_file_def ,'10G')      		
            else:                  
                job_id = sbatch(command_list, job_file_def, threads, time='0-10:00:00', mem='2G')
            
            job_id_list = job_id_list + job_id
            command_list = []
            
#        if len(job_id_list)==200:
#            exit()
 
if command_list:
    if bigr:
        job_id = cc.send_job_array(command_list, 'day', '10:00:00', job_file_def ,'10G')      		
    else:                  
        job_id = sbatch(command_list, job_file_def, threads, time='0-05:00:00', mem='2G')
 
    job_id_list = job_id_list + job_id
    command_list=[]
    
    if bigr:
        # Wait until registrations are all done
        cc.wait(job_id_list)

# Step 1b: Subjects similarity to MNI (independent from other steps)
print '# Step 1b: Subjects similarity to MNI '
command_list=[]
job_id_list=[]
for input_image in os.listdir(image_dir):
    if re.match('.*.nii.gz', input_image ): # or re.match('.*m[01][0-9].nii.gz', input_image ):
        subject_name=os.path.splitext(os.path.splitext(input_image)[0])[0]        
        subject_skull_stripped = os.path.join( input_dir, 'Template_space', subject_name, 'brain_image.nii.gz' )
           
#        if not subject_name in templates:
#            print ('Continue, templates first')
#            continue
        
        brain_mask=os.path.join(segmentation_dir, subject_name, 'brain_mask.nii.gz')
        if not os.path.exists(brain_mask):
            raise IOError(ENOENT, 'Not a file: Run pipeline_roi_iris.py to generate this', brain_mask)

        result_dir=os.path.join(input_dir, 'Template_space', subject_name,'Brain_image_in_MNI_space')
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        
        if os.path.exists(os.path.join(result_dir,'result.nii.gz')):
            continue

        subject_to_mni_function='~/scripts/iris_pipeline/subject_to_mni.py'
        process_command = 'python %s %s --brain %s --mni %s --threads %i --out %s --sim'% (subject_to_mni_function, subject_skull_stripped, brain_mask, mni_brain_mask, threads, result_dir)
  
        command_list.append(process_command)
            					
        job_file_def = job_file + '_subject2mni_' + subject_name
        if len(command_list) == number_of_jobs_per_node:
            if bigr:
                job_id = cc.send_job_array(command_list, 'day', '00:10:00', job_file_def ,'2G')      		
            else:                  
                job_id = sbatch(command_list, job_file_def, threads, time='0-00:10:00', mem='2G')
            
            job_id_list = job_id_list + job_id
            command_list = []
            
#        if len(job_id_list)==200:
#            exit()
 
if command_list:
    if bigr:
        job_id = cc.send_job_array(command_list, 'day', '00:10:00', job_file_def ,'2G')      		
    else:                  
        job_id = sbatch(command_list, job_file_def, threads, time='0-00:10:00', mem='2G')
 
    job_id_list = job_id_list + job_id
    command_list=[]
    
    if bigr:
        # Wait until registrations are all done
        cc.wait(job_id_list)

# Step 2: Transform tissue segmentation to template space
print '# Step 2: Transform tissue segmentation to template space'
job_id_list=[]

job_id_list_previous=job_id_list
job_id_list=[]
command_list=[]
for input_image in os.listdir(image_dir):
    if re.match('.*.nii.gz', input_image ): # or re.match('.*m[01][0-9].nii.gz', input_image ):
        subject_name=os.path.splitext(os.path.splitext(input_image)[0])[0]        
        subject_image=os.path.join(image_dir, input_image)
        brain_mask=os.path.join(segmentation_dir, subject_name, 'brain_mask.nii.gz')

#        if not subject_name in templates:
#            print ('Continue, templates first')
#            continue
                
        brain_mask=os.path.join(segmentation_dir, subject_name, 'brain_mask.nii.gz')
        seg=os.path.join(segmentation_dir, subject_name, 'seg.nii.gz')
        tissue=os.path.join(tissue_dir, subject_name, 'tissue.nii.gz')
        gm=os.path.join(spm_dir, 'wc1w' + subject_name + '.nii.gz')
        segmentations=[brain_mask, seg, tissue, gm]
        segmentation_names=['brain_mask', 'seg','tissue', 'wc1w' + subject_name ]
        nns=['--nn', '--nn', '--nn', '--no-nn'] #nn if nearest neighbor transform needed
        
        transform=os.path.join(input_dir, 'Template_space', subject_name,'Template_registration','invert','InverseTransformParameters.txt')
        
        for segmentation,nn,segmentation_name in zip(segmentations,nns,segmentation_names):        
            
            segmentation_output=os.path.join(input_dir, 'Template_space', subject_name, segmentation_name + '_in_template_space', 'result.nii.gz') 
            segmentation_output_positive=os.path.join(input_dir, 'Template_space', subject_name, segmentation_name + '_in_template_space', 'result_positive.nii.gz') 

            if os.path.exists(segmentation_output) or os.path.exists(segmentation_output_positive):
                continue

            segmentation_to_template_function='~/scripts/iris_pipeline/segmentations_to_template.py'
            process_command = 'python %s --input %s --subject %s --transform %s %s --threads %i --out %s'% (segmentation_to_template_function, segmentation, subject_name, transform, nn, threads, input_dir)

            command_list.append(process_command)
            
					
        job_file_def = job_file + '_seg2template_' + subject_name
        
        if len(command_list) >= number_of_jobs_per_node:  
            if bigr:
                job_id = cc.send_job_array(command_list[:number_of_jobs_per_node], 'day', '10:00:00', job_file_def ,'10G')      		
            else:             
                job_id = sbatch(command_list[:number_of_jobs_per_node], job_file_def, threads, time='0-05:00:00', mem='2G',dependency=job_id_list_previous)
            
            command_list = command_list[number_of_jobs_per_node:]
            job_id_list = job_id_list + job_id
     
if command_list:
    if bigr:
        job_id = cc.send_job_array(command_list, 'day', '10:00:00', job_file_def ,'10G')      		
    else:                  
        job_id = sbatch(command_list, job_file_def, threads, time='0-02:00:00', mem='2G', dependency=job_id_list_previous)
 
    job_id_list = job_id_list + job_id
    command_list=[]
    
    if bigr:
        # Wait until registrations are all done
        cc.wait(job_id_list)

# Step 3: Mask template space images by positive Jacobian values
print '# Step 3: Mask template space images by positive Jacobian values'
job_id_list=[]

job_id_list_previous=job_id_list
job_id_list=[]
command_list=[]

threads=1
number_of_jobs_per_node=24
for input_image in os.listdir(image_dir):
    if re.match('.*.nii.gz', input_image ): # or re.match('.*m[01][0-9].nii.gz', input_image ):
        subject_name=os.path.splitext(os.path.splitext(input_image)[0])[0]        

#        if not subject_name in templates:
#            print ('Continue, templates first')
#            continue
 
        segmentation_names = ['Brain_image', 'Image', 'brain_mask', 'seg','tissue', 'wc1w' + subject_name ]
        input_images = [os.path.join(input_dir, 'Template_space', subject_name, s + '_in_template_space', 'result.nii.gz') for s in segmentation_names]
        output_images = [os.path.join(input_dir, 'Template_space', subject_name, s + '_in_template_space', 'result_positive.nii.gz') for s in segmentation_names]
    
        spatial_jacobian = os.path.join(input_dir, 'Template_space', subject_name, 'Brain_image_in_template_space', 'spatialJacobian.nii.gz')
        for input_image,output_image in zip(input_images,output_images):        
            
            if os.path.exists(output_image):
                continue

            mask_jacobian_function='~/scripts/iris_pipeline/mask_negative_jacobian.py'
            process_command = 'python %s --mask %s --jac %s --out %s'% (mask_jacobian_function, input_image, spatial_jacobian, output_image)

            command_list.append(process_command)
            
					
        job_file_def = job_file + '_jacobian_mask_' + subject_name
        
        if len(command_list) >= number_of_jobs_per_node:  
            if bigr:
                job_id = cc.send_job_array(command_list[:number_of_jobs_per_node], 'day', '01:00:00', job_file_def ,'2G')      		
            else:             
                job_id = sbatch(command_list[:number_of_jobs_per_node], job_file_def, threads, time='0-00:01:00', mem='1G',dependency=job_id_list_previous)
            
            command_list = command_list[number_of_jobs_per_node:]
            job_id_list = job_id_list + job_id
     
if command_list:
    if bigr:
        job_id = cc.send_job_array(command_list, 'day', '01:00:00', job_file_def ,'2G')      		
    else:                  
        job_id = sbatch(command_list, job_file_def, threads, time='0-00:01:00', mem='1G', dependency=job_id_list_previous)
 
    job_id_list = job_id_list + job_id
    command_list=[]
    
    if bigr:
        # Wait until registrations are all done
        cc.wait(job_id_list)

    
# Step 4: Create mean template space images
print '# Step 4: Create mean template space images'

#job_id_list_previous=[] 
job_id_list_previous=job_id_list
job_id_list=[]
command_list=[]

threads=1
input_images=[os.path.join(input_dir, 'Template_space', t, 'Brain_image_in_template_space', 'result_positive.nii.gz') for t in templates]
output_file = os.path.join(input_dir, 'Template_space', input_images[0].split(os.sep)[-2] + '.nii.gz' )
if not os.path.exists( output_file ):       
    process_command = ' '.join([bin_average, '-in'] + input_images + ['-outmean', output_file ])
    print process_command
    command_list.append(process_command)

input_images=[os.path.join(input_dir, 'Template_space', t, 'tissue_in_template_space', 'result_positive.nii.gz') for t in templates]
number_of_labels='4'
output_file = os.path.join(input_dir, 'Template_space', input_images[0].split(os.sep)[-2] + '.nii.gz' )
if not os.path.exists( output_file ):       
    process_command = ' '.join([bin_combine, '-n', number_of_labels, '-m', 'VOTE','-in'] + input_images + ['-outh', output_file ])
    print process_command
    command_list.append(process_command)

input_images=[os.path.join(input_dir, 'Template_space', t, 'brain_mask_in_template_space', 'result_positive.nii.gz') for t in templates]
number_of_labels='2'
output_file = os.path.join(input_dir, 'Template_space', input_images[0].split(os.sep)[-2] + '.nii.gz' )
if not os.path.exists( output_file ):       
    process_command = ' '.join([bin_combine, '-n', number_of_labels, '-m', 'VOTE','-in'] + input_images + ['-outh', output_file ])
    print process_command
    command_list.append(process_command)

input_images=[os.path.join(input_dir, 'Template_space', t, 'seg_in_template_space', 'result_positive.nii.gz') for t in templates]
number_of_labels='84'
output_file = os.path.join(input_dir, 'Template_space', input_images[0].split(os.sep)[-2] + '.nii.gz' )
if not os.path.exists( output_file ):       
    process_command = ' '.join([bin_combine, '-n', number_of_labels, '-m', 'VOTE','-in'] + input_images + ['-outh', output_file ])
    print process_command
    command_list.append(process_command)

#Or mask
mask_function='~/scripts/iris_pipeline/create_any_mask.py'
input_images=[os.path.join(input_dir, 'Template_space', t, 'tissue_in_template_space', 'result_positive.nii.gz') for t in templates]
output_file = os.path.join(input_dir, 'Template_space', 'gm_any_in_template_space.nii.gz' )
if not os.path.exists( output_file ):       
    process_command = ' '.join(['python ', mask_function, '--input'] + input_images + ['--out', output_file, '--value', '2'])
    print process_command
    command_list.append(process_command)


job_file_def = job_file + '_createtemplate'

if command_list:
    if bigr:
        job_id = cc.send_job_array(command_list, 'day', '10:00:00', job_file_def ,'10G')      		
    else:
        job_id = sbatch(command_list, job_file_def, threads, time='0-01:00:00', mem='2G', dependency=job_id_list_previous)
    
    job_id_list = job_id_list + job_id
    command_list=[]

    if bigr:
        # Wait until registrations are all done
        cc.wait(job_id_list)
        

# Step 4: Calculate features
print '# Step : Calculate features'	
job_id_list_previous=job_id_list
command_list=[]
job_id_list=[]
register_templates_mni=True
threads=1
for input_image in os.listdir(image_dir):
    if re.match('.*.nii.gz', input_image ): # or re.match('.*m[01][0-9].nii.gz', input_image ):
        subject_name=os.path.splitext(os.path.splitext(input_image)[0])[0]        
        subject_image=os.path.join(image_dir, input_image)
       
#        if not subject_name in templates:
#            print ('Continue, templates first')
#            continue
        brain_mask=os.path.join(segmentation_dir, subject_name, 'brain_mask.nii.gz')
        spatial_jacobian = os.path.join(input_dir, 'Template_space', subject_name, 'Brain_image_in_template_space','spatialJacobian.nii.gz' )
        t1w_template_image = os.path.join(input_dir, 'Template_space', subject_name, 'Brain_image_in_template_space','result.nii.gz' )
        gm_density_map = os.path.join(input_dir, 'Template_space', subject_name, 'wc1w' + subject_name + '_in_template_space','result_positive.nii.gz')
       
        brain_mask_template = os.path.join(input_dir, 'Template_space', subject_name, 'brain_mask_in_template_space','result_positive.nii.gz')
#        gm_or_mask =  os.path.join(input_dir, 'Template_space', 'gm_in_template_space.nii.gz')
    
        result_dir=os.path.join(input_dir, 'Template_space', subject_name,'Brain_image_in_template_space')
        result_image=os.path.join(result_dir,'gmModulatedJacobian.nii.gz')
        result_image_t1w=os.path.join(result_dir,'T1wModulatedJacobian.nii.gz')
        if os.path.exists(result_image) and os.path.exists(result_image_t1w):
            continue

        voxelwise_features_function='~/scripts/iris_pipeline/voxelwise_features.py'
        process_command = 'python %s --brain %s --jac %s --t1 %s --gm %s --mask %s --out %s'% (voxelwise_features_function, brain_mask, spatial_jacobian, t1w_template_image, gm_density_map, brain_mask_template, result_dir )
       
        command_list.append(process_command)
        
        register_templates_mni=False
    					
        job_file_def = job_file + '_features_' + subject_name
        if len(command_list) == number_of_jobs_per_node:
            if bigr:
                job_id = cc.send_job_array(command_list, 'day', '01:00:00', job_file_def ,'2G')      		
            else:                  
                job_id = sbatch(command_list, job_file_def, threads, time='0-01:00:00', mem='2G', dependency=job_id_list_previous)
                
            job_id_list = job_id_list + job_id
            command_list = []
            
#        if len(job_id_list)==200:
#            exit()
 
if command_list:
    if bigr:
        job_id = cc.send_job_array(command_list, 'day', '01:00:00', job_file_def ,'2G')      		
    else:                  
        job_id = sbatch(command_list, job_file_def, threads, time='0-01:00:00', mem='2G', dependency=job_id_list_previous)
 
    job_id_list = job_id_list + job_id
    command_list=[]
    
    if bigr:
        # Wait until registrations are all done
        cc.wait(job_id_list)
        
exit()       
# Step 5: Quality control image processing (create png files for inspection)
print '### Step 5: Quality control image processing (create png files for inspection)'	
import quality_check
every_roi_separate = False #Create inspect file for every individual roi of the Hammers atlas.
quality_check.main(input_dir)
