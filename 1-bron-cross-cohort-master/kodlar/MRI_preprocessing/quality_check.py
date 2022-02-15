#!/bin/env python


import os
import re
#from pylab import *
from optparse import OptionParser
import overlay_png
import image_png
import overlay_atlas_png


try:
    from bigr.clustercontrol import ClusterControl
    cc = ClusterControl()
    bigr=1
    from register_asl_newregistration import get_asl_images
except ImportError:
    from slurm import sbatch
    bigr=0
    
print bigr

def main(data_dir, work_subdir='atlas_work_temp', seg_list=['seg'], every_roi_separate=False):
	"""Quality check: create png files for inspection"""

	# Store data dir
#	data_dir = os.path.abspath(os.path.expanduser(sys.argv[1]))
	print "Data dir: " + data_dir
	t1w_dir = os.path.join( data_dir, 'T1w' )
	nuc_dir = os.path.join( t1w_dir, 'nuc' )
	spm_dir =  os.path.join( t1w_dir, 'spm' )
	flair_dir = os.path.join( data_dir, 'FLAIR' )
	asl_dir = os.path.join( data_dir, 'ASL' )
	dti_dir = os.path.join( data_dir, 'DTI' )
	cbf_dir = os.path.join( data_dir, 'CBF_pvc' )
	if not os.path.exists(cbf_dir):
		cbf_dir = os.path.join( data_dir, 'CBF' )
	tissue_dir = os.path.join( data_dir, 'Tissue' )
	feature_dir = os.path.join( data_dir, 'Features' )
	hammers_dir = os.path.join( data_dir, 'Hammers' )
	inspect_dir = os.path.join( data_dir, 'Inspect' )
	atlas_dir = os.path.join( '/scratch','ebron','IRIS', 'Hammers_n30r83' )
	
	if not os.path.exists( inspect_dir ):
		os.mkdir( inspect_dir )	
	
	# Read and filter input files
	input_files = [];
	for filename in os.listdir( t1w_dir ):
		if re.match('.*.nii.gz', filename ):
			input_files.append( filename )
	
	if not input_files:
   		t1w_dir=nuc_dir
   		for filename in os.listdir( nuc_dir ):
			if re.match('.*.nii.gz', filename ):
				input_files.append( filename )  
		
	input_files.sort()
	
	#for k, filename in enumerate( input_files ):
	#    print( '%03d: %s' % (k, filename ) )
		
	lists = [];		
	for i, filename in enumerate( input_files ):
			(filebase,ext)=os.path.splitext(filename) 
			(filebase,ext)=os.path.splitext(filebase) 
			#(iris, filebase)=filebase.split('_')
			print( '%03d: %s' % (i, filebase ) )
			lists.append( filebase )

	command_list=[]
	for j in range(0, len( input_files )): #Loop over all subjects
			l =  lists[j]
#		if not (l == "PSI_53062" or l == "PSI57396"):
#			continue
			seg_list_string='[\"' + '\",\"' .join(str(seg) for seg in seg_list) + '\"]'
			python_command="import sys; sys.path.append(\"%s\"); import overlay_png; import image_png; import overlay_atlas_png; from quality_check import do_check; do_check(\"%s\", \"%s\", \"%s\", \"%s\", \"%s\", \"%s\", \"%s\", \"%s\", \"%s\", \"%s\", \"%s\",%s,%s)" % (os.path.dirname(os.path.abspath(__file__)), l, t1w_dir, spm_dir, flair_dir, inspect_dir, hammers_dir, feature_dir, asl_dir, cbf_dir, dti_dir, work_subdir, seg_list_string, every_roi_separate)
			s_command = "python -c '%s'" % python_command
		
			command_list.append(s_command)	
			print command_list  
			run_id = 'quality_check' 
		
			if len(command_list)>=24:	
				if bigr:
					job_id=cc.send_job_array( command_list, 'day', '00:30:00', run_id, '4G')		
					print job_id
				else:
					sbatch(command_list, run_id + l, 1)
				command_list = [];
				exit()
					
			# Wait until jobs are all done
	# 		cc.wait(job_id)


	if command_list:	
		if bigr:        
			job_id = cc.send_job_array( command_list, 'day', '00:30:00', run_id, '4G')		
			print job_id
		else:
			sbatch(command_list, run_id + l, 1)            
		# Wait until jobs are all done
# 		cc.wait(job_id)
		command_list = [];
		
	
def do_check(l, t1w_dir, spm_dir, flair_dir, inspect_dir, hammers_dir, feature_dir, asl_dir, cbf_dir, dti_dir, work_subdir, seg_list, every_roi_separate):	
# 	# Step 0: Check T1w images
# 	print '# Step 0: Check T1w images' 
# 	if os.path.exists(t1w_dir):			
# 		t1w = os.path.join( t1w_dir , l + '.nii.gz')		
# 		if os.path.exists(t1w):
# 			image_png.main(t1w,inspect_dir)
# 	else:
# 		print 'No T1 image available'
#
#	# Step 0: Check FLAIR images
#	print '# Step 0: Check FLAIR images' 
#	if os.path.exists(flair_dir):			
#		flair = os.path.join( flair_dir , l + '.nii.gz')		
#		if os.path.exists(flair):
#			image_png.main(flair,inspect_dir)
#	else:
#		print 'No FLAIR image available'

	# Step 1: Check SPM tissue segmentation
	print '# Step 1: Check SPM tissue segmentation' 
	if os.path.exists(spm_dir):			
		for i,t in enumerate(['gm']): #,'wm'
			t1w = os.path.join( t1w_dir , l + '.nii.gz')
			tissue = os.path.join( spm_dir, l + '_' + t + '.nii.gz')
			if not os.path.exists(tissue):
				tissue = os.path.join( spm_dir, 'wc' + str(i) + 'w' + l + '.nii.gz')				
			if os.path.exists(t1w) and os.path.exists(tissue):
				overlay_png.main(t1w,tissue,inspect_dir)
	else:
		print 'No tissue segmentation available'

	# Step 2a: Check multi-atlas segmentations: brain_mask
	print '# Step 2a: Check multi-atlas segmentations: brain_mask' 
	if os.path.exists(hammers_dir):
		t = 'brain_mask'
		t1w = os.path.join( t1w_dir , l + '.nii.gz')
		tissue = os.path.join( hammers_dir, work_subdir, l, t + '.nii.gz')			
		if os.path.exists(t1w) and os.path.exists(tissue):
			overlay_png.main(t1w,tissue,inspect_dir)
	else:
		print 'No brain_mask available'
		
	# Step 2b: Check multi-atlas segmentations: regions
	print '# Step 2b: Check multi-atlas segmentations: regions' 
	if os.path.exists(hammers_dir):
		rois_separate=[]
		n0=0
		for t in seg_list:
			seg=t
			print t
			if t in ['brain_mask']:
				continue
			elif t in ['brain']:
				#Brain 
				n=3
			elif t in ['hemisphere']:
				#Hemisphere
				n=4            
			elif t in ['lobe']:
				#Lobe
				n=12
			elif t in ['seg']:
				#seg
				n=84 
				seg='region'
				
			elif t in ['precentralgyrussuperior']:	
				n0=50
				n=52
			else:
				seg='region'					
				hammers_txt = os.path.join( '/scratch', 'ebron', 'IRIS', 'Hammers_n30r83', 'Hammers_mith_atlases_structure_names_r83.txt')
				f = open(hammers_txt, 'r')	
				for line in f:
					[roinr, sep, region_name]  = line.partition('	')
					[region_name, rest,rest] = region_name.partition('\n')
					specific_regions=[t + ' left', t + ' right']						
					if region_name in specific_regions:
						rois_separate.append(roinr)
			
			t1w = os.path.join( t1w_dir , l + '.nii.gz')
			if rois_separate:					
				for regionnr in rois_separate:					
					tissue = os.path.join( feature_dir, work_subdir, seg + '_mask', l , str(regionnr) + '.nii.gz')
				
					if os.path.exists(t1w) and os.path.exists(tissue):
						overlay_png.main(t1w,tissue,inspect_dir)
			else:					
				tissue = os.path.join( hammers_dir, work_subdir, l , t + '.nii.gz')
				if os.path.exists(t1w) and os.path.exists(tissue):
					overlay_atlas_png.main(t1w,tissue,inspect_dir)
					
				if every_roi_separate:	
					out_dir=os.path.join(inspect_dir, seg)
		
					if not os.path.exists(out_dir ):
						os.mkdir( out_dir )	
						
					for regionnr in range(n0,n):					
						tissue = os.path.join( feature_dir, work_subdir, seg + '_mask', l , str(regionnr) + '.nii.gz')
					
						print tissue
						if os.path.exists(t1w) and os.path.exists(tissue):
							overlay_png.main(t1w,tissue,out_dir)
	else:
		print 'No multi-atlas segmentation available'

	# Step 3: Check ASL-T1w registration
	print '# Step 3: Check ASL-T1w registration'
	if os.path.exists(asl_dir):
		k_t1w = l
		ks=get_asl_images(asl_dir,k_t1w)
		for k in ks: 
			asl = os.path.join( asl_dir , k + '_difference.nii.gz')
			tissue = os.path.join( t1w_dir, 'asl_space', 'gm', k , 'result.nii.gz')
			
			if os.path.exists(asl) and os.path.exists(tissue):
				overlay_png.main(asl,tissue,inspect_dir)
	else:
		print 'No ASL imaging data available'
	
	# Step 4: Check CBF maps
	print '# Step 4: Check CBF maps'
	if os.path.exists(cbf_dir):
		k_t1w = l
	
		ks=get_asl_images(asl_dir,k_t1w)
		for k in ks: 
			asl = os.path.join( cbf_dir , k, 'cbf_gm_pure_corrected.nii.gz')
			tissue = os.path.join( t1w_dir, 'asl_space', 'brain_mask', k , 'result.nii.gz')

			if os.path.exists(asl) and os.path.exists(tissue):
				overlay_png.main(asl,tissue,inspect_dir)
	else:
		print 'No ASL imaging data available'	
		
	# Step 5: Check DTI-T1w registration
	print '# Step 5: Check DTI-T1w registration'
	if os.path.exists(dti_dir):
		registration_folder='registrations'
		fa = os.path.join( dti_dir , work_subdir, l, registration_folder, 'dti_FA', 'result.nii.gz')
		tissue = os.path.join( t1w_dir, 'spm',  'wc2w' + l + '.nii.gz')

		if os.path.exists(fa) and os.path.exists(tissue):
			overlay_png.main(fa,tissue,inspect_dir)
	else:
		print 'No DTI imaging data available'

	# Step 6: Check template space 
	print '# Step 6: Check template space'
	if os.path.exists(os.path.join(t1w_dir, work_subdir)):			
		t1w= os.path.join(t1w_dir, work_subdir, 'mean_space_' + l, 'result.nii.gz')
		tissue = os.path.join( feature_dir, work_subdir + '_nomask', 'gm_vote_mask.nii.gz')
		
		if os.path.exists(t1w) and os.path.exists(tissue):
			overlay_png.main(t1w,tissue,inspect_dir)
	else:
		print 'No template space registrations available'
	
	# Step 7: CBF - template space 
	print '# Step 7: CBF - template space'
	if os.path.exists(os.path.join(cbf_dir, work_subdir,'mean_space_cbf_cbfmap')):	
		k_t1w = l		
		ks=get_asl_images(asl_dir,k_t1w)
		for k in ks: 					
			cbf= os.path.join(cbf_dir, work_subdir, 'mean_space_cbf_cbfmap', k, 'result.nii.gz')
			tissue = os.path.join( t1w_dir, work_subdir, 'spm_template_space_nomask', k_t1w, 'result.nii.gz')
			
			if os.path.exists(cbf) and os.path.exists(tissue):
				overlay_png.main(cbf,tissue,inspect_dir)
	else:
		print 'No template space registrations for CBF available'		

	# Step 8: DTI - template space 
	print '# Step 8: DTI - template space'
	if os.path.exists(os.path.join(dti_dir, work_subdir, 'IRIS_050', 'registrations_mean_space')):	
		dti= os.path.join(dti_dir, work_subdir, l, 'registrations_mean_space', 'dti_FA', 'result.nii.gz')
		tissue = os.path.join( t1w_dir, work_subdir, 'wm_prob_template_space_nomask', l, 'result.nii.gz')
		
		if os.path.exists(dti) and os.path.exists(tissue):
			overlay_png.main(dti,tissue,inspect_dir)
	else:
		print 'No template space registrations for DTI available'	
		

if __name__ == '__main__':
	# Parse input arguments
	parser = OptionParser(description="Quality check: create png files for inspection", usage="Usage: python %prog input_dir. Use option -h for help information.")
	(options, args) = parser.parse_args()
	work_subdir ='atlas_work_temp';
	print 'Starting...'

	if len(args) != 1:
		parser.error("wrong number of arguments")
		
	data_dir=args[0]	
	lobe_list=['seg']
	every_roi_separate = False
# 	lobe_list=['Insula', 'Hippocampus']
# 	every_roi_separate = True
	
	
	main(data_dir, work_subdir, lobe_list, every_roi_separate)		
