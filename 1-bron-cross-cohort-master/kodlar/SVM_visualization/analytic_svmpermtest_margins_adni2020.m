%Trains SVM using the specified list file and
%saves P-map as well as model
%needs libsvm and NIFTI loaders pre installed (if you dont have these see
%the INSTALL file that came with the package)
%To create your own example 'trainlist' see the train list that came with
%package and follow the instructions in the README
%Reading from a list file

%Code by Bilwaj Gaonkar
%Code obtained from: http://www.rad.upenn.edu/sbia/Bilwaj.Gaonkar/supl_matl_nimg.zip
%Citation: Gaonkar et al., Medical Image Analysis 2013, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3767485/

%Modified by Esther Bron

% Settings
sign_level=0.05;
pmap_offset=0;
c_par = 0.0313;

output_directory= fullfile('/scratch', 'ebron', 'ADNI', 'Pmaps');

% Load data
data_file = fullfile('/scratch', 'ebron', 'ADNI', 'adni_ad_t1_baseline_masked.hdf5');
CF_MATSC = h5read(data_file, '/data');
CF_MATSC = CF_MATSC';
size(CF_MATSC)


% Load mask
mask_file = fullfile('/scratch', 'ebron', 'ADNI', 'Template_space', 'brain_mask_in_template_space.nii.gz');
mask = load_nii(mask_file);

% Load labels
label_file = fullfile('/scratch', 'ebron', 'ADNI', 'adni_ad_t1_baseline_masked.csv');
[PTID Age Sex ICV Diagnosis images data_filelist] = csvimport(label_file, 'columns', {'PTID', 'Age', 'Sex', 'ICV', 'Diagnosis', 'images', 'hdf5'});

% Labels should be -1 and 1
labels = ones(size(Diagnosis));
x_cn = find(strncmp(Diagnosis,'CN',2));
labels(x_cn) = -1;

% Prior of positive class
p=sum(labels==1)/max(size(labels));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ctr=1;

m=mean(CF_MATSC);


[model]=svmtrain(labels,double(CF_MATSC),['-s 0 -t 0 -c ' num2str(c_par)]);
MODEL_SVM.model=model;
MODEL_SVM.M{1}=m;
clear model
clear m
% MODEL_SVM.F_MAT=F_MAT;

save(fullfile(output_directory, 'model_svm_t1.mat'), 'MODEL_SVM');

w = MODEL_SVM.model.SVs' * MODEL_SVM.model.sv_coef;
X=CF_MATSC;
[r,c]=size(CF_MATSC);
clear CF_MATSC;
J=ones(r,1);
size(X)
if pmap_offset
    K=X*X'+0.00001*eye(size(X,1));
else
    K=X*X';
end
Z=inv(K)+(inv(K)*J*inv(-J'*inv(K)*J)*J'*inv(K));
C=X'*Z;

SD=sqrt(sum(C.^2,2)*(4*p-4*p^2));
mean_w=sum(C,2)*(2*p-1); 

%NEW LINES pmap 2.0 met margins
%Analytic distribution predicted by formulation
%Denominators
w_star=w;
D=sqrt(sum(SD.*SD+mean_w.*mean_w));
mean_normw=mean_w/D;
SD_normw=SD/D;
normw_star=w_star./norm(w_star);
pmap=2*normcdf(-abs(normw_star-mean_normw),zeros(size(w_star)),SD_normw);

save(fullfile(output_directory, 'pmap_t1.mat'), 'pmap');