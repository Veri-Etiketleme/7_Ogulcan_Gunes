import cv2, os

def image_getter(base_dir='./leapGestRecog'):
    for part in os.listdir(base_dir):
        for img_class in os.listdir(os.path.join(base_dir,part)):
            if '_moved' in img_class:
                if 'fist' in img_class:
                    os.system(f'mv {os.path.join(base_dir,part,img_class)}/* {os.path.join(base_dir,part,"03_fist")}')
                else:
                    os.system(f'mv {os.path.join(base_dir,part,img_class)}/* {os.path.join(base_dir,part,"01_palm")}')
                    
                os.system(f'rm -rf {os.path.join(base_dir,part,img_class)}')
                    
    for part in os.listdir(base_dir):
        for img_class in os.listdir(os.path.join(base_dir,part)):
            for i in os.listdir(os.path.join(base_dir,part,img_class)):
                image = cv2.imread(os.path.join(base_dir,part,img_class,i),cv2.IMREAD_GRAYSCALE)
                clahe = cv2.createCLAHE(16,(24,64))
                new_img = clahe.apply(image)
                cv2.imwrite(os.path.join(base_dir,part,img_class,i),new_img)

                
                
image_getter()      
