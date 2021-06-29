import numpy as np
import pandas as pd
import pims
import matplotlib.pyplot as plt 
import cv2
import os
import glob
from PIL import Image, ImageDraw
from pims import ImageSequence
from skimage import io, measure
import glob
import os
import time
import pandas as pd
import numpy as np

# 这个标准只把相互重合的像素标记成蓝色， 也可以把 （i,i+1）变成（i-1，i+2) 这样可以扩大标记范围
def get_neighbor(i, j, img):
    w = img.shape[1]
    h = img.shape[0]
    
    results = []
    for y in range(j, j+1):
        for x in range(i, i+1):
            if x < 0 or x >= h:
                continue
            if y < 0 or y >= w:
                continue
            results.append([x, y])
    return results

def single_handle(img, seg, df_frame):
    # delete the blue channel
    
    ls = []
    (r, g, b) = cv2.split(img)
    zeros =np.zeros(img.shape[:2], dtype = "uint8")
    img = cv2.merge([r,g,zeros])
    
    

    R_image = np.zeros(img.shape, dtype = "uint8")
    R_image[:,:,0] = img[:,:,0]
    
    G_image = np.zeros(img.shape, dtype = "uint8")
    G_image[:,:,1] = img[:,:,1]
    
    B_image = np.zeros(img.shape, dtype = "uint8")
    B_image[:,:,2] = img[:,:,2]
    

    
    R_seg = seg
    
    
 
    g_array = np.array(g)
    result = (g_array > 90) * g_array
    #m_result = (result < 250) * result
    p_result = np.ones(result.shape) * 255
    G_seg = (result > 30) * p_result
    
    g_draw = G_image.copy()

    # contours
    
    contours = measure.find_contours(G_seg, 0.8)
    
    boundaries = list(filter(lambda x:len(x) > 10, contours))
    boundaries1 = list(filter(lambda x:len(x) > 10, contours))
    boundaries2 = []
    for a_boundary in boundaries1:
          a_boundary = np.float32(a_boundary)
          boundaries2.append(a_boundary)
    total_n = len(boundaries2)
    
    #boundaries = np.float32(boundaries)
    v_boundaries = np.vstack(boundaries)
    all_bound = np.zeros(img.shape, dtype='uint8')
    for a_bound in v_boundaries:
        i = int(round(a_bound[0]))
        j = int(round(a_bound[1]))
        
        all_bound[i, j] = [0, 255, 0]
        neighbors = get_neighbor(i, j, g_draw)
        for a_neighbor in neighbors:
            #print (a_neighbor)
            
            all_bound[a_neighbor[0], a_neighbor[1]] = [0, 255, 0]

    
    extend = np.all(all_bound == [0,255,0], axis=-1)

    red_points = np.zeros(img.shape, dtype='uint8')
    
    
    total_c = 0
     #print(df_frame.shape[0])
    #print(df_frame.head())

    T_or_F_list = []
    test_ls = []
    for p in range(0,df_frame.shape[0]):
        
        k1 = df_frame['POSITION_X'].iloc[p]
        k2 = df_frame['POSITION_Y'].iloc[p]
        
        stage = False
        a_boundaries = list(filter(lambda x:cv2.pointPolygonTest(x, (k2,k1), False) >= 0, boundaries2))
         
        if len(a_boundaries) ==0:
            pass
        else:
            
            for a_boundary in a_boundaries:    
                
               
                for a_point in a_boundary: 
                    i = int(round(a_point[0]))
                    j = int(round(a_point[1]))
                    neighbors = get_neighbor(i, j, g_draw)
                    for a_neighbor in neighbors:
                        if R_seg[a_neighbor[0], a_neighbor[1]] == 255:
                            stage = True
                              
                            red_points[i, j] = [255, 0, 0]
                            
                      
                    
                            
                         
        if stage == True:
            total_c = total_c +1 
            T_or_F_list.append(True)
        else:
            T_or_F_list.append(False)
        
            #break
    contact_ls = T_or_F_list
    
    
      
    Colored_bound = img.copy()
    #Colored_bound=cv2.merge(r, g, b)
    extend = np.all(red_points == [255,0,0], axis=-1)
    #Colored_bound[extend] = [255, 0, 0] I marked as blue
    Colored_bound[extend] = [0, 0, 225]
    img = Colored_bound
    return img, total_n, total_c, contact_ls, test_ls
    #io.imsave('dataset2_output/{}'.format(t), Colored_bound)



        


if __name__ == '__main__':
    
    imglist = glob.glob(os.path.join('./raw_data', '*tif'))
    total_N = []
    total_C = []
    dic = {'imgname': [], 'contact_times': [], 'total_particle_number': [], 'contact_ratio':[]}
    df  = pd.DataFrame(data=dic)
    
    
    for a_imglist in imglist:
        
        
        imgdir =os.path.split(a_imglist)[1]
        imgname  = imgdir.split('.')[0] 
        print (imgdir)
        seg_dir = '../segmentation_interface/segmentation/' + imgname + '*.tif'
        segs = ImageSequence(seg_dir)
        
        print('segs have ' + str(len(segs))+ ' slices')
        images= io.imread(a_imglist, plugin="tifffile")
        spots = pd.read_csv(os.path.join('./trackmate',imgname)+'auto.csv')
        ttt= os.path.join('./trackmate',imgname)+'auto.csv'
        spots['ER_contact']=' '
        spots['ER_contact']='False'
        print('Load '+ imgname + ', dataset has the shape: ' + str(spots.shape))
        # loading time
        st = time.process_time()
        contact_List = []
        test_List = []
        Shenghuan = 0
        for i in range(0,images.shape[0]):
            df_frame  =  spots[spots['FRAME']==i]
            Shenghuan = Shenghuan + df_frame.shape[0]
            #df_frame = pd.DataFrame(df_frame,index=0)
            images[i], n, c, contact_ls, test_ls = single_handle(images[i], segs[i], df_frame)
            total_N.append(n)
            total_C.append(c)
            contact_List.extend(contact_ls)
            test_List.extend(test_ls)
            p = round(i+1/images.shape[0])
            duration = round(time.process_time() - st, 2)
            remaining = round(duration * 100 / (0.01 + p) - duration, 2)
            
            print("Loading {0}%, spent time{1}s, expecting remaining time{2}s".format(p, duration, remaining), end="\r")
            

        spots['ER_contact']=contact_List
        
        sum_N = 0
        sum_C = 0
        for ps in range(10, len(total_N), int(len(total_N)/5)):
            pn_3 = 0
            pc_3 = 0
            for p in (ps-1, ps+2):
                pn_3 = pn_3 + total_N[p]
                pc_3 = pc_3 + total_C[p]
            pn_aver = int(pn_3/3)
            pc_aver = int(pc_3/3)
            sum_N = sum_N + pn_aver
            sum_C = sum_C + pc_aver
        ratio = sum_C / sum_N
        print('Finished editting img: '+ imgname)
        io.imsave('dataset_output/{}'.format(imgdir), images)
        
        df = df.append([{'imgname': imgname, 'contact_times': sum_C, 'total_particle_number':sum_N, 'contact_ratio':ratio}], ignore_index=True)
        df.to_csv("dataset_output/0703/summary.csv")
        
        spots.to_csv(ttt)


  