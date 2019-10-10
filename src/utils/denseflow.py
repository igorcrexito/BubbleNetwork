import os,sys
import numpy as np
import cv2
from PIL import Image
from multiprocessing import Pool
import argparse
from IPython import embed #to debug
import scipy.misc
import glob

#videos_root = "D:/HMDB51/videos/brush_hair/"

def ToImg(raw_flow,bound):
    '''
    this function scale the input pixels to 0-255 with bi-bound

    :param raw_flow: input raw pixel value (not in 0-255)
    :param bound: upper and lower bound (-bound, bound)
    :return: pixel value scale from 0 to 255
    '''
    margin = 10
    flow=raw_flow
    flow[flow>margin]=margin
    flow[flow<-margin]=-margin
    flow-=-margin
    
    #max_val = lambda x: max(max(x.flatten()), abs(min(x.flatten())))
    #min_val = lambda x: min(min(x.flatten()), abs(max(x.flatten())))
    #flow = flow / max_val(flow)
    
    #max_val = max(flow.flatten())
    #min_val = min(flow.flatten())
    
    #flow[:] = [x - min_val for x in flow] 
    #coefficient = 255/max_val
    #flow[:] = [x * coefficient for x in flow]
    
    flow*=(255/float(2*margin))
    return flow

def rescale_flow(image):
    max = image.max()
    min = image.min()
    
    image[:] = [x - min for x in image]
    
    coefficient = 255/max
    image[:] = [x * coefficient for x in image]
    
    #image = np.true_divide(image, max)
    #image = np.multiply(image, 255)
    return image

def save_flows(flows,image,save_dir,num,bound):
    '''
    To save the optical flow images and raw images
    :param flows: contains flow_x and flow_y
    :param image: raw image
    :param save_dir: save_dir name (always equal to the video id)
    :param num: the save id, which belongs one of the extracted frames
    :param bound: set the bi-bound to flow images
    :return: return 0
    '''
    #rescale to 0~255 with the bound setting
    flow_x=ToImg(flows[...,0],bound)
    flow_y=ToImg(flows[...,1],bound)
    if not os.path.exists(os.path.join(data_root,new_dir,save_dir)):
        os.makedirs(os.path.join(data_root,new_dir,save_dir))

    #save the image
    save_img=os.path.join(data_root,new_dir,save_dir,'img_{:05d}.jpg'.format(num))
    scipy.misc.imsave(save_img,image)

    #save the flows
    save_x=os.path.join(data_root,new_dir,save_dir,'flow_x_{:05d}.jpg'.format(num))
    save_y=os.path.join(data_root,new_dir,save_dir,'flow_y_{:05d}.jpg'.format(num))
    flow_x_img=Image.fromarray(flow_x)
    flow_y_img=Image.fromarray(flow_y)
    scipy.misc.imsave(save_x,flow_x_img)
    scipy.misc.imsave(save_y,flow_y_img)
    return 0

def dense_flow(video_list, activity):

    base_path = "D:/HMDB51/videos/" + activity + "/" 

    for i in range(0, len(video_list)):
        
        #capture video information
        videocapture=cv2.VideoCapture(base_path + video_list[i])
        print(base_path + video_list[i])
        frame_width = int(videocapture.get(3))
        frame_height = int(videocapture.get(4))
        len_frame=number_of_frames = int(videocapture.get(cv2.CAP_PROP_FRAME_COUNT))
        
        #instantiate a video writer
        out = cv2.VideoWriter(base_path + "/flow/" "flow_" +video_list[i], cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))
        
        #iterate over frames
        image,prev_image,gray,prev_gray=None,None,None,None
        frame_num = 0
        for ziv in range(0, len_frame):
            ret, frame = videocapture.read() #reading frames

            #getting first frame
            if frame_num==0:
                prev_image=frame
                #cv2.imshow('image', frame)
                prev_gray=cv2.cvtColor(prev_image,cv2.COLOR_RGB2GRAY)
                frame_num+=1
                continue

            image=frame
            
            blank_image = np.zeros((frame_height,frame_width,3), np.uint8)
            
            try:
                gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
                frame_0=prev_gray
                frame_1=gray

                ##default choose the tvl1 algorithm
                dtvl1=cv2.createOptFlow_DualTVL1()
                flowDTVL1=dtvl1.calc(frame_0,frame_1,None)
                
                flow_x=ToImg(flowDTVL1[...,0],bound)
                flow_y=ToImg(flowDTVL1[...,1],bound)
                #flow_x_img=Image.fromarray(flow_x)
                #flow_y_img=Image.fromarray(flow_y)
                
                #flow_x_img = rescale_flow(flow_x_img)
                #flow_y_img = rescale_flow(flow_y_img)
                
                #print(np.shape(flowDTVL1[...,0]))
                blank_image[:,:,0] = flow_x
                blank_image[:,:,1] = np.zeros_like(flow_x)
                blank_image[:,:,2] = flow_y
                
                
                out.write(blank_image)
                #save_flows(flowDTVL1,image,save_dir,frame_num,bound) #this is to save flows and img.
                prev_gray=gray
                prev_image=image
                frame_num+=1
            except:
                print("frame is already in grayscale")
                
            

        out.release()
        videocapture.release()

def get_video_list():
    video_list=[]
    for cls_names in os.listdir(videos_root):
        cls_path=os.path.join(videos_root,cls_names)
        for video_ in os.listdir(cls_path):
            video_list.append(video_)
    video_list.sort()
    return video_list,len(video_list)

if __name__ =='__main__':

    activity = "cartwheel"
    videos_root='D:/HMDB51/videos/' + activity + "/" 
    print(videos_root)
    
    #specify the augments
    num_workers=1
    step=1
    bound=15
    s_=0
    e_=13320
    new_dir="flows"
    mode="run"
    
    #retrieving dataset files
    os.chdir(videos_root)
    
    #retrieving the list of interest videos
    list_of_videos = glob.glob("*.avi")
    #print(len(list_of_videos))
    dense_flow(list_of_videos, activity)
    