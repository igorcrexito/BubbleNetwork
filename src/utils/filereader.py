import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import cv2
import keras
import glob
from sklearn.utils import class_weight
import math

resized_width = 112
resized_height = 112
channels = 3
final_number_of_frames = 16
appearance_dimension = 4096

def read_flow_videos(video_path, clip_size, number_of_timesteps, protocol_list):
    
    train_counter = 0
    test_counter = 0
    
    video_path = video_path + "/flow/"
    
    #retrieving dataset files
    os.chdir(video_path)
    
    #retrieving the list of interest videos
    list_of_videos = glob.glob("flow*")
    print(len(list_of_videos))
    
    #retrieving number of samples in the folder
    number_of_samples = len(list_of_videos)
    number_of_training_samples, number_of_testing_samples = analyze_protocol_file(protocol_list) 
    sample_list_train = np.zeros((number_of_training_samples, number_of_timesteps* clip_size, resized_width, resized_height, channels))    
    sample_list_test = np.zeros((number_of_testing_samples, number_of_timesteps* clip_size, resized_width, resized_height, channels))
    
    #sample_list_train = np.zeros((number_of_training_samples, number_of_timesteps, clip_size, resized_width, resized_height, channels))    
    #sample_list_test = np.zeros((number_of_testing_samples, number_of_timesteps, clip_size, resized_width, resized_height, channels))
    
    #iterating over videos
    for vid in range(0, number_of_samples):
        
        video_name = list_of_videos[vid]
        #print(video_name)
        capture = cv2.VideoCapture(video_path + video_name)
        number_of_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        
        #instantiating a frame list
        frame_list = []
        
        #retrieving video frames
        for i in range (0, number_of_frames):
            ret, frame = capture.read() #reading frames
            
            frame = cv2.resize(frame, (resized_width, resized_height)) #adjusting frame size
            frame_list.append(frame)
            
            #cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        #subsampling video to a fixed number of frames
        frame_list = subsample_video(frame_list, number_of_frames, final_number_of_frames)
        #video = reshape_video(frame_list, final_number_of_frames, number_of_timesteps, clip_size, resized_width, resized_height, channels)
        #print(np.shape(frame_list))
        
        
        # When everything done, release the video capture object
        capture.release()
        if (check_video_split(protocol_list, video_name, 1) == 1):
            #sample_list_train[train_counter,:] = video
            sample_list_train[train_counter,:] = frame_list
            train_counter+=1
        elif(check_video_split(protocol_list, video_name, 1) == 2):
            #sample_list_test[test_counter,:] = video
            sample_list_test[test_counter,:] = frame_list
            test_counter+=1

    return sample_list_train, sample_list_test

def read_appearance_files(appearance_path, clip_size, number_of_timesteps, protocol_list):

    train_counter = 0
    test_counter = 0
    
    #retrieving dataset files
    os.chdir(appearance_path)
    
    #retrieving the list of interest videos
    list_of_appearance_files = glob.glob("*.txt")
    #print(len(list_of_videos))
    
    #retrieving number of samples in the folder
    number_of_samples = len(list_of_appearance_files)
    number_of_training_samples, number_of_testing_samples = analyze_protocol_file(protocol_list) 
    sample_list_train = np.zeros((number_of_training_samples, number_of_timesteps* clip_size, appearance_dimension))    
    sample_list_test = np.zeros((number_of_testing_samples, number_of_timesteps* clip_size, appearance_dimension))    
    #sample_list_train = np.zeros((number_of_training_samples, number_of_timesteps, appearance_dimension))    
    #sample_list_test = np.zeros((number_of_testing_samples, number_of_timesteps, appearance_dimension))    
    
    
    #print(len(number_of_samples))
    for vid in range(0, number_of_samples):
        
        #retrieve videos from list of txt files
        video_name = list_of_appearance_files[vid]
        #print(video_name)
        
        #compose the path to the activations file
        activations = appearance_path + video_name
        
        #instantiating a frame list
        frame_list = []
        
        #retrieving activations from files
        appearance_file = tuple(open(activations, 'r'))
        number_of_frames = len(appearance_file)
        
        for i in range(0, len(appearance_file)):
            line = appearance_file[i]
            line = line[:-2] #just removing a final blank space
            data = line.split(' ')
            frame_list.append(data)

        #subsample list to gather only a specific number of frames
        subsampled_list = subsample_video(frame_list, number_of_frames, final_number_of_frames)
        #print(np.shape(subsampled_list))
        #video = reshape_appearance(subsampled_list, final_number_of_frames, number_of_timesteps, clip_size, appearance_dimension)
        
        if (check_video_split(protocol_list, video_name, 0) == 1):
            #sample_list_train[train_counter,:] = video
            sample_list_train[train_counter,:] = subsampled_list
            train_counter+=1
        elif(check_video_split(protocol_list, video_name, 0) == 2):
            #sample_list_test[test_counter,:] = video
            sample_list_test[test_counter,:] = subsampled_list
            test_counter+=1
    
    #for i in range(0, len(sample_list_train)):
    #    print(sample_list_train[0][0])
    return sample_list_train, sample_list_test
        
def reshape_video(frame_list, number_of_frames, number_of_timesteps, clip_size, resized_width, resized_height, channels):
    
    #instantiate an array to store reshaped videos
    sample_list = np.zeros((number_of_timesteps, clip_size, resized_width, resized_height, channels), dtype=np.uint8)    

    #variables to control positioning
    current_timestep = 0
    current_clip = 0
    
    for i in range(0, number_of_frames):
        
        sample_list[current_timestep, current_clip,:] = frame_list[i]
        current_clip = current_clip + 1
        
        if (current_clip == clip_size):
            current_clip = 0
            current_timestep = current_timestep + 1

    return sample_list

def reshape_appearance(frame_list, number_of_frames, number_of_timesteps, clip_size, appearance_dimension):
    
    #instantiate an array to store reshaped videos
    sample_list = np.zeros((number_of_timesteps, clip_size, appearance_dimension))    

    #variables to control positioning
    current_timestep = 0
    current_clip = 0
    
    for i in range(0, number_of_frames):
        
        sample_list[current_timestep, current_clip,:] = frame_list[i]
        current_clip = current_clip + 1
        
        if (current_clip == clip_size):
            current_clip = 0
            current_timestep = current_timestep + 1

    #print(np.shape(sample_list))
    video_list = []
    video_list.append(sample_list[0,0])
    video_list.append(sample_list[1,0])
    video_list.append(sample_list[2,0])
    video_list.append(sample_list[3,0])
    
    return video_list
    
def subsample_video(video, number_of_frames, final_number_of_frames):
    
    #instantiate a frame list
    frame_list = []
    
    #compute the sampling coefficient
    sampling_coefficient = math.floor(number_of_frames/final_number_of_frames)
    
    #iterate over frames
    sampling_counter = 0
    for i in range(0, number_of_frames):
        if (i == 0):
            frame_list.append(video[i])
            sampling_counter = sampling_counter + 1
        else:
            if (sampling_counter == sampling_coefficient):
                frame_list.append(video[i])
                sampling_counter = 1
            else:
                sampling_counter = sampling_counter + 1
            
    #checking if it satisfies the size
    #if (len(frame_list) < final_number_of_frames):
    #    frame_list.append(video[len(video)-1])
    frame_list = frame_list[0:final_number_of_frames]
    
    #print(str(sampling_coefficient) + ' ' + str(number_of_frames) + ' ' + str(len(frame_list)))
    return frame_list

def read_protocol_path(base_protocol_path, split, video_path):
    protocol_path = base_protocol_path + str(split) + "/"
    os.chdir(protocol_path)
    #print(protocol_path)
    
    data_list = []
        
    protocol = protocol_path + video_path + "_test_split" + str(split) + ".txt"
    #print(protocol)
        
    #retrieving activations from files
    protocol_file = tuple(open(protocol, 'r'))
    number_of_videos = len(protocol_file)
    
    for i in range(0, number_of_videos):
            line = protocol_file[i]
            data = line.split(' ')
            data_list.append(data)
            
    return data_list

def analyze_protocol_file(protocol_file):
    
    training_samples = 0
    testing_samples = 0
    
    for i in range(0, len(protocol_file)):
        data = protocol_file[i]
        if int(data[1]) == 1:
            training_samples+=1
        elif int(data[1]) == 2:
            testing_samples+=1
    
    return training_samples, testing_samples

def check_video_split(protocol_file, video_name, flow):
    
    split = 0
    if flow == 1:
        video_name = video_name[5:]
    else:
        video_name = video_name[:-4]
        video_name = video_name+".avi"
    
    for i in range(0, len(protocol_file)):
        data = protocol_file[i]
        if (video_name == data[0]):
            split = int(data[1])
            break
        
    return split