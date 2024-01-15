# Draw bounding boxes on the pictures

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import os
import shutil
import re


def save_bbox(frames_dir, output_dir, bbox_txt_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for bbox_txt in os.listdir(bbox_txt_dir):
        # Get the video number and create the directory for this video if it doesn't exist
        filename = re.search(r'M(\d+)', bbox_txt).group(1)
        video_dir = f'{output_dir}/M{filename}'
        os.makedirs(video_dir, exist_ok=True)

        print(f'{output_dir}/M{filename}')

        occluded_vehicles = set()  # Keep track of occluded vehicles

        with open(os.path.join(bbox_txt_dir, bbox_txt), 'r') as f:
            for line in f:
                frame = line.strip().split(',')
                vehicle_id = int(frame[1])
                occlusion = int(frame[7])

                if occlusion > 1:
                    occluded_vehicles.add(vehicle_id)

        with open(os.path.join(bbox_txt_dir, bbox_txt), 'r') as f:
            for line in f:
                frame = line.strip().split(',')
                frame_nb = f'{int(frame[0]):06d}'
                vehicle_id = int(frame[1])
                occlusion = int(frame[7])
                bbox = list(map(int, frame[2:6]))
                image_path = f'{frames_dir}/M{filename}/img{frame_nb}.jpg'
                image = cv2.imread(image_path)
                x, y, w, h = bbox
                bounding_box_image = image[y:y+h, x:x+w]

                if occlusion > 1:
                    occluded_dir = os.path.join(video_dir, f'veh{vehicle_id}/occluded')
                    os.makedirs(occluded_dir, exist_ok=True)
                    cv2.imwrite(f'{occluded_dir}/{frame_nb}.jpg', bounding_box_image)

                if occlusion <= 1 and vehicle_id in occluded_vehicles:
                    non_occluded_dir = os.path.join(video_dir, f'veh{vehicle_id}/non-occluded')
                    os.makedirs(non_occluded_dir, exist_ok=True)
                    cv2.imwrite(f'{non_occluded_dir}/{frame_nb}.jpg', bounding_box_image)



#save_bbox('UAV-benchmark-M','Annotated_images','whole')

#Count the number of occluded and non-occluded bounding boxes and the number of vehicles in the Annotated_images directory
def count_frames(annotated_images_dir):
    occluded = 0
    non_occluded = 0
    vehicles = 0
    for video in os.listdir(annotated_images_dir):
        for vehicle in os.listdir(annotated_images_dir+'/'+video):
            vehicles += 1
            for occlusion in os.listdir(annotated_images_dir+'/'+video+'/'+vehicle):
                for image in os.listdir(annotated_images_dir+'/'+video+'/'+vehicle+'/'+occlusion):
                    if occlusion == 'occluded':
                        occluded += 1
                    else:
                        non_occluded += 1
    return occluded, non_occluded, vehicles

#occluded_frames, non_occluded_frames, nb_vehicles = count_frames('Annotated_images')

#Count the number of occluded frames and how much occlusion there is on the image
def count_occlusion(bbox_txt_dir):
    small_occlusion = 0
    medium_occlusion = 0
    large_occlusion = 0
    for video in os.listdir(bbox_txt_dir):
        with open(bbox_txt_dir + '/' + video, 'r') as f:
            for line in f:
                frame = line.strip().split(',')
                occlusion = int(frame[7])
                if occlusion == 4:
                    small_occlusion += 1
                elif occlusion == 3:
                    medium_occlusion += 1
                elif occlusion == 2:
                    large_occlusion += 1
    
    return small_occlusion, medium_occlusion, large_occlusion

#small_occlusion, medium_occlusion, large_occlusion = count_occlusion('UAV-benchmark-MOTD_v1.0/GT')

#Make a histogram with the number of occluded images per vehicle
def histogram_occlusion(annotated_images_dir, output_file):
    occluded = []
    for video in os.listdir(annotated_images_dir):
        for vehicle in os.listdir(annotated_images_dir+'/'+video):
            occluded.append(len(os.listdir(annotated_images_dir+'/'+video+'/'+vehicle+'/occluded')))
    plt.hist(occluded, bins=20)
    plt.xlabel('Number of occluded images')
    plt.ylabel('Number of vehicles')
    plt.title('Number of occluded images per vehicle')
    plt.savefig(output_file)
    plt.close()

#histogram_occlusion('Annotated_images', 'occlusion_hist.png')
                
#Check if every vehicle has at least one non-occluded bounding box
def check_non_occluded(annotated_images_dir):
    for video in os.listdir(annotated_images_dir):
        for vehicle in os.listdir(annotated_images_dir+'/'+video):
            if not os.path.exists(annotated_images_dir+'/'+video+'/'+vehicle+'/non-occluded'):
                return False
    return True


#Count the percentage of vehicles that have at least one non-occluded bounding box
def count_non_occluded(annotated_images_dir):
    non_occluded = 0
    for video in os.listdir(annotated_images_dir):
        for vehicle in os.listdir(annotated_images_dir+'/'+video):
            if os.path.exists(annotated_images_dir+'/'+video+'/'+vehicle+'/non-occluded'):
                non_occluded += 1
    return non_occluded


#Create a csv file with the number of occluded and non-occluded bounding boxes and the number of vehicles and if every vehicle has at least one non-occluded bounding box
def create_csv(annotated_images_dir, bbox_txt_dir, output_file):
    occluded_frames, non_occluded_frames, nb_vehicles = count_frames(annotated_images_dir)
    small_occlusion, medium_occlusion, large_occlusion = count_occlusion(bbox_txt_dir)
    histogram_occlusion(annotated_images_dir, 'occlusion_hist.png')
    non_occluded = count_non_occluded(annotated_images_dir)
    df = pd.DataFrame({'occluded_frames': occluded_frames, 'non_occluded_frames': non_occluded_frames, 'nb_vehicles': nb_vehicles, 'non_occluded_veh': non_occluded, 'small_occlusion': small_occlusion, 'medium_occlusion': medium_occlusion, 'large_occlusion': large_occlusion}, index=[0])
    df.to_csv(output_file, index=False)

#create_csv('Annotated_images', 'whole', 'stats.csv')




