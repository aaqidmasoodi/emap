import cv2
import base64
import torch
import os
import numpy as np
import tensorflow as tf
from keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.metrics.pairwise import cosine_similarity

# Ear Detection Models for Automatic Detection and Cropping of ear image
weights = os.path.join(os.path.dirname(os.path.realpath(__file__)), '', './detector/best.pt')
model_ear_detect = torch.hub.load('ultralytics/yolov5', 'custom', path=weights)

# Feature Extraction Model
model_feature_extract = VGG16(weights='imagenet', include_top=False)

# Noise Filter
# Removes the noise that can be caused by different lighting conditions 
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# Feature extraction of ear image
def extract_features(image):
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe_image = clahe.apply(gray_image)
    stacked_image = np.stack((clahe_image,)*3, axis=-1)
    preprocessed_image = preprocess_input(stacked_image)
    ready_image = tf.expand_dims(preprocessed_image, axis=0)

    # generate features using vgg16
    features = model_feature_extract.predict(ready_image)

    # encode the features as 'bytes-utf-8'
    encoded_features = base64.b64encode(features).decode('utf-8')

    return encoded_features

def get_cosine_similarity_score(feature_set_1, feature_set_2):

    similarity = cosine_similarity(feature_set_1.reshape(1,-1), feature_set_2.reshape(1,-1))
    
    return similarity


def find_match(features_current, ears_all):

    current_match = [None, 0]
    features_current = base64.b64decode(features_current)
    features_current = np.frombuffer(features_current, dtype='float32').reshape((1, 7, 7, 512))

    '''
        this loop will loop over all the stored ears in the database
        and search for a match with the current provided ear
    '''
    
    for ear in ears_all:

        # select all features for a particular ear
        db_ear_features_all = [
            base64.b64decode(ear.features_1),
            base64.b64decode(ear.features_2),
            base64.b64decode(ear.features_3),
            base64.b64decode(ear.features_4),
        ]

        for db_ear_features in db_ear_features_all:
            
            # read the data as np array
            db_ear_features = np.frombuffer(db_ear_features, dtype='float32').reshape((1, 7, 7, 512))

            # this will grab the similarity score between the two images.
            similarity = get_cosine_similarity_score(features_current, db_ear_features)
    
            if similarity > .65:
                # current_match[1] refers to the similarity score
                if current_match[1] < similarity:
                    current_match = [ear, similarity]

    return current_match

# Ear Detection Modal
# this functions prepares the crop coordinates for the image around the ear
def denormalize(x_norm, y_norm, w_norm, h_norm, image_shape):
    """Converts normalized bounding box coordinates to denormalized pixel coordinates."""
    img_width, img_height = image_shape[1], image_shape[0]
    x_pixel = int(x_norm * img_width)
    y_pixel = int(y_norm * img_height)
    w_pixel = int(w_norm * img_width)
    h_pixel = int(h_norm * img_height)
    x_min = max(0, x_pixel - w_pixel // 2)
    y_min = max(0, y_pixel - h_pixel // 2)
    x_max = min(img_width - 1, x_pixel + w_pixel // 2)
    y_max = min(img_height - 1, y_pixel + h_pixel // 2)
    return x_min, y_min, x_max - x_min, y_max - y_min

def detect_and_crop_ear(image):

    # Model
    detections = []

    # prepare image for the model
    image_rgb = image[:, :, ::-1] 

    # detect
    results = model_ear_detect(image_rgb, size=224)

    # Run Detection
    for tensor1, tensor2 in zip(results.xywhn, results.xywh):
        for result1, result2 in zip(tensor1, tensor2):
            x_norm, y_norm, w_norm, h_norm, _, _ = result1.numpy()
            x_, y_, w_, h_, _, _ = result2.numpy()
            detections.extend([denormalize(x_norm, y_norm, w_norm, h_norm, image.shape)])

    # if ear has been detected in the image
    if detections:
        x, y, w, h = detections[0]
        extend = 75
        ear_img = image[y-extend:y+h+extend, x-extend:x+w+extend]

        return ear_img[:, :, ::-1] 

    return None