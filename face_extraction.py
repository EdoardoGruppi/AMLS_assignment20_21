# Import packages
from pathlib import Path
import cv2
import os
from face_recognition import face_locations
from shutil import copy2


# todo implement color_mode or let only the gray mode?
def faces_recognition(dataset_directory, img_size=(96, 96)):
    # Dataset path
    path = './Datasets/{}'.format(dataset_directory)
    dataset_dir = '{}/img'.format(path)
    # # Create the name of the data_directory to return
    data_directory = '{}_faces'.format(path)
    # Create directory for extracted faces images
    faces_directory = '{}/img'.format(data_directory)
    Path(faces_directory).mkdir(parents=True, exist_ok=True)
    # copy the labels.csv file into the new folder
    copy2('{}/labels.csv'.format(path), data_directory)
    # List of all the images available
    files = sorted(os.listdir(dataset_dir), key=lambda x: int(x.split(".")[0]))
    # Extract face for each image in the directory
    cnt = 0
    files_not_detected = []
    for file in files:
        image_path = os.path.join(dataset_dir, file)
        # Load the jpg file into a numpy array
        image = cv2.imread(image_path)
        # convert to gray
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Find all the faces in the image using a pre-trained convolutional neural network.
        # This method is more accurate than the default HOG model, but it's slower
        # unless you have an nvidia GPU and dlib compiled with CUDA extensions. But if you do,
        # this will use GPU acceleration and perform well.
        face_places = face_locations(gray, number_of_times_to_upsample=0, model="hog")
        if len(face_places) == 0:
            face_places = face_locations(gray, number_of_times_to_upsample=0, model="cnn")
        if len(face_places) == 0:
            cnt += 1
            print("In {0}, {1} face found!! --------------- counter: {2}".format(file, len(face_places), cnt))
            files_not_detected.append(file)
        else:
            # instead of ...for face_place in face_places
            # For each image only one detected face will be considered
            # Print the location of each face in this image
            bottom, right, top, left = face_places[0]
            # You can access the actual face itself like this:
            face_image = image[bottom:top, left:right]
            resized = cv2.resize(face_image, img_size)
            cv2.imwrite(os.path.join(faces_directory, file), resized)
    return data_directory.split('/')[-1], files_not_detected

