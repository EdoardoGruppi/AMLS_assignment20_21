# Import packages
from pathlib import Path
import cv2
import os
from face_recognition import face_locations
from shutil import copy2


def smiles_extraction(dataset_name, img_size=(96, 48)):
    """
        Extracts smiles from images where at least one face is detected.
        The images extracted are placed in a new dedicated folder and are all in 'grayscale'.

        :param dataset_name: name (not path) of the folder where to extract smiles.
        :param img_size: desired size (a,b) of the output images where a is suggested to be two times b.
            default_value=(96, 48)
        :return: the name of the novel folder and the list of files where no smiles are detected.
    """
    # Dataset path
    path = './Datasets/{}'.format(dataset_name)
    dataset_directory = '{}/img'.format(path)
    # Create the name of the data_directory to return
    data_directory = '{}_smiles'.format(path)
    # Create directory for extracted faces images
    faces_directory = '{}/img'.format(data_directory)
    Path(faces_directory).mkdir(parents=True, exist_ok=True)
    # copy the labels.csv file into the new folder
    copy2('{}/labels.csv'.format(path), data_directory)
    # List of all the images available
    files = sorted(os.listdir(dataset_directory), key=lambda x: int(x.split(".")[0]))
    # Extract face for each image in the directory
    counter = 0
    files_not_detected = []
    for file in files:
        image_path = os.path.join(dataset_directory, file)
        # Load the jpg file into a numpy array
        gray = cv2.imread(image_path)
        # convert to gray
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        # Find faces with a model based on HOG algorithm
        face_places = face_locations(gray, number_of_times_to_upsample=0, model="hog")
        if len(face_places) == 0:
            # Find faces with a pre-trained CNN. It is more accurate than the default HOG method but it is slower.
            # With GPU and dlib compiled with CUDA extensions it will perform faster
            face_places = face_locations(gray, number_of_times_to_upsample=0, model="cnn")
        if len(face_places) == 0:
            # If no faces are detected save the name of the file in a dedicated list
            counter += 1
            print("In {0}, no faces found!! --------------- counter: {1}".format(file, counter))
            files_not_detected.append(file)
        else:
            # instead of ...for face_place in face_places
            # For each image only one detected face will be considered
            # Print the location of each face in this image
            top, right, bottom, left = face_places[0]
            # Select the region of interest in the original rgb image
            # Take only the lower half of the face
            top += round((bottom - top) / 2)
            face_image = gray[top:bottom, left:right]
            # Resize the region of interest and save in the created directory
            resized = cv2.resize(face_image, img_size)
            cv2.imwrite(os.path.join(faces_directory, file), resized)
    return data_directory.split('/')[-1], files_not_detected


def faces_extraction(dataset_name, img_size=(96, 96)):
    """
        Extracts faces if detected from images. It returns one face for every picture.
        The extracted images are placed in a new dedicated folder and are all in 'rgb'.

        :param dataset_name: name (not path) of the folder where to extract faces.
        :param img_size: desired size of the output images. default_value=(96, 96)
        :return: the name of the novel folder and the list of files where no faces are detected.
    """
    # Dataset path
    path = './Datasets/{}'.format(dataset_name)
    dataset_directory = '{}/img'.format(path)
    # Create the name of the data_directory to return
    data_directory = '{}_faces'.format(path)
    # Create directory for extracted faces images
    faces_directory = '{}/img'.format(data_directory)
    Path(faces_directory).mkdir(parents=True, exist_ok=True)
    # copy the labels.csv file into the new folder
    copy2('{}/labels.csv'.format(path), data_directory)
    # List of all the images available
    files = sorted(os.listdir(dataset_directory), key=lambda x: int(x.split(".")[0]))
    # Extract face for each image in the directory
    counter = 0
    files_not_detected = []
    for file in files:
        image_path = os.path.join(dataset_directory, file)
        # Load the jpg file into a numpy array
        image = cv2.imread(image_path)
        # convert to gray
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Find faces with a model based on HOG algorithm
        face_places = face_locations(gray, number_of_times_to_upsample=0, model="hog")
        if len(face_places) == 0:
            # Find faces with a pre-trained CNN. It is more accurate than the default HOG method but it is slower.
            # With GPU and dlib compiled with CUDA extensions it will perform faster
            face_places = face_locations(gray, number_of_times_to_upsample=0, model="cnn")
        if len(face_places) == 0:
            # If no faces are detected save the name of the file in a dedicated list
            counter += 1
            print("In {0}, no faces found!! --------------- counter: {1}".format(file, counter))
            files_not_detected.append(file)
        else:
            # instead of ...for face_place in face_places
            # For each image only one detected face will be considered
            # Print the location of each face in this image
            bottom, right, top, left = face_places[0]
            # Select the region of interest in the original rgb image
            face_image = image[bottom:top, left:right]
            # Resize the region of interest and save in the created directory
            resized = cv2.resize(face_image, img_size)
            cv2.imwrite(os.path.join(faces_directory, file), resized)
    return data_directory.split('/')[-1], files_not_detected
