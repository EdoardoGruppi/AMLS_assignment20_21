# Import packages
from pathlib import Path
import cv2
import os
from face_recognition import face_locations


def faces_recognition(data_directory, img_size=(100, 100)):
    # Dataset path
    path = './Datasets/{}'.format(data_directory)
    dataset_dir = '{}/img'.format(path)
    # Create directory for extracted faces images
    face_rec_dir = '{}_face_rec/img'.format(path)
    Path(face_rec_dir).mkdir(parents=True, exist_ok=True)
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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Find all the faces in the image using a pre-trained convolutional neural network.
        # This method is more accurate than the default HOG model, but it's slower
        # unless you have an nvidia GPU and dlib compiled with CUDA extensions. But if you do,
        # this will use GPU acceleration and perform well.
        face_places = face_locations(image, number_of_times_to_upsample=0, model="hog")
        if len(face_places) == 0:
            face_places = face_locations(image, number_of_times_to_upsample=0, model="cnn")
        if len(face_places) == 0:
            cnt += 1
            print("In {0}, {1} face found!!  -------------- counter: {2}".format(file, len(face_places), cnt))
            files_not_detected.append(file)
        for face_place in face_places:
            # Print the location of each face in this image
            top, right, bottom, left = face_place
            # You can access the actual face itself like this:
            face_image = image[top:bottom, left:right]
            resized = cv2.resize(face_image, img_size)
            cv2.imwrite(os.path.join(face_rec_dir, file), resized)
    return (len(files) - len(files_not_detected)), files_not_detected


