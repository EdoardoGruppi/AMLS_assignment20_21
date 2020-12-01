# Import package
from pathlib import Path
import os
import cv2


# Faces ================================================================================================================
def viola_jones_faces(data_directory, min_size=(70, 70), scale_factor=1.05, min_neighbors=3):
    """
    Extract faces via different viola_jones algorithms and save them into the folder 
    './Datasets/data_directory_viola/img'
    
    :param data_directory: name of the folder where the images are placed.
    :param min_size: min_size parameter of the detectMultiScale() function. default_value=(70,70). 
    :param scale_factor: scale_factor parameter of the detectMultiScale() function. default_value=1.05. 
    :param min_neighbors: min_neighbors parameter of the detectMultiScale() function. default_value=3.
    :return: the list and the number of files where faces are not detected by viola-jones algorithms.
    """
    # Dataset path
    path = os.path.join('./Datasets', data_directory)
    dataset_dir = os.path.join(path, 'img')
    # Create directory for extracted faces images
    viola_dir = os.path.join(path + '_viola', 'img')
    Path(viola_dir).mkdir(parents=True, exist_ok=True)
    # List of all the images available
    files = os.listdir(dataset_dir)
    # Extract face for each image in the directory
    result = []
    cnt = 0
    for file in files:
        image_path = os.path.join(dataset_dir, file)
        # imread() converts the input image to an OpenCV object.
        image = cv2.imread(image_path)
        # A common practice in image processing is to first convert the input image to gray scale. This is because
        # detecting luminance, as opposed to color, will generally yield better results in object detection.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Load two different viola-jones solutions.
        # Only in the case the first does not work the second will be involved.
        faceCascade_profile = cv2.CascadeClassifier("haarcascade_profileface.xml")
        faceCascade_frontal_alt = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
        # Define hyper parameters of the viola-jones algorithms. detectMultiScale() generates a series of rectangles
        # in the form of Rect(x,y,w,h) for every detected face in the image.
        faces = faceCascade_frontal_alt.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size
        )
        # If the first does not detect any face...
        if len(faces) == 0:
            faces = faceCascade_profile.detectMultiScale(
                gray,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=min_size
            )
        # If a face is found extract and save it in the folder dedicated.
        for (x, y, w, h) in faces:
            extracted_image = image[y:y+h, x:x+w]
            resized = cv2.resize(extracted_image, (100, 100))
            cv2.imwrite(os.path.join(viola_dir, file), resized)
        # If no faces are found notify it and insert the file in a related list
        if len(faces) == 0:
            cnt += 1
            print("In {0}, {1} face found!!  -------------- counter: {2}".format(file, len(faces), cnt))
            result.append(file)
    # Return the list of files where faces are not detected by viola-jones algorithms and its length.
    return len(result), result

# num_examples, results = viola_jones_faces('celeba', min_size=(40, 40), scale_factor=1.2, min_neighbors=0)
# print(results)
# print(num_examples)


# Smile ================================================================================================================
def viola_jones_smile(data_directory, min_size=(30, 30), scale_factor=1.3, min_neighbors=5):
    """
    Extract smiles through a specific viola_jones algorithm and save them into the folder
    './Datasets/data_directory_viola/img'

    :param data_directory: name of the folder where the images are placed.
    :param min_size: min_size parameter of the detectMultiScale() function. default_value=(70,70).
    :param scale_factor: scale_factor parameter of the detectMultiScale() function. default_value=1.05.
    :param min_neighbors: min_neighbors parameter of the detectMultiScale() function. default_value=3.
    :return:
    """
    # Dataset path
    path = os.path.join('./Datasets', data_directory)
    dataset_dir = os.path.join(path, 'img')
    # Create directory for extracted faces images
    viola_dir = os.path.join(path + '_viola', 'img')
    Path(viola_dir).mkdir(parents=True, exist_ok=True)
    # List of all the images available
    files = os.listdir(dataset_dir)
    # Extract smile for each image in the directory
    cnt = 0
    for file in files:
        image_path = os.path.join(dataset_dir, file)
        # imread() converts the input image to an OpenCV object.
        image = cv2.imread(image_path)
        # A common practice in image processing is to first convert the input image to gray scale. This is because
        # detecting luminance, as opposed to color, will generally yield better results in object detection.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faceCascade = cv2.CascadeClassifier("haarcascade_smile.xml")
        # Define hyper parameters of the viola-jones algorithms. detectMultiScale() generates a series of rectangles
        # in the form of Rect(x,y,w,h) for every detected face in the image.
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size
        )
        # If a smile is found extract and save it in the folder dedicated
        for (x, y, w, h) in faces:
            extracted_image = image[y:y + h, x:x + w]
            resized = cv2.resize(extracted_image, (40, 40))
            cv2.imwrite(os.path.join(viola_dir, file), resized)
        # If no smiles are found notify it and insert the file in a related list
        if len(faces) == 0:
            cnt += 1
            print("In {0}, {1} face found!!  -------------- counter: {2}".format(file, len(faces), cnt))

# viola_jones_smile('celeba_faces')
