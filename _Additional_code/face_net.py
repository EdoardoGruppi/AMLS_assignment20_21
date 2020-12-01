# Import packages
from pathlib import Path
import os
from mtcnn.mtcnn import MTCNN
import cv2


def face_net(data_directory, required_size=(100, 100)):
    """
    It extracts a single face from a given image via a MTCNN detector. It requires an important amount of time and
    computational resources.

    :param data_directory: name of the folder where the images are placed.
    :param required_size: shaped of the face images saved. default_value=(100,100).
    :return: list of files where faces are not detected.
    """
    # Dataset path
    path = os.path.join('./Datasets', data_directory)
    dataset_dir = os.path.join(path, 'img')
    # Create directory for extracted faces images
    viola_dir = os.path.join(path + '_faceNet', 'img')
    Path(viola_dir).mkdir(parents=True, exist_ok=True)
    # List of all the images available
    files = sorted(os.listdir(dataset_dir), key=lambda x: int(x.split(".")[0]))
    cnt = 0
    result = []
    # Extract face for each image in the directory
    for file in files:
        # Create the image path
        image_path = os.path.join(dataset_dir, file)
        # Load image from file
        image = cv2.imread(image_path)
        # Create the detector
        detector = MTCNN()
        # Detect faces in the given image
        results = detector.detect_faces(image)
        # If no faces are found notify it and insert the file in a related list
        if len(results) == 0:
            cnt += 1
            print("In {0}, {1} face found!!  -------------- counter: {2}".format(file, len(results), cnt))
            result.append(file)
        # Otherwise extract the region of interest from the image
        else:
            x1, y1, width, height = results[0]['box']
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height
            # Extract the face
            extracted_image = image[y1:y2, x1:x2]
            # Resize image extracted to the required size
            resized = cv2.resize(extracted_image, required_size)
            # Save image extracted and reshaped
            cv2.imwrite(os.path.join(viola_dir, file), resized)
            print("Saving {0}.".format(file))
    # Return list of files where faces are not detected
    return result

# face_net('celeba')
