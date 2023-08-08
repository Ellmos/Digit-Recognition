import sys
from random import randint, random
import numpy as np
import cv2
import struct
from array import array
import os


def RotateImage(image, angle):
    imageCenter = tuple(np.array(image.shape[1::-1]) / 2)
    rotMat = cv2.getRotationMatrix2D(imageCenter, angle, 1.0)
    rotated = cv2.warpAffine(image, rotMat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return rotated


def TranslateImage(image, x, y):
    M = np.float32([[1, 0, x],
                    [0, 1, y]])
    return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

def ShowImage(image):
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def ReadMnistFiles(imagesPath, labelsPath):
    labels = []
    # Read labels file
    with open(labelsPath, 'rb') as file:
        magic, size = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
        labels = array("B", file.read())

    # Read images file
    with open(imagesPath, 'rb') as file:
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
        image_data = array("B", file.read())

    arraySize = rows * cols
    images = [np.array(image_data[i * arraySize: (i+1) * arraySize]) for i in range(size)]

    return np.array(images, dtype=">u1"), np.array(labels, dtype=">u1")


def ModifyDataset(images, labels, isTrainDataSet, nbrNewImage):
    newDirectory = os.path.dirname(os.path.abspath(__file__)) + '/modifiedMnist/'
    if isTrainDataSet:
        imageFile = open(newDirectory + "train-images.idx3-ubyte", "wb")
        labelFile = open(newDirectory + "train-labels.idx1-ubyte", "wb")
    else:
        imageFile = open(newDirectory + "t10k-images.idx3-ubyte", "wb")
        labelFile = open(newDirectory + "t10k-labels.idx1-ubyte", "wb")


    # ----------------------------Beginning of the idx files------------------------------#
    nbrOriginalImage = len(images)
    # Bytes at the beginning of a mnist-like idx files
    # first 4 bytes is magic number
    #   first 2 bytes are 0
    #   3rd byte is dataType (\x08 = unsigned int)
    #   4th byte number of dimensions of the vector/matrix: 0 for single, 1 for vectors, 2 for matrices....
    imageMagicNumber = b"\x00\x00\x08\x01"
    labelMagicNumber = b"\x00\x00\x08\x03"

    # The sizes of each dimension are 4-byte integers (in big endian)
    nbrNewImagesBytes = int(nbrNewImage).to_bytes(4, "big")
    _28 = int(28).to_bytes(4, "big")
    if isTrainDataSet:
        labelFile.write(imageMagicNumber + nbrNewImagesBytes)
        imageFile.write(labelMagicNumber + nbrNewImagesBytes + _28 + _28)
    else:
        labelFile.write(imageMagicNumber + nbrNewImagesBytes)
        imageFile.write(labelMagicNumber + nbrNewImagesBytes + _28 + _28)


    # ----------------------------Modification and writing of image in file------------------------------#
    ROTATIONRANGE = 10
    TRANSLATIONRANGE = 4
    for i in range(nbrNewImage):
        index = i % nbrOriginalImage
        image = images[index].reshape((28, 28))

        # Apply random rotation
        angle = randint(-ROTATIONRANGE, ROTATIONRANGE)
        rotated = RotateImage(image, angle)

        # Apply random translation
        xTranslation, yTranslation = randint(-TRANSLATIONRANGE, TRANSLATIONRANGE), randint(-TRANSLATIONRANGE, TRANSLATIONRANGE)
        translated = TranslateImage(rotated, xTranslation, yTranslation)

        # Add random noise and clamp values between 0 and 255
        noise = np.zeros(image.shape, np.uint8)
        cv2.randn(noise, 0, random()*50)
        noise = np.maximum(noise, 0)
        noised = cv2.add(translated, noise)
        final = np.minimum(noised, 255).reshape(784)

        imageFile.write(final.tobytes("C"))
        labelFile.write(labels[index].tobytes("C"))

        if i % 1000 == 0:
            print(i, "/", nbrNewImage)

    imageFile.close()
    labelFile.close()


def ModifyMnist(lenTraining, lenTest):
    directory = os.path.dirname(os.path.abspath(__file__)) + '/mnist/'
    training_images_path = directory + 'train-images.idx3-ubyte'
    training_labels_path = directory + 'train-labels.idx1-ubyte'
    test_images_path = directory + 't10k-images.idx3-ubyte'
    test_labels_path = directory + 't10k-labels.idx1-ubyte'

    trainingImages, trainingLabels = ReadMnistFiles(training_images_path, training_labels_path)
    testImages, testLabels = ReadMnistFiles(test_images_path, test_labels_path)

    if lenTraining != 0:
        ModifyDataset(trainingImages, trainingLabels, isTrainDataSet=True, nbrNewImage=lenTraining)
    if lenTest != 0:
        ModifyDataset(testImages, testLabels, isTrainDataSet=False, nbrNewImage=lenTest)

    exit()


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 0:
        print("No parameter given using default: 60000 10000\nFor help use --usage as parameter")
        ModifyMnist(60000, 10000)
    
    elif len(args) == 1 and args[0] == "--usage":
        sys.exit("Needs 2 parameters\nFirst parameter is number of images for training DataSet (use 0 to skip)\nSecond paramater is number of images for test DataSet (use 0 to skip)\nDefault: 60000 10000")

    elif len(args) != 2:
        sys.exit("Invalid parameter number: needs 2\nFirst parameter is number of images for training DataSet (use 0 to skip)\nSecond paramater is number of images for test DataSet (use 0 to skip)\nDefault: 60000 10000")
        
    try:
        int(args[0])
        int(args[1])
    except ValueError:
        sys.exit("Invalid type: parameters needs to be int")

    nbrImagesTrain = int(args[0])
    nbrImagesTest = int(args[1])
    if nbrImagesTrain < 0 or nbrImagesTest < 0:
        sys.exit("Invalid type: parameters needs to be positive (or 0 to skip)")

    ModifyMnist(nbrImagesTrain, nbrImagesTest)
