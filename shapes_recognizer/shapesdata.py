#!/usr/bin/env python
# coding: utf-8

# # Building a Deep Neural Network
# <hr>
# 
# A DNN can be decomposed as the following sequence of operations:
# * Input Data
# * Forward Propagation to obtain an Output Data
# * Evaluate Output Data (compute current Cost)
# * Given current Cost, do Back-propagation to update weights
# * Repeat from beggining with updated weights

# In[1]:


from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


def load_images():
    """
    Arguments:
    None
    
    Returns:
    images -- python list containing image data:
                    images[line] = (original image, raw image as numpy array, shape of drawing)
                    original image -- <PIL.PngImagePlugin.PngImageFile image mode=RGB>
                    raw image -- numpy array of shape (28,28,3)
                    shape of drawing -- "circle", "square", "triangle"
    """

    folder = "shapes/"
    shapes = ["circles/","squares/", "triangles/"]
    filename = "drawing"
    extension = ".png"

    circles = np.empty((0, 2))
    squares = np.empty((0, 2))
    triangles = np.empty((0, 2))

    for i in range(1,101):
        file_path = folder + "circles/" + filename + "(" + str(i) + ")" + extension

        img = Image.open(file_path)
        raw = np.asarray(img)
        shape = np.asarray(["circle"])
        entry = np.array([raw, shape])
        circles = np.append(circles, [entry], axis=0)
        
        file_path = folder + "squares/" + filename + "(" + str(i) + ")" + extension

        img = Image.open(file_path)
        raw = np.asarray(img)
        shape = np.asarray(["square"])
        entry = np.array([raw, shape])
        squares = np.append(squares, [entry], axis=0)
        
        file_path = folder + "triangles/" + filename + "(" + str(i) + ")" + extension

        img = Image.open(file_path)
        raw = np.asarray(img)
        shape = np.asarray(["triangle"])
        entry = np.array([raw, shape])
        triangles = np.append(triangles, [entry], axis=0)

    return circles, squares, triangles


# In[3]:


def split_set(data, splits = [.7, .85]):
    """
    Arguments:
        data -- list to be splitted
        splits -- array of splitting points. Ex.: [.7, .85] will split in two points producing three sets.
    
    Returns:
        list of sets: (train_set, dev_set, test_set)
    """
    
    np.random.seed(42)
    np.random.shuffle(data)
    
    train_set = data[:70]
    dev_set = data[70:85]
    test_set = data[85:]
    
    
    return (train_set, dev_set, test_set)


def load_data():
    np.random.seed(42)
    
    circles, squares, triangles = load_images()   
    
    circles_train, circles_dev, circles_test = split_set(circles)
    squares_train, squares_dev, squares_test = split_set(squares)
    triangles_train, triangles_dev, triangles_test = split_set(triangles)
    
    train = np.concatenate([circles_train, squares_train, triangles_train], axis=0)
    np.random.shuffle(train)
    train_set, train_label = np.split(train, 2, axis=1)

    dev = np.concatenate([circles_dev, squares_dev, triangles_dev], axis=0)
    np.random.shuffle(dev)
    dev_set, dev_label = np.split(dev, 2, axis=1)

    test = np.concatenate([circles_test, squares_test, triangles_test], axis=0)
    np.random.shuffle(test)
    test_set, test_label = np.split(test, 2, axis=1)
    
    train_set = np.squeeze(train_set)
    dev_set = np.squeeze(dev_set)
    test_set = np.squeeze(test_set)
    
    train_label = np.squeeze(train_label)
    dev_label = np.squeeze(dev_label)
    test_label = np.squeeze(test_label)
    
    return (train_set, train_label, dev_set, dev_label, test_set, test_label)

def print_img(array, array_label, index):
    plt.imshow(array[index])
    print("This image is a {}".format(array_label[index][0]))
    





