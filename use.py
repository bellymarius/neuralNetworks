import convert
import os
from neural_network import *
import numpy as np
import pickle
import matplotlib.pyplot as plt

net_name = "Overfitter2"
files = ["ex{}.png".format(i) for i in range(1,5)]
load = False
build_image = True
display_feature = False




def get(filename):
    "return the object save in the file filename.data"
    with open(filename,'rb') as f: 
        get_record = pickle.Unpickler(f) 
        return get_record.load()

root = os.path.dirname(__file__)
net_path = os.path.join(root,"networks", "{}.data".format(net_name))
net = get(net_path)

if load:
    path_train_images = os.path.join(root,"data", "train_images.data")
    path_train_labels = os.path.join(root,"data", "train_labels.data")
    path_test_images = os.path.join(root,"data", "test_images.data")
    path_test_labels = os.path.join(root,"data", "test_labels.data")

    training_features = convert.get(path_train_images)
    training_labels = convert.get(path_train_labels)
    testing_features = convert.get(path_test_images)
    testing_labels = convert.get(path_test_labels)






for file in files :
    examples_path = os.path.join(root,"examples",file)
    features = convert.from_PNG_to_vect(examples_path, N=255)
    if display_feature :
        print("features :\n",features)
    if build_image :
        img = features.reshape(28,28)
        plt.clf()
        plt.imshow(img,cmap = "gray")
        plt.show()
    prediction = net.predict(features)
    print("classification : {}\n ".format(prediction))


