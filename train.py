"""
Training file

Best Performance : 50/100/0.7 -> 96.08
"""

from neural_network import *
import os
import matplotlib.pyplot as plt
import pickle

#Auxiliaries functions

def get(filename):
    """Return the object in the file filename.data"""
    with open(filename,'rb') as f: 
        get_record = pickle.Unpickler(f) 
        return get_record.load()


#Constants
"""Modify easily the settings of the network, the behavior of the programme and the locations of the objects"""
create = False #create a new_network
name = 'ENS2'
nb_cycles = 10
batch_size = 100 #batch size
learning_rate = 0.70
structure = [784,20,20,10] #numbers of layers and neurons
root = os.path.dirname(__file__)
path_train_images = os.path.join(root,"data","train_images.data")
path_train_labels = os.path.join(root,"data", "train_labels.data")
path_test_images = os.path.join(root,"data", "test_images.data")
path_test_labels = os.path.join(root,"data", "test_labels.data")
train = False
test = True
save = True
display_weights_and_biases = False
training_features = get(path_train_images)
training_labels = get(path_train_labels)
testing_features = get(path_test_images)
testing_labels = get(path_test_labels)
training_set_size = len(training_features)
testing_set_size = len(testing_features)
n,p = training_labels[0].shape
assert p==1
assert len(training_labels) == training_set_size
assert len(testing_labels) == testing_set_size

if create :
    net = NeuralNetwork(structure)
else :
    net = get(os.path.join(root,"networks","{}.data".format(name)))

#Training
if train:
    print("Training...")
    for i in range(nb_cycles):
        net.train(training_features,training_labels,batch_size,learning_rate = learning_rate, display_progress = False)
        print("End of the {}-th loop of {}".format(i+1,nb_cycles))
    print("Training completed !")

    
#Testing
if test :
    print("""Testing...""")
    net.test(testing_features,testing_labels)
    print("Tesing completed !")
    
#Sauvegarde
"""Sauvegarde le reseau de neurones"""
if save :
    net.save(os.path.join(root,"networks","{}.data".format(name)))


#Poids et Biais
if display_weights_and_biases :
    print("Matrices des poids : ",net.weights, "\n")
    print("Matrice des biais : ",net.biases, " \n")

