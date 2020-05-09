"""
convert contient des fonctions pour tranformer des donnees en vecteurs compatibles
"""

import numpy as np
import gzip
from PIL import Image
import pickle

"""principales fonctions pour convertir en vecteurs"""

def from_np_array_to_vect(t,N = 255):
    """renvoie un vecteur numpy de dimension (n,1)
    qui contient les mêmes éléments que le tableau numpy t de dimension (,n) ramenés sur le segment 0,1 """
    return normalise_vect(np.array([[x] for x in t]),N)

def from_matrix_to_vect(t,N=255):
    n,p = t.shape
    a = np.zeros((n*p,1))
    for i in range(n*p):
        j = i % n
        k = i // n
        a[i] = max(0,t[k][j])
    return normalise_vect(a,N)

def from_PNG_to_vect(filename,N = 255):
    im=Image.open(filename)
    t = np.array(im)
    n,p = t.shape
    a = np.zeros((n*p,1))
    for i in range(n*p):
        j = i % n
        k = i // n
        a[i] = t[k][j]
    return normalise_vect(a,N)


def from_int_to_vect(n):
    a = np.zeros((10,1))
    a[n][0] = 1
    return a



"""Gerer les images du Mnist"""

def from_images_mnist_to_vect(fname,IMG_DIM):
    np_array_result = []
    n_bytes_per_img = IMG_DIM*IMG_DIM

    with gzip.open(fname, 'rb') as f:
        bytes_ = f.read()
        data = bytes_[16:]

        if len(data) % n_bytes_per_img != 0:
            raise Exception('Something wrong with the file')

        np_array_result = np.frombuffer(data, dtype=np.uint8).reshape(
            len(bytes_)//n_bytes_per_img, n_bytes_per_img)
        
        return from_np_array_map_to_list(from_np_array_to_vect,np_array_result)
        


def from_labels_mnist_to_vect(fname):
    result = []
    with gzip.open(fname, 'rb') as f:
        bytes_ = f.read()
        data = bytes_[8:]
        result = np.frombuffer(data, dtype=np.uint8)
    return from_np_array_map_to_list(from_int_to_vect,result)



"""appliquer une fonction a une liste ou a un tableau numpy"""

def from_np_array_map_to_list(f,npa):
    """retourne une liste des images des éléments du tableau numpy de
    dimension (,n) par la fonction f
    affiche tous les 1000 éléments l'indice qu'elle traite"""
    l = [x for x in npa]
    list_map(f,l)
    return l




def list_map(f,l):
    """applique la fonction f a tous les éléments de la liste l
        affiche tous les 1000 éléments l'indice qu'elle traite"""
    n = len(l)
    for i in range(n):
        if i % 1000 == 999:
            print(i)
        t = l[i]
        l[i]  = f(t)



def get(filename):
    """recupere l'objet sauve par la fonction save dans le fichier filename"""
    with open(filename,'rb') as f: 
        get_record = pickle.Unpickler(f) 
        return get_record.load()


def normalise_vect(X,N=255):
    return X/255


def save(obj,filename):
        """sauve le reseau dans le fichier filename"""
        with open(filename,'wb') as f:
            record = pickle.Pickler(f)
            record.dump(obj)






