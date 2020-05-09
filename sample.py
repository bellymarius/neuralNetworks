import os
import numpy as np
import pickle
from PIL import Image
import matplotlib.pyplot as plt

display_feature = True
save_image = False
i = 213




def get(filename):
    """recupere l'objet sauve par la fonction save dans le fichier filename"""
    with open(filename,'rb') as f: 
        get_record = pickle.Unpickler(f) 
        return get_record.load()


root = os.path.dirname(__file__)
features_path = os.path.join(root,"data","train_images.data")

features = get(features_path)

sample = features[i]
sample = np.floor(sample * 255).astype(int)
#sample = sample * 255


sample = sample.reshape(28,28)

if display_feature :
	print("""Voici un élément de l'espace des features :\n""",sample)

plt.clf()
plt.imshow(sample,cmap = "gray")
plt.show()
