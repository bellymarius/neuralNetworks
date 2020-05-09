#Réseau de Neurones

Ce projet est une implémentation des réseaux de neurones artificiels en Python.

#Description

neural_network.py contient la classe NeuralNetwork qui représente un réseau de neurones
Cette classe contient nottament les méthodes train, test et predict.

train.py est un script pour entraîner, tester puis sauvegarder un réseau de neurones.
Il est possible de faire varier plusieurs paramètres, comme le taux d'apprentissage (ligne 27),
le nombre de boucle sur le jeu d'entraînement (ligne 25) ou encore le nombre d'exemples à aggréger à chaque étape de la descente (ligne 26).

use.py est un script pour étiqueter des images.
Pour ajouter une image, insérez-la dans le répertoire "examples"
puis ajoutez le nom du fichier à la variable files (ligne 9)

overfitting.py est un script pour visualiser l'effet de surapprentissage des données.
Le taux d'apprentissage et le nombre de boucle y est beaucoup plus important (L'exécution est donc très longue !). Le résultat est une courbe commme celle présentée dans "overfitting.png"

sample.py permet de visualiser un échantillon du jeu d'entraînement

convert.py contient diverses fonctions pour transformer des données brutes en numpy array compatible avec mon implémentation. Certaines fonctions de ce fichier ne sont pas le miennes

#Date

Ce projet a été initié en Mai 2019 et subit régulièrement des modifications

#Amélioration en cours

Possibilité de choisir la fonction d'activation ReLU
Représentation visuelle du poids global de chaque pixel sur un neurone
A long terme, j'aimerais accomplir l'implémentation de réseaux de convolutions
puis de réseaux adversariaux.

