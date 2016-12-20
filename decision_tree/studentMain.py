#!/usr/bin/python

""" lecture and example code for decision tree unit """

import sys
sys.path.append("../naive_bayes/")
from class_vis import prettyPicture, output_image
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
import classifyDT as dt
from sklearn.metrics import accuracy_score

terrain_train, terrain_labels_train, terrain_test, terrain_labels_test = makeTerrainData()



### the classify() function in classifyDT is where the magic
### happens--fill in this function in the file 'classifyDT.py'!
clf = dt.classify(terrain_train, terrain_labels_train)


pred = clf.predict(terrain_test)

acc = accuracy_score(pred, terrain_labels_test)


#### grader code, do not modify below this line

prettyPicture(clf, terrain_test, terrain_labels_test)
output_image("test.png", "png", open("test.png", "rb").read())
