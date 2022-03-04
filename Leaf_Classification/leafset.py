from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import mahotas
import cv2
import os
import h5py

dataset_dir = "dataset"
train_dir = "dataset/train"
test_dir = "dataset/test"

if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

class_number = 17

class_names = ["acer_ginnala", "acer_griseum","amelanchier_laevis","betula_alleghaniensis",
            "carya_cordiformis","carya_glabra","cercis_canadensis","chionanthus_retusus",
            "cryptomeria_japonica","fraxinus_nigra","fraxinus_pennsylvanica","gymnocladus_dioicus",
            "malus_baccata","malus_coronaria","platanus_acerifolia","platanus_occidentalis","quercus_shumardii"]

fixed_size       = tuple((800, 800))
h5_data          = 'output/data.h5'
h5_labels        = 'output/labels.h5'
bins             = 8    

# Hu Moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

# Haralick Texture
def fd_haralick(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick

# Color Histogram
def fd_histogram(image, mask=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# get the training labels
train_labels = os.listdir(train_dir)

# sort the training labels
train_labels.sort()
print(train_labels)

global_features = []
labels          = []

# loop over the training data sub-folders
for training_name in train_labels:
    dir = os.path.join(train_dir, training_name)
    current_label = training_name

    for fileName in os.listdir(dir):
        f = os.path.join(dir, fileName)
        if os.path.isfile(f):
            image = cv2.imread(f)
            image = cv2.resize(image, fixed_size)
            
            # Global feature extraction
            fv_hu_moments = fd_hu_moments(image)
            fv_haralick   = fd_haralick(image)
            fv_histogram  = fd_histogram(image)

            global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

            # append the labels and feature vectors
            labels.append(current_label)
            global_features.append(global_feature)

    print("[STATUS] processed folder: {}".format(current_label))

print("[STATUS] completed Global Feature Extraction...")

# get the overall feature vector size
print("[STATUS] feature vector size {}".format(np.array(global_features).shape))

# get the overall training label size
print("[STATUS] training Labels {}".format(np.array(labels).shape))

# encode the target labels
targetNames = np.unique(labels)
le          = LabelEncoder()
target      = le.fit_transform(labels)
print("[STATUS] training labels encoded...")

# scale features
scaler            = MinMaxScaler(feature_range=(0, 1))
rescaled_features = scaler.fit_transform(global_features)
print("[STATUS] feature vector normalized...")

print("[STATUS] target labels: {}".format(target))
print("[STATUS] target labels shape: {}".format(target.shape))

# save the vectors using HDF5
h5f_data = h5py.File(h5_data, 'w')
h5f_data.create_dataset('dataset_1', data=np.array(rescaled_features))

h5f_label = h5py.File(h5_labels, 'w')
h5f_label.create_dataset('dataset_1', data=np.array(target))

h5f_data.close()
h5f_label.close()

print("[STATUS] end of training..")