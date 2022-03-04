import h5py
import os
import glob
import cv2
import numpy as np
import warnings
from matplotlib import pyplot
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from leafset import *

warnings.filterwarnings('ignore')

num_trees = 100
test_size = 0.10
seed      = 9
train_dir = "dataset/train"
test_dir  = "dataset/test"
save_dir = "savedImages/"
h5_data    = "output/data.h5"
h5_labels  = "output/labels.h5"
scoring    = "accuracy"
imageCount = 0

train_labels = os.listdir(train_dir)

train_labels.sort()

# create models
models = []
models.append(('LR', LogisticRegression(random_state=seed)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier(random_state=seed)))
models.append(('RF', RandomForestClassifier(n_estimators=num_trees, random_state=seed)))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(random_state=seed)))

results = []
names   = []

# import the feature vector and trained labels
h5f_data  = h5py.File(h5_data, 'r')
h5f_label = h5py.File(h5_labels, 'r')

global_features_string = h5f_data['dataset_1']
global_labels_string   = h5f_label['dataset_1']

global_features = np.array(global_features_string)
global_labels   = np.array(global_labels_string)

h5f_data.close()
h5f_label.close()

print("[STATUS] features shape: {}".format(global_features.shape))
print("[STATUS] labels shape: {}".format(global_labels.shape))

print("[STATUS] training started...")

# split the training and testing data
(trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(global_features),
                                                                                          np.array(global_labels),
                                                                                          test_size=test_size,
                                                                                          random_state=seed)

print("[STATUS] splitted train and test data...")
print("Train data  : {}".format(trainDataGlobal.shape))
print("Test data   : {}".format(testDataGlobal.shape))
print("Train labels: {}".format(trainLabelsGlobal.shape))
print("Test labels : {}".format(testLabelsGlobal.shape))

# 10-fold cross validation
for name, model in models:
    kfold = KFold(n_splits=10)
    cv_results = cross_val_score(model, trainDataGlobal, trainLabelsGlobal, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# boxplot algorithm comparison
fig = pyplot.figure()
fig.suptitle('Machine Learning algorithm comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

# Testing the model

# create the model - Random Forests
clf  = RandomForestClassifier(n_estimators=num_trees, random_state=seed)

clf.fit(trainDataGlobal, trainLabelsGlobal)

for file in glob.glob(test_dir + "/*.jpg"):
    image = cv2.imread(file)
    image = cv2.resize(image, fixed_size)

    fv_hu_moments = fd_hu_moments(image)
    fv_haralick   = fd_haralick(image)
    fv_histogram  = fd_histogram(image)

    global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaled_feature = scaler.fit_transform(global_feature.reshape(1, -1))

    prediction = clf.predict(rescaled_feature)[0]
    length = len(file)
    string = file[13:(length-4)]
    
    # show predicted label and the actual name on image
    cv2.putText(image, train_labels[prediction], (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (32,178,170), 3)
    if train_labels[prediction] in file:
        cv2.putText(image, "Correct!", (300,700), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,128,0), 3)
    else:
        cv2.putText(image, f"Incorrect! {string}", (300,700), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), 3)

    # # display the output image
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

    # save the mask 
    if(cv2.imwrite(save_dir + "saved" + str(imageCount) + ".png", image)):
        imageCount = imageCount + 1
