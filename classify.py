#!./py2/bin/python
# -*- coding: utf-8 -*-
import time
import glob
import numpy as np
from PIL import Image
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

dirname = './data/*/*.png'
img_height = 30
img_width = 30
hidden_neurons = img_height*img_width*10

# get a list of image filenames
image_filenames = glob.glob(dirname)

# class maps for lookups
classes = [2, 3, 5, 7]
classes_index_map = {2: 0, 3: 1, 5: 2, 7: 3}
inverse_classes_index_map = {0: 3, 1: 5, 2: 7, 3: 2}

# prepare training data
input_data = []

training_data_count = 80
testing_data_count = 20
total_count = training_data_count + testing_data_count

# organize I/O data from images and labels
for fn in image_filenames:
    label_number = int(fn.split('/')[-2])
    input_data.append({'label': label_number, 'image': Image.open(fn).convert('RGB'), 'index': classes_index_map[label_number]})

# generate training image and label arrays from training dataset
images_array = np.array([np.array(x['image'])[:,:,0] for x in input_data])
image_labels = np.array([x['label'] for x in input_data])

# close open filehandles; no longer needed
[_['image'].close() for _ in input_data]

# reshape input data to flatten each image in the array
reshaped_input_data = images_array.reshape(total_count, img_height*img_width)

# create a matrix of desired outputs based on output labels array
desired_output_matrix = np.zeros((total_count, len(classes)), dtype=int)
for i in range(total_count):
    desired_output_matrix[i][classes_index_map[image_labels[i]] - 1] = 1

X = reshaped_input_data
y = desired_output_matrix

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# normalize the data
sc = StandardScaler()
sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# train
print("\nStarted training...")
training_start_time = time.time()
mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, max_iter=1000, verbose=False, hidden_layer_sizes=(hidden_neurons,))
mlp.fit(X_train_std, y_train)
training_end_time = time.time()
total_training_time = str(int(training_end_time - training_start_time))

# predict
prediction_start_time = time.time() * 1000
y_pred = mlp.predict(X_test_std)
y_pred_prob = mlp.predict_proba(X_test_std)
prediction_end_time = time.time() * 1000
total_prediction_time = str(int(prediction_end_time - prediction_start_time))

print("...training and prediction completed\n")

print("\nProbability Matrix:\n")
print(y_pred_prob)
print("\nActual Output:\n")
print(y_test)
print("\nPredicted Output:\n")
print(y_pred)
# accuracy
accuracy = accuracy_score(y_test, y_pred)

# print training metadata
print("\nTraining Metadata:\n")
print("Image classes: {}".format(classes))
print("Image dimensions: {} x {}".format(img_height,img_width))
print("Number of hidden neurons".format(hidden_neurons))
print("Number of training images: {}".format(training_data_count))
print("Total time in training: {}s".format(total_training_time))
print("Number of testing images: {}".format(testing_data_count))
print("Total time in testing: {}ms".format(total_prediction_time))
print("Prediction accuracy: {}%".format(accuracy*100))

# Create report directory if does not exist
import os
if not os.path.exists('report'):
    os.makedirs('report')

# Create HTML table with accuracy report
html = """<html><table border="1" style="text-align:center;"><tr><th>#</th><th>Actual Image</th><th>Class Label</th><th>Prediction</th><th>Match?</th></tr>"""

for i, y in enumerate(y_pred):
    html += "<tr><td>{}</td>".format(i+1)
    pred_index = np.argmax(y)
    test_index = np.argmax(y_test[i])
    html += "<td><img src=\"{}.png\"></td>".format(i)
    html += "<td>{}</td>".format(inverse_classes_index_map[test_index])
    html += "<td>{}</td>".format(inverse_classes_index_map[pred_index] if y_pred[i].any() else "-")
    # print("This image was recognized as: {}".format(inverse_classes_index_map[pred_index]))
    img = Image.fromarray(X_test[i].reshape(img_height, img_width))
    img.save('./report/{}.png'.format(i))
    if (y_pred[i] == y_test[i]).all():
        # print("Images match!")
        html += "<td>{}</td>".format("👍")
    else:
        # print("Images do not match! Actual image: {}".format(inverse_classes_index_map[test_index]))
        html += "<td>{}</td>".format("👎")
html += "</table></html>"

with open('./report/report.html', 'w') as f:
    f.write(html)

print("\nAccuracy report created: report.html")

#import pdb; pdb.set_trace()

