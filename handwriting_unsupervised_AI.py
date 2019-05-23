import codecademylib3_seaborn
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

#From a total of 43 people, 30 contributed to the training set and different 13 to the test set.
digits = datasets.load_digits()
print(digits.data)
print(digits.target)

#plt.gray()
#plt.matshow(digits.images[100])
#plt.show()
#print(digits.target[100])

model = KMeans(n_clusters = 10, random_state = 42)
model.fit(digits.data)

fig = plt.figure(figsize=(8, 3))

fig.suptitle("Cluser Center Images", fontsize=14, fontweight='bold')

for i in range(10):

  # Initialize subplots in a grid of 2X5, at i+1th position
  ax = fig.add_subplot(2, 5, 1 + i)

  # Display images
  ax.imshow(model.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)

plt.show()


new_samples = np.array([
[0.00,0.00,0.00,3.02,7.53,4.33,2.73,0.00,0.00,0.00,0.97,7.38,5.77,5.92,7.07,0.29,0.00,0.23,5.93,7.39,1.20,3.40,7.62,2.73,0.00,4.77,7.62,2.65,0.75,2.11,7.62,4.27,2.64,7.61,7.44,7.30,7.60,7.60,7.62,5.56,2.57,6.09,5.78,4.47,2.74,2.12,4.95,6.78,0.00,0.00,0.00,0.00,0.00,0.00,3.42,7.62,0.00,0.00,0.00,0.00,0.00,0.00,2.19,7.62],
[0.00,0.00,1.43,2.28,2.28,0.45,0.00,0.00,1.13,5.91,7.60,7.61,7.62,6.01,0.22,0.00,3.72,7.61,3.55,1.06,2.73,7.62,2.43,0.00,0.45,1.82,0.00,0.00,0.07,7.62,3.05,0.00,0.75,4.25,5.33,4.63,2.04,7.62,2.97,0.00,4.48,7.52,5.85,7.45,7.62,6.93,0.67,0.00,3.94,7.53,4.71,7.54,7.53,7.53,4.85,1.06,0.60,6.39,6.85,4.93,0.90,4.53,7.54,3.35],
[0.00,0.00,0.60,2.12,2.28,2.96,0.22,0.00,0.00,3.33,7.54,7.61,7.61,7.61,1.29,0.00,0.60,7.45,5.54,1.88,3.33,7.62,0.76,0.00,1.52,7.62,2.20,0.00,6.14,6.92,0.15,0.00,0.60,7.07,7.30,6.08,7.62,3.63,0.00,0.00,0.00,1.21,5.47,7.62,7.61,6.99,0.37,0.00,0.00,0.07,6.38,6.77,3.41,7.62,0.76,0.00,0.00,1.51,7.61,6.75,6.63,7.54,0.53,0.00],
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,2.88,5.34,5.33,5.33,5.10,0.30,0.00,1.51,7.62,6.16,5.33,6.53,7.62,0.75,0.00,2.28,7.62,4.64,4.26,7.14,7.62,0.76,0.00,0.59,5.77,6.86,6.85,6.39,7.62,0.76,0.00,0.00,0.00,0.00,0.00,2.28,7.62,0.76,0.00,0.00,0.00,0.00,0.00,2.28,7.62,0.76,0.00,0.00,0.00,0.00,0.00,2.28,7.62,0.76,0.00]
])
new_labels = model.predict(new_samples)

print(new_labels)

#map out each of the labels with the digits we think it represents
for i in range(len(new_labels)):
  if new_labels[i] == 0:
    print(0, end='')
  elif new_labels[i] == 1:
    print(9, end='')
  elif new_labels[i] == 2:
    print(2, end='')
  elif new_labels[i] == 3:
    print(1, end='')
  elif new_labels[i] == 4:
    print(6, end='')
  elif new_labels[i] == 5:
    print(8, end='')
  elif new_labels[i] == 6:
    print(4, end='')
  elif new_labels[i] == 7:
    print(5, end='')
  elif new_labels[i] == 8:
    print(7, end='')
  elif new_labels[i] == 9:
    print(3, end='')