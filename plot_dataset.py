#this file will plot the points geenrated in the dataset to get an idea of the reachable space of the data

import matplotlib.pyplot as plt
import csv

x_vals, y_vals = [], []

#going through the dataset
with open("data/dataset.csv", newline = "") as file:
    reader = csv.reader(file)
    next(reader) #to skip the header
    for row in reader:
        #extracting x and y values
        x_vals.append(float(row[0]))
        y_vals.append(float(row[1]))

plt.scatter(x_vals, y_vals, s = 1, color = "purple") #s is the size of each point
plt.axis("equal")
plt.title("Span of the 2D arm")
plt.grid(True)
plt.show()