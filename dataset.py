from forward_kinematics import fdotk
import random, csv
with open("data/dataset.csv", "w", newline = "") as file:
    writer = csv.writer(file)
    writer.writerow(["x", "y", "theta1", "theta2"]) #writing the header
    for _ in range(10000):
        #generating dataset with predefined restraints
        theta1 = random.uniform(-90, 90) #only moves left to right
        theta2 = random.uniform(0, 135) #only bends forward
        x, y = fdotk(theta1, theta2)
        writer.writerow([x, y, theta1, theta2])