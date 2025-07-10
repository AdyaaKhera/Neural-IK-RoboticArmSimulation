#This file contains the fk code for a 2D robot to evalute the ik model. It also contains a matplotlib simulation
from math import radians, cos, sin
import matplotlib.pyplot as plt

link1, link2 = 100, 100

def fdotk(theta1, theta2): #theta1 and theta2 in degrees and we will be using fixed lengths for the links
    theta1 = radians(theta1)
    theta2 = radians(theta2)
    #final coordinates of the end-effector
    x = link1*cos(theta1) + link2*cos(theta1+theta2)
    y = link1*sin(theta1) + link2*sin(theta1+theta2)
    return x,y

def plot_arm(theta1, theta2):
    theta1 = radians(theta1)
    theta2 = radians(theta2)
    #elbow position
    x1 = link1*cos(theta1)
    y1 = link1*sin(theta1)
    #end-effector position calculated to double check the result from fdotk
    x2 = x1 + link2*cos(theta2+theta1)
    y2 = y1 + link2*sin(theta2+theta1)
    #plotting the arm
    x_val = [0, x1, x2]
    y_val = [0, y1, y2]
    plt.plot(x_val, y_val, color = "blue", linewidth = 3)
    plt.scatter(x_val, y_val, color = "red") #marking the joints
    plt.axis("equal")
    plt.grid(True)
    plt.title(f"2D Robot Arm: θ₁ = {round(theta1*180/3.14)}°, θ₂ = {round(theta2*180/3.14)}°")
    plt.show()