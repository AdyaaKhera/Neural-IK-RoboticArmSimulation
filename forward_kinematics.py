#This file contains the fk code for a 2D robot to evalute the ik model. It also contains a matplotlib simulation

from math import radians, degrees, cos, sin
import matplotlib.pyplot as plt
import matplotlib.patches as patches

link1, link2 = 100, 100

def fdotk(theta1, theta2): #theta1 and theta2 in degrees and we will be using fixed lengths for the links
    theta1 = radians(theta1)
    theta2 = radians(theta2)
    #final coordinates of the end-effector
    x = link1*cos(theta1) + link2*cos(theta1+theta2)
    y = link1*sin(theta1) + link2*sin(theta1+theta2)
    return x,y

def plot_arm(theta1, theta2, ax, lines=None, joints=None, text=None, animate=False, last_angles=(0, 0)):

    #calculating coordinates using angles
    def get_positions(t1, t2):
        t1, t2 = radians(t1), radians(t2)
        x1 = link1 * cos(t1)
        y1 = link1 * sin(t1)
        x2 = x1 + link2 * cos(t1 + t2)
        y2 = y1 + link2 * sin(t1 + t2)
        return [0, x1, x2], [0, y1, y2], (x1, y1), t1, t2

    #initializing containers
    if lines is None: lines = []
    if joints is None: joints = []
    if text is None: text = []

    #transition
    if animate:
        t1_curr, t2_curr = last_angles
        steps = 20
        for step in range(1, steps + 1):
            interp_t1 = t1_curr + (theta1 - t1_curr) * step / steps
            interp_t2 = t2_curr + (theta2 - t2_curr) * step / steps
            x_vals, y_vals, elbow, t1, t2 = get_positions(interp_t1, interp_t2)

            if lines:
                lines[0].set_data(x_vals, y_vals)
            else:
                lines.append(ax.plot(x_vals, y_vals, color="blue", linewidth=3)[0])

            if joints:
                joints[0].set_offsets(list(zip(x_vals, y_vals)))
            else:
                joints.append(ax.scatter(x_vals, y_vals, color="red"))

            plt.pause(0.01)

    #final positions
    x_vals, y_vals, elbow, t1, t2 = get_positions(theta1, theta2)

    if lines:
        lines[0].set_data(x_vals, y_vals)
    else:
        lines.append(ax.plot(x_vals, y_vals, color="blue", linewidth=3)[0])

    if joints:
        joints[0].set_offsets(list(zip(x_vals, y_vals)))
    else:
        joints.append(ax.scatter(x_vals, y_vals, color="red"))

    #clearing previous text and angle arcs
    for t in text:
        t.remove()
    text.clear()

    #removing previous angle arcs if any
    [child.remove() for child in ax.patches[:]]

    #drawing θ₁ arc (at origin)
    arc1 = patches.Arc((0, 0), 30, 30, angle=0, theta1=0, theta2=theta1, color="purple", linewidth=1.5)
    ax.add_patch(arc1)
    text.append(ax.text(15, 5, f"θ₁ = {theta1:.2f}", color="purple", fontsize=12))

    #drawing θ₂ arc (at elbow joint)
    elbow_x, elbow_y = elbow
    base_angle = degrees(t1)
    arc2 = patches.Arc((elbow_x, elbow_y), 30, 30, angle=base_angle, theta1=0, theta2=theta2, color="purple", linewidth=1.5)
    ax.add_patch(arc2)
    text.append(ax.text(elbow_x + 15*cos(t1), elbow_y + 15*sin(t1), f"θ₂ = {theta2:.2f}", color="purple", fontsize=12))

    plt.draw()
