# Neural Inverse Kinematics for 2D Robotic Arm

A simulation of inverse kinematics using a neural network (MLP) to predict joint angles for a 2-link planar robot arm. This project includes a fully interactive Matplotlib animation where users can click on target positions to simulate the robot's response in real-time generating the required angles.

---

## Features

- **Neural Inverse Kinematics** — MLP predicts joint angles `(θ₁, θ₂)` for any target `(x, y)` within reach
- **Smooth Arm Animation** — Interpolates between arm positions for a realistic robotic feel
- **Interactive Plot** — Click anywhere to move the robot’s end-effector to that position
- **Constraint Handling** — Automatically filters unreachable targets or invalid poses
- **Model Trained from Scratch** — Custom dataset generated with physical constraints
- **Forward Kinematics Validation** — Ensures predictions result in geometrically correct poses for trained dataset

---

## Tech Stack

| Tool/Library        | Purpose                                  |
|---------------------|------------------------------------------|
| `PyTorch`           | Neural network for inverse kinematics    |
| `Matplotlib`        | Animation and plotting                   |
| `CSV`               | Dataset I/O                              |
| `Python 3.9+`       | Core language                            |

---

##Notes/Limitations

- Only supports planar 2-link arms with fixed link lengths
- Trained on elbow-up configuration only
- Limited span due to limited dataset
- Does not simulate dynamics, torque, or collisions
