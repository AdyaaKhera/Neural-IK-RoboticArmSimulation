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
| `NumPy`             | Numerical operations                     |
| `CSV`               | Dataset I/O                              |
| `Python 3.9+`       | Core language                            |

---

## Folder Structure

Neural-IK-RoboticArmSimulation/
│
├── data/
│ └── dataset.csv # (x, y) → (θ₁, θ₂) dataset
├── forward_kinematics.py # FK + animated plot_arm()
├── dataset.py # Dataset generator
├── plot_dataset.py # Visualizing span
├── inverse_model.py # Neural IK + simulation UI
├── media/ # Project media
└── README.md # Project documentation


---

##Notes/Limitations

- Only supports planar 2-link arms with fixed link lengths
- Trained on elbow-up configuration only
- Limited span due to limited dataset
- Does not simulate dynamics, torque, or collisions