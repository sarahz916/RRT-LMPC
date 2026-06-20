# RRT-LMPC

This project combines rapid-exploring random trees (RRT), spline-based path representation, and learning model predictive control (LMPC) for trajectory planning around obstacles.

Developed as a CS 159 Spring 2021 final project by Aaron Feldman and Sarah Zou.

## Problem

The goal is to move a simple dynamical system from a start state to a goal state while avoiding obstacles and respecting control limits. The project first generates a feasible geometric path, then uses demonstrations and LMPC to track and improve trajectories under the modeled dynamics.

## Methods

- RRT search for feasible paths through obstacle fields
- Cubic spline fitting for smooth path parameterization
- Demonstration trajectory generation
- Learning model predictive control with finite-horizon optimization
- Dynamics and linearization checks for controller validation

## Repository Structure

- `lmpcMain.py`: main experiment script connecting RRT, spline fitting, demonstrations, and LMPC
- `lmpc.py`: LMPC controller implementation
- `ftocpLMPC.py`: finite-time optimal control problem setup
- `verifyingDynamics.py`: dynamics verification utilities
- `verifyingLinearization.py`: linearization verification utilities
- `make_demos/`: RRT, environment, spline, and demonstration-generation helpers
- `path.npy`, `xDemo.npy`, `uDemo.npy`, `spline.pkl`: saved path and demonstration artifacts

## How to Run

Install the Python dependencies:

```bash
pip install numpy scipy matplotlib cvxpy
```

Run the main experiment:

```bash
python lmpcMain.py
```

## Notes

This is a research/course project rather than a polished package. The code is most useful as evidence of control, planning, optimization, and Python modeling work.
