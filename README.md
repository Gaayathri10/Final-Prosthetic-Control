# Adaptive Neural Control for Continuous Prosthetic Arm Movement

Control systems are ubiquitous in engineering, yet most commercial prosthetic devices still rely on discrete classification based control. While they are effective, these systems often produce unnatural, stepwise motion and require significant cognitive effort from the user.

This project begins with the premise that the missing element in advanced prosthetic control is a physiologically grounded, continuously adaptive mapping between surface EMG signals and limb kinematics. Rather than treating EMG as a classification problem (open/close, flex/extend), we treat it as a dynamic system identification and control problem.

The goal is to develop and evaluate a neural network–based framework that maps raw surface EMG signals to continuous wrist and finger trajectories, while incorporating reinforcement learning to improve controllability and embodiment over time.

---

## Problem Statement

Human neuromuscular systems do not come with datasheets. Muscle activation patterns:

- Drift over time (fatigue, electrode shift)
- Vary across users
- Exhibit nonlinear, time-dependent dynamics
- Are inherently noisy

Most existing prosthetic control systems rely on classification (e.g., LDA, SVM), which:
- Reduce rich signals to discrete commands
- Struggle with simultaneous multi-DOF movement
- Require frequent recalibration
- Increase cognitive burden

This project investigates whether continuous prediction models (RCNN/Transformer) can better capture the temporal structure of EMG and produce smoother, more natural motion.

---

## Project Structure

| File | Purpose |
|------|---------|
| `config.py` | Global parameters and hyperparameters |
| `data_loader.py` | Loads and labels raw EMG datasets |
| `preprocess.py` | Bandpass (20–450 Hz), notch filtering, windowing |
| `features.py` | Feature extraction for LDA / SVM models |
| `classical_models.py` | Baseline classifier implementation |
| `rcnn_model.py` | Neural network architecture definition |
| `train.py` | Neural model training pipeline |
| `evaluate.py` | Performance metrics and benchmarking |
| `adaptive.py` | Reinforcement learning / online adaptation |
| `main.py` | Executes full experiment workflow |

