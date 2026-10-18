import numpy as np
from sklearn.metrics import f1_score


def old(y_true, probs):
    thresholds = np.linspace(0.05, 0.95, 19)
    best_t = 0.5
    best_f1 = -1.0
    for t in thresholds:
        pred = (probs >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, best_f1

cases = [
    (np.array([0,0,0,0,0,1,1,1]), np.array([0.52,0.51,0.50,0.49,0.48,0.60,0.70,0.80])),
    (np.array([0,0,0,0,0,1,1,1,1]), np.array([0.72,0.71,0.68,0.65,0.63,0.75,0.80,0.82,0.85])),
    (np.array([0,0,0,0,0,0,1,1,1,1]), np.array([0.62,0.61,0.60,0.59,0.58,0.57,0.90,0.88,0.86,0.84])),
    (np.array([0,0,0,0,0,1,1,1]), np.array([0.12,0.11,0.13,0.14,0.15,0.88,0.91,0.93])),
    (np.array([0,0,0,0,0,1,1,1]), np.array([0.42,0.41,0.40,0.39,0.38,0.56,0.72,0.81])),
    (np.array([0,0,0,0,1,1,1]), np.array([0.51,0.50,0.49,0.48,0.62,0.74,0.82])),
    (np.array([0,0,0,0,1,1,1]), np.array([0.35,0.34,0.33,0.32,0.60,0.75,0.90])),
]
for i, (y, p) in enumerate(cases,1):
    print(i, old(y,p))
