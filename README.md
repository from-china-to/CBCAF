# GBCAF
Jinfan Chen, Xingyue Zhao, Junrui Li, **Chang Liu*** [A Granular Ball Connectivity-Based Anomaly Factor]

## Abstract
Anomaly detection technology has important application value when dealing with vast, multi-source and complex raw data, medical data is a particularly typical scenario among it, but how to identify anomalous data quickly and accurately remains a core challenge. Existing anomaly detection methods, whether based on deep learning or local density, suffer from dual limitations: high susceptibility to noise interference and substantial computational overhead. To address these issues, a granular-ball computing-based anomaly detection model is presented in this paper, named Granular Ball Connectivity-Based Anomaly Factor (GBCAF). The algorithm utilizes granular-ball to characterise data; by combining the granular-ball model with the Connectivity-based Anomaly Factor (CAF), it conducts a granular-ball density anomaly factor, in order to perform anomaly detection. In addition, this method is experimentally compared with ten other algorithms, and the results demonstrate it has good adaptability and effectiveness in medical data anomaly detection. The code is publicly available online at [https://github.com/from-china-to/GBCAF](https://github.com/from-china-to/GBCAF).

## Usage

You can run DEMO_GBCAF.py:
```python
import numpy as np
from scipy.io import loadmat
from sklearn.neighbors import NearestNeighbors
from GBCAF import generate_granular_balls, normalize_features, all_acd


def get_gbcaf_scores(file_path="Example.mat", k=2):
    mat_data = loadmat(file_path)
    X = normalize_features(mat_data['table_data'])

    balls = generate_granular_balls(X)
    centers = np.array([ball.center for ball in balls])
    members = [ball.indices for ball in balls]

    actual_k = min(k + 1, len(centers))
    nn = NearestNeighbors(n_neighbors=actual_k)
    nn.fit(centers)
    _, neighbor_indices = nn.kneighbors(centers)

    true_neighbor_ids = neighbor_indices[:, 1: actual_k]

    acd = all_acd(centers, true_neighbor_ids)
    acd = np.round(acd, 4)

    denominator = acd[true_neighbor_ids].mean(axis=1)
    denominator = np.round(denominator, 4)

    ball_scores = np.divide(acd, denominator, out=np.ones_like(acd), where=denominator > 0)

    scores = np.empty(len(X), dtype=float)
    for value, indices in zip(ball_scores, members):
        scores[indices] = value

    formatted_scores = [float(f"{s:.4f}") for s in scores]
    print(formatted_scores)


if __name__ == "__main__":
    get_gbcaf_scores("Example.mat", k=2)
```

You can get outputs as follows:
```
[0.9474, 0.9474, 0.9474, 1.111]
```

## Citation
If you find GBCAF useful in your research, please consider citing:
```
@article{CHEN,
title = {A Granular Ball Connectivity-Based Anomaly Factor},
author = {Jinfan Chen and Xingyue Zhao and Junrui Li and Chang Liu},

## Contact
If you have any question, please contact liuchangai@alu.scu.edu.cn.
