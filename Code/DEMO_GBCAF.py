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