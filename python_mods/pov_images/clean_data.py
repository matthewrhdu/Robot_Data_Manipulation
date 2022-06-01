import numpy as np


def clean_data(data: np.ndarray):
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    x_upper = x <= 1
    x_lower = x >= -0.5
    y_upper = y <= 1
    y_lower = y >= -0.5

    mask = np.array([all([x_upper[i], x_lower[i], y_upper[i], y_lower[i]]) for i in range(len(x))])
    return np.column_stack((x[mask], y[mask], z[mask]))


for i in range(5):
    data = np.load(f"img_pov{i}_items2.npy")
    data = clean_data(data)
    data[:, 1] = data[:, 1] - 0.01
    data[:, 0] = data[:, 0] + 0.05
    np.save(f"img_pov{i}_filtered.npy", data)

