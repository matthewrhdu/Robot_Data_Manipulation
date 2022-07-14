import numpy as np

data = np.load("finger_samples.npy")
upper, lower = [], []
for i in range(3):
    line = data[:, i]
    avg = np.average(line)
    std = np.std(line)

    upper.append(int(avg + 2 * std))
    lower.append(int(avg - 2 * std))


print(f"lower = np.array({lower})")
print(f"upper = np.array({upper})")
