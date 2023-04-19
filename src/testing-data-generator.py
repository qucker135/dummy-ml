import numpy as np

def label(data: np.ndarray):
    values = []
    for sample in data:
        if sample[0] > 0.3 and sample[1] > 0.3:
            values.append(1.0)
        elif sample[1] > 0.5:
            values.append(3.0)
        elif sample[0] > 0.6:
            values.append(4.0)
        else:
            values.append(2.0)
    
    return np.insert(data.copy(), data.shape[1], values, axis=1)

unlabeled_data = np.random.uniform(0.0, 1.0, (1000,2))
labeled_data = label(unlabeled_data)
print(labeled_data)

np.savetxt("data.csv", labeled_data, delimiter=",")
