import numpy as np

data = np.array([])
counter = np.zeros(4)

for _ in range(100000):
    x = np.random.randint(1, 5)
    match x:
        case 1:
            data = np.append(data, [
                np.random.uniform(0.3, 1.0),
                np.random.uniform(0.3, 1.0),
                1
            ])
        case 2:
            y = np.random.randint(4)
            match y:
                case 1:
                    data = np.append(data, [
                        np.random.uniform(0.0, 0.2),
                        np.random.uniform(0.0, 0.3),
                        2
                    ])
                case 2:
                    data = np.append(data, [
                        np.random.uniform(0.2, 0.4),
                        np.random.uniform(0.0, 0.3),
                        2
                    ])
                case 3:
                    data = np.append(data, [
                        np.random.uniform(0.4, 0.6),
                        np.random.uniform(0.0, 0.3),
                        2
                    ])
                case 0:
                    data = np.append(data, [
                        np.random.uniform(0.0, 0.3),
                        np.random.uniform(0.3, 0.5),
                        2
                    ])
        case 3:
            data = np.append(data, [
                np.random.uniform(0.0, 0.3),
                np.random.uniform(0.5, 1.0),
                3
            ])
        case 4:
            data = np.append(data, [
                np.random.uniform(0.6, 1.0),
                np.random.uniform(0.0, 0.3),
                4
            ])
    counter[x - 1] += 1


data = data.reshape(-1, 3)
print(data)
print(counter)
np.savetxt("data.csv", data, delimiter=",")
