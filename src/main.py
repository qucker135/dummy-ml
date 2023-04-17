import numpy as np
import activations as ac

sa = ac.SigmoidActivation()
print(sa.compute(np.array([0.0, 0.1, 1.0])))
