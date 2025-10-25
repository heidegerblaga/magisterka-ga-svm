import random, numpy as np

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
