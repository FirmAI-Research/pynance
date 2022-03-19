import numpy as np
import matplotlib.pyplot as plt

def calc_payoff(s0, k, c, nshares, sT):
    y1= (sT-s0) * nshares
    y2 = np.where(sT > k,((k - sT) + c) * nshares, c * nshares)
    y3 = np.where(sT > k,((k - s0) + c) * nshares,((sT- s0) + c) * nshares )

    return [y1,y2,y3]