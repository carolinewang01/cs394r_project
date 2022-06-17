import numpy as np
from matplotlib import pyplot as plt
import seaborn as sb
sb.set_theme()

taus = [0.5353, 0.8553, 0.8078, 0.0406, 0.4016, 0.9822, 0.1382, 0.9267, 0.9831,
         0.0285, 0.5707, 0.2034, 0.2130, 0.7031, 0.1640, 0.3427, 0.2987, 0.9669,
         0.6470, 0.6122, 0.1656, 0.7448, 0.0680, 0.6213, 0.2359, 0.9027, 0.4412,
         0.0645, 0.6283, 0.5442, 0.2878, 0.2242]

action1 = [ 0.0563,  0.4826,  0.3439, -0.1055,  0.0168, 10.0155, -0.0369,
           7.2873,  9.9512, -0.0178,  0.0344, -0.0510,  0.0102,  0.2382,
           0.0110, -0.0194,  0.0277,  8.8057,  0.1600,  0.0858, -0.0167,
           0.1378, -0.0529,  0.1365,  0.0139,  4.8766,  0.0796, -0.0164,
           0.2414,  0.0155, -0.0283,  0.0254]

action2 = [ 0.8302,  0.9442,  0.9549,  0.2947,  0.7591,  1.0768,  0.5789,
           0.9818,  1.0757,  0.2269,  0.8812,  0.6376,  0.6628,  0.9382,
           0.5853,  0.7801,  0.7589,  1.0408,  0.9090,  0.8733,  0.5620,
           0.9451,  0.3384,  0.8919,  0.6475,  0.9569,  0.8461,  0.3778,
           0.9005,  0.8265,  0.7392,  0.6675]


cdf_action1 = zip(taus, action1)
cdf_action1 = list(cdf_action1)
cdf1 = sorted(cdf_action1, key=lambda x:x[0])
cdf1 = np.array(cdf1)

cdf_action2 = zip(taus, action2)
cdf_action2 = list(cdf_action2)
cdf2 = sorted(cdf_action2, key=lambda x:x[0])
cdf2 = np.array(cdf2)


taus = sorted(taus)

plt.plot(taus, cdf1[:,1], label="action1: 90% rew=0 10% rew=9")
plt.plot(taus, cdf2[:,1], label="action2: 90% rew=1 10% rew=0")

plt.xlabel("taus")
plt.ylabel("returns")
plt.title("cdf of learned return distributions of actions")
plt.legend()
plt.show()

