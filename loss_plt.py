import numpy as np
import matplotlib.pyplot as plt # 画图库

plt.plot(w_list, mse_list)
plt.ylabel('Loss')
plt.xlabel('w')
plt.show()