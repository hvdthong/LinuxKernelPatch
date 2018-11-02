import numpy as np; np.random.seed(42)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# df1 = pd.DataFrame({"x": np.sort(np.random.rand(30)),
#                     "f": np.sort(np.random.rayleigh(size=30))})
# # df2 = pd.DataFrame({"x": np.sort(np.random.rand(30)),
# #                     "g": 500-0.1*np.sort(np.random.rayleigh(20,size=30))**2})
# df2 = pd.DataFrame({"x": np.sort(np.random.rand(30)),
#                     "f": np.sort(np.random.rayleigh(size=30))})
#
# fig, ax = plt.subplots()
# # ax2 = ax.twinx()
# sns.lineplot(x="x", y="f", data=df1, ax=ax, label="df1")
# sns.lineplot(x="x", y="f", data=df2, ax=ax, label="df2")
#
#
# ax.legend()
# plt.show()

fmri = sns.load_dataset("fmri")
ax = sns.lineplot(x="timepoint", y="signal", data=fmri)
print ax