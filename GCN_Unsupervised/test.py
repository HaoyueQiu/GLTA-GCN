import numpy as np
from sklearn import preprocessing
import torch
# a = np.asarray([[0.07716928,0.03395055,0.07363877,0.08779999,0.09124018,0.15031101,0.21393532,0.2563119,0.09196077,0.15753494,0.22839014,0.27403742,
#         0.0905839 ,0.14892179 ,0.17746575, 0.25223324, 0.09057538, 0.15295361,0.17971317 ,0.25437343 ,0.06341638, 0.30254194 ,0.32573557 ,0.3223823, 0.34228417]])

# b = preprocessing.normalize(a)
# print(b)
a = torch.rand((3,12))
print(a)
b = a.sum(axis=1).unsqueeze(-1)
c = a/b
print(c.sum(axis=1))
