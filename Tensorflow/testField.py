import numpy as np

a = np.random.randn(2, 2, 2, 2)

b = a 

print(a)
print("--------------------------------")
print(a[0,0,:,0])

print(np.allclose(a,b))