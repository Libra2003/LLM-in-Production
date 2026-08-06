import scipy
import numpy as np

matrix = np.array(
    [
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0],
    ]  

)
u, s, vt = scipy.sparse.linalg.svds(matrix, k=1)
print(u, s, vt)

svd_matrix = u * s * vt
print(svd_matrix)