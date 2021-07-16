import numpy as np
import itertools

def is_invertible(A):
    A = A.copy()
    m = A.shape[0]
    for i in range(m):
        p = np.where(A[i:,i])[0]
        if len(p) == 0:
            return False
        rr = i + p[0]
        row = A[rr,i:].copy()
        A[rr,i:] = A[i,i:]
        A[i,i:] = row
        A[i+1:,i:] ^= A[i+1:,i,None]*row[None,:]
    return True

def get_invertible_matrix(m):
    while True:
        A = np.random.randint(0,2,[m,m])
        if is_invertible(A):
            return A
        
def bin_matrix_to_longs(A):
    _, N = A.shape
    return np.sum(2**np.arange(N)*A, axis=1)

def random_LTA(m):
    b = np.random.randint(0,2,m)
    A = np.eye(m, dtype=int)
    for i in range(1, m):
        for j in range(i):
            A[i,j] = np.random.randint(0,2)
    return A, b        
        
def random_UTA(m):
    A, b = random_LTA(m)
    return A.T, b       

def random_GA(m):
    A = get_invertible_matrix(m)
    b = np.random.randint(0,2,m)
    return A, b

def random_PI(m):
    A = perm_matrix(np.random.permutation(m))
    b = np.zeros(m, dtype=np.uint8)
    return A, b

def get_perm(A, b):
    # affine map to permutation
    m = len(b)
    z = np.array([list(i) for i in itertools.product([0, 1], repeat=m)])[:,::-1]
    z_prime = (A@z.T + np.tile(np.expand_dims(b,1),(1,2**m)))%2
    pi = np.array(bin_matrix_to_longs(z_prime.T),dtype=int)
    return pi

