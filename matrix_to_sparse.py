from scipy import sparse
import numpy as np

def solve_quiz(matrix: np.matrix):
    print("Matrix:")
    print(matrix)
    
    print("_____________________________________________________________")
    
    print("CSR Sparse matrix: ")
    sparse_matrix = sparse.csr_matrix(matrix)
    
    print("the csr row start array: ")
    print(sparse_matrix.indptr)
    print("the csr column index array: ")
    print(sparse_matrix.indices)
    print("the csr data array: ")
    print(sparse_matrix.data)
    print("_____________________________________________________________")
    print("COO Sparse matrix: ")
    sparse_matrix = sparse.coo_matrix(matrix)
    
    print("the coo row index array: ")
    print(sparse_matrix.row)
    print("the coo column index array: ")
    print(sparse_matrix.col)
    print("the coo data array: ")
    print(sparse_matrix.data)
    print("_____________________________________________________________")
    
    print("CSC Sparse matrix: ")
    sparse_matrix = sparse.csc_matrix(matrix)

    print("the csc column start array: ")
    print(sparse_matrix.indptr)
    print("the csc row index array: ")
    print(sparse_matrix.indices)
    print("the csc data array: ")
    print(sparse_matrix.data)

    print("_____________________________________________________________")

    

if __name__ == "__main__":
    matrix = np.matrix([[12, 0, 8, 0 ,0], [0, 0, 0, 0, 0], [0, 4, 10, 0, 0],[3, 9, 0, 15, 9],[0, 0, 0, 7, 5]])
    solve_quiz(matrix)