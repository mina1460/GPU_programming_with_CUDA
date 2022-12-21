import numpy as np 
import scipy.sparse as sp


matrix = [[0,1,0,0,0,0],[0,0,0,1,0,0],[0,1,0,0,0,0],[1,0,1,0,1,0],[0,1,0,1,0,1],[1,0,0,0,0,0]]

def convert_to_jds(matrix):
    nnz_per_rows = [ np.count_nonzero(row) for row in matrix]

    dict_row_count_index = {}
    for i in range(len(nnz_per_rows)):
       dict_row_count_index[i] = nnz_per_rows[i]
    print(dict_row_count_index)

    sorted_dict_row_count_index = sorted(dict_row_count_index.items(), key=lambda x: x[1], reverse=True)
    print(sorted_dict_row_count_index)
    for i in range(len(sorted_dict_row_count_index)):
        print("Row index:", sorted_dict_row_count_index[i][0], "Number of non-zero elements:", sorted_dict_row_count_index[i][1])


    data = []
    row_index = []
    col_index = []
    section_index = [0]
    
    dict_counts = {}
    for i in nnz_per_rows:
        if i in dict_counts:
            dict_counts[i] += 1
        else:
            dict_counts[i] = 1

    stopped_index = 0
    dup_values = []
    for key, value in dict_counts.items():
        dup_values.append(value)
    dup_values = sorted(dup_values, reverse=True)
    x = int(input("Number of sections: "))
    if x < 0:
        unique_non_zero_count = len(set(nnz_per_rows))
        num = unique_non_zero_count
        print("AUTO\n")
    else:
        num = x
    for i in range(num):
        if x < 0:
            num = dup_values[0]
            dup_values.pop(0)
            print(f"Section {i+1} has {num} non-zero elements")

        else:
            num = int(input(f"enter the number of rows in section {i+1}: " ))
        non_zero_count_in_section = 0
        for i in range(stopped_index, stopped_index + num):
            print("\nRow index:", sorted_dict_row_count_index[i][0], "Number of non-zero elements:", sorted_dict_row_count_index[i][1])
            row_index.append(sorted_dict_row_count_index[i][0])
            for index, element in enumerate(matrix[sorted_dict_row_count_index[i][0]]):
                if element != 0:
                    print(f"{index}:[{element}]", end=" ")
                    data.append(element)
                    col_index.append(index)
                    non_zero_count_in_section += 1
            print()
        
        section_index.append(section_index[-1] + non_zero_count_in_section)
        stopped_index += num

    print("Data:", data)
    print("Row index:", row_index)
    print("Column index:", col_index)
    print("Section index:", section_index)

def convert_to_csr(matrix):
    print("Converting to CSR format")
    csr_matrix = sp.csr_matrix(matrix)
    print("Data: ", csr_matrix.data)
    print("Col indices: ", csr_matrix.indices)
    print("Row pointers: ", csr_matrix.indptr)
    print("\n")

def convert_to_coo(matrix):
    print("Converting to COO format")
    coo_matrix = sp.coo_matrix(matrix)
    print("Data: ", coo_matrix.data)
    print("Row indices: ", coo_matrix.row)
    print("Col indices: ", coo_matrix.col)
    print("\n")

def convert_to_ELL(matrix):
    nnz_mat = [[] for i in range(len(matrix))]
    col_mat = [[] for i in range(len(matrix))]
    for r_idx, row in enumerate(matrix):
        for col_idx, element in enumerate(row):
            if element != 0:
                nnz_mat[r_idx].append(element)
                col_mat[r_idx].append(col_idx)
    print("Converting to ELL format")
    max_nnz = max([len(row) for row in nnz_mat])

    print("Max number of non-zero elements in a row:", max_nnz)
    for index, row in enumerate(nnz_mat):
        while len(row) < max_nnz:
            row.append('*')
            col_mat[index].append('*')

    transpose = list(map(list, zip(*nnz_mat)))
    transpose_col = list(map(list, zip(*col_mat)))
    print("Data: ", transpose)
    print("Column indices: ", transpose_col)
    print("\n")

#convert_to_ELL(matrix)
convert_to_csr(matrix)
#convert_to_coo(matrix)
#convert_to_jds(matrix)

