import torch
import streamlit as st

st.header('Gaussian Elimination')
#ask the user for the matrix size 
st.write("please enter matrix size")
rows = st.number_input("please number of rows", value=2)
columns = st.number_input("Please enter number of columns", value=3)

#create a matrix that is number of rows by number of columns where user can input values
matrix = torch.zeros(rows,columns)
for i in range(rows):
    for j in range(columns):
        matrix[i,j] = st.number_input(f"please enter value for row {i+1} and column {j+1}",value= 0.0, step=1.0)
st.table(matrix)
calculate = st.button("calculate")
if calculate:
    st.write("Augmented matrix:")
    st.write(matrix)
    # Define the augmented matrix
    aug = matrix
    #aug = torch.FloatTensor([[10,3,5, 13], [2,9,10, 1], [12,19,10, 19]])
    
    # Get the size of the matrix
    size = aug.size()
    width = size[1]
    length = size[0]

    for k in range(length):
        # Find the row with the maximum value in column k
        max_index = torch.argmax(torch.abs(aug[k,k])) + k
        
        # Swap the current row with the row containing the maximum value
        if k != max_index:
            aug[[k, max_index]] = aug[[max_index, k]]
        
        # Check if the pivot is 0
        if aug[k, k] == 0:
            st.warning("Pivot is 0, cannot proceed with Gaussian elimination.")
            break
        
        # Divide the current row by the pivot
        aug[k] = aug[k] / aug[k, k]
        
        # Subtract the current row from all the other rows to get zeros in the lower triangle
        for i in range(length):
            if i == k:
                continue
            factor = aug[i, k]
            aug[i] -= factor * aug[k]

    st.write("Result:")
    st.write(aug)
