def euler_method(f, t_range, initial_point, num_iterations):
    t0, y0 = initial_point
    t_start, t_end = t_range
    h = (t_end - t_start) / num_iterations

    y = y0
    for _ in range(num_iterations):
        t = t0
        y = y + h * f(t, y)
        t0 += h

    return y

def func(t, y):
    return t - y**2

t_range = (0, 2)
initial_point = (0, 1)
num_iterations = 10

final_y = euler_method(func, t_range, initial_point, num_iterations)
print(f"{final_y:.16f}")

def runge_kutta_method(f, t_range, initial_point, num_iterations):
    t0, y0 = initial_point
    t_start, t_end = t_range
    h = (t_end - t_start) / num_iterations

    t = t0
    y = y0
    for _ in range(num_iterations):
        k1 = h * f(t, y)
        k2 = h * f(t + h/2, y + k1/2)
        k3 = h * f(t + h/2, y + k2/2)
        k4 = h * f(t + h, y + k3)
        
        y += (k1 + 2*k2 + 2*k3 + k4) / 6
        t += h

    return y

def func(t, y):
    return t - y**2

t_range = (0, 2)
initial_point = (0, 1)
num_iterations = 10

final_y = runge_kutta_method(func, t_range, initial_point, num_iterations)
print(f"\n{final_y:.16f}")

def gaussian_elimination(A):
    n = len(A)
    for i in range(n):
        # Partial pivoting
        max_index = max(range(i, n), key=lambda x: abs(A[x][i]))
        A[i], A[max_index] = A[max_index], A[i]
        
        for j in range(i+1, n):
            factor = A[j][i] / A[i][i]
            for k in range(i, n+1):
                A[j][k] -= factor * A[i][k]
    return A

def backward_substitution(U, b):
    n = len(U)
    x = [0] * n
    for i in range(n-1, -1, -1):
        x[i] = int((b[i] - sum(U[i][j] * x[j] for j in range(i+1, n))) / U[i][i])
    return x

# Define the augmented matrix
A = [[2, -1, 1, 6],
     [1, 3, 1, 0],
     [-1, 5, 4, -3]]

# Perform Gaussian elimination
U = gaussian_elimination(A)

# Extract the augmented part (b)
b = [row[-1] for row in U]

# Extract the upper triangular part
U = [row[:-1] for row in U]

# Perform backward substitution
solution = backward_substitution(U, b)

print("\n[", end="")
print(*solution, sep=" ", end="")
print("]")

def lu_factorization(matrix):
    n = len(matrix)
    L = [[0] * n for _ in range(n)]
    U = [[0] * n for _ in range(n)]

    for j in range(n):
        L[j][j] = 1

        for i in range(j+1):
            s1 = sum(U[k][j] * L[i][k] for k in range(i))
            U[i][j] = matrix[i][j] - s1

        for i in range(j, n):
            s2 = sum(U[k][j] * L[i][k] for k in range(j))
            L[i][j] = (matrix[i][j] - s2) // U[j][j]

    return L, U

# Define the matrix
matrix = [
    [1, 1, 0, 3],
    [2, 1, -1, 1],
    [3, -1, -1, 2],
    [-1, 2, 3, -1]
]

# Perform LU factorization
L, U = lu_factorization(matrix)

# Print out the L matrix

for row in L:
    print("[[", end="")
    print(*row, sep=", ", end="")
    print("]]")
print("\n")
# Print out the U matrix
for row in U:
    print("[[", end="")
    print(*row, sep=", ", end="")
    print("]]")

def is_diagonally_dominant(matrix):
    n = len(matrix)
    for i in range(n):
        diagonal_value = abs(matrix[i][i])
        row_sum = sum(abs(matrix[i][j]) for j in range(n) if j != i)
        if diagonal_value <= row_sum:
            return False
    return True

# Define the matrix
matrix = [
    [9, 0, 5, 2, 1],
    [3, 9, 1, 2, 1],
    [0, 1, 7, 2, 3],
    [4, 2, 3, 12, 2],
    [3, 2, 4, 0, 8]
]

# Check if the matrix is diagonally dominant
result = is_diagonally_dominant(matrix)
print("\n")
print(result)

def determinant(matrix):
    n = len(matrix)
    if n == 1:
        return matrix[0][0]
    elif n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    else:
        det = 0
        for j in range(n):
            submatrix = [row[:j] + row[j+1:] for row in matrix[1:]]
            det += matrix[0][j] * (-1) ** j * determinant(submatrix)
        return det

def is_positive_definite(matrix):
    n = len(matrix)
    for k in range(1, n + 1):
        submatrix = [[matrix[i][j] for j in range(k)] for i in range(k)]
        det = determinant(submatrix)
        if det <= 0:
            return False
    return True

# Define the matrix
matrix = [
    [2, 2, 1],
    [2, 3, 0],
    [1, 0, 2]
]

# Check if the matrix is positive definite
result = is_positive_definite(matrix)
print("\n")
print(result)
