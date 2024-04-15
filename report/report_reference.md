---
title: 數值方法 Homework 2 Report
author: 馬楷翔 110550074
date: \today{}
CJKmainfont: "Microsoft YaHei"
---

## Question 1

Apply gaussian elimination with partial pivoting to solve the augmented matrix, code:

```python
import numpy as np


def gaussian_elimination_with_partial_pivoting(A):
    n = A.shape[0]
    for i in range(n):
        # 在 i 列中找到絕對值對大的行
        max_row = np.argmax(np.abs(A[i:, i])) + i
        # 將 pivot 換到當前行
        A[[i, max_row]] = A[[max_row, i]]

        # 對下方的行進行消去
        for j in range(i + 1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] -= factor * A[i, i:]

    # 反向替代
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (A[i, -1] - np.dot(A[i, i + 1 : n], x[i + 1 : n])) / A[i, i]

    return x


if __name__ == "__main__":
    aug_mat = np.array(
        [
            [3, 1, -4, 7],
            [-2, 3, 1, -5],
            [2, 0, 5, 10],
        ],
        dtype=float,
    )
    print(gaussian_elimination_with_partial_pivoting(aug_mat))

```

And the results are:

`[3.20987654 0.2345679  0.71604938]`

Which is the same as the calculation I did by hand, with no row interchanges needed (every pivot is the largest element in its column).

## Question 2

### Initialization: Find the result at (6,2)

The code for finding intersection:

```python
def eq1(x):
    return (104 - 0.1 * x) / 51.7


def eq2(x):
    return (5.1 * x - 16) / 7.3


if __name__ == "__main__":
    # 生成 x 值
    x_values = np.linspace(0, 10, 400)

    # 計算對應 y 值
    y1_values = eq1(x_values)
    y2_values = eq2(x_values)

    # 畫圖
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y1_values, label='0.1x + 51.7y = 104')
    plt.plot(x_values, y2_values, label='5.1x - 7.3y = 16', linestyle='--')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Graph of the System of Equations')
    plt.legend()
    plt.grid(True)

    # 標記交點
    plt.plot(6, 2, 'ro') # 預期交點
    plt.text(6, 2, '(6,2)', verticalalignment='bottom')
    
    plt.show()
```

Which yields the following graph:

![Graph of the System of Equations](./assets/Figure_1.png)

### (a) Solve using 3 significant digits of precision and row interchanges

The code for solving the system of equations (Here I use `np.float16` since it's accurate enough for 3 significant digits):

```python
import numpy as np


def gaussian_elimination(A, b):
    n = len(b)
    Ab = np.hstack([A, b.reshape(-1, 1)])  # 建立增廣矩陣

    for i in range(n):
        # 對下方的行進行消去
        for j in range(i + 1, n):
            factor = Ab[j, i] / Ab[i, i]
            Ab[j, i:] -= factor * Ab[i, i:]

    # 反向替代
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i + 1 : n], x[i + 1 : n])) / Ab[i, i]

    return x


if __name__ == "__main__":
    A = np.array([[0.1, 51.7], [5.1, -7.3]], dtype=np.float16)
    b = np.array([104, 16], dtype=np.float16)

    sol = gaussian_elimination(A, b)

    # 格式化輸出為三位有效數字
    sol_formatted = np.round(sol, 3)

    print(sol_formatted)

```

The result is `[6.252 2.   ]`, which is a bit off compared to the expected result.


### (b) repeat (a) but use partial pivoting

The code for solving the system of equations with partial pivoting:

```python
import numpy as np


def gaussian_elimination(A, b):
    n = len(b)
    Ab = np.hstack([A, b.reshape(-1, 1)])  # 建立增廣矩陣

    for i in range(n):
        # 在 i 列中找到絕對值對大的行
        max_row = np.argmax(np.abs(Ab[i:, i])) + i
        # 將 pivot 換到當前行
        Ab[[i, max_row]] = Ab[[max_row, i]]

        # 對下方的行進行消去
        for j in range(i + 1, n):
            factor = Ab[j, i] / Ab[i, i]
            Ab[j, i:] -= factor * Ab[i, i:]

    # 反向替代
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i + 1 : n], x[i + 1 : n])) / Ab[i, i]

    return x


if __name__ == "__main__":
    A = np.array([[0.1, 51.7], [5.1, -7.3]], dtype=np.float16)
    b = np.array([104, 16], dtype=np.float16)

    sol = gaussian_elimination(A, b)

    # 格式化輸出為三位有效數字
    sol_formatted = np.round(sol, 3)

    print(sol_formatted)

```

The result is `[5.998 2.   ]`, which is closer to the expected result.

### (c) repeat (a) but use scaled partial pivoting

The code for solving the system of equations with scaled partial pivoting:

```python
import numpy as np


def gaussian_elimination(A, b):
    n = len(b)
    Ab = np.hstack([A, b.reshape(-1, 1)])  # 創建增廣矩陣
    scaling_factors = np.max(np.abs(A), axis=1)  # 計算比例

    for i in range(n):
        # 在 i 列中找到比例絕對值最大的行
        row_ratios = np.abs(Ab[i:n, i]) / scaling_factors[i:n]
        max_ratio_index = np.argmax(row_ratios) + i

        # 將 pivot 換到當前行
        Ab[[i, max_ratio_index]] = Ab[[max_ratio_index, i]]
        scaling_factors[[i, max_ratio_index]] = scaling_factors[[max_ratio_index, i]]  # 交換比例因子

        # 消去
        for j in range(i + 1, n):
            factor = Ab[j, i] / Ab[i, i]
            Ab[j, i:] -= factor * Ab[i, i:]

    # 反向替代
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i + 1 : n], x[i + 1 : n])) / Ab[i, i]

    return x


if __name__ == "__main__":
    A = np.array([[0.1, 51.7], [5.1, -7.3]], dtype=np.float16)
    b = np.array([104, 16], dtype=np.float16)

    sol = gaussian_elimination(A, b)

    # 格式化輸出為三位有效數字
    sol_formatted = np.round(sol, 3)

    print(sol_formatted)

```

The result is `[5.998 2.   ]`, which is the same as (b).

## Question 3

The code for LU decomposition of matrix A:

```python
import numpy as np

# have to use this instead of scipy.linalg.lu because the latter does not support LU decomposition without permutation
from scipy.sparse.linalg import splu

if __name__ == "__main__":
    A = np.array(
        [
            [2, -1, 3, 2],
            [2, 2, 0, 4],
            [1, 1, -2, 2],
            [1, 3, 4, -1],
        ],
        dtype=float,
    )
    n = A.shape[0]

    # A 不是一個非奇異的方陣
    if not (A.shape[0] == A.shape[1] and np.linalg.det(A) != 0):
        exit(1)

    # 對 A 做 LU 分解 (PA = LU)
    slu = splu(
        A,
        permc_spec="NATURAL",
        diag_pivot_thresh=0,
        options={"SymmetricMode": True},
    )

    # 做 LDU 分解讓 L 對角線變成 2
    L = slu.L.toarray()
    U = slu.U.toarray()
    D = np.diag([2] * n)
    Dinv = np.diag([1 / 2] * n)

    # A = (L @ D) @ (Dinv @ U)
    L_D = np.dot(L, D)
    Dinv_U = np.dot(Dinv, U)

    print(L_D)
    print(Dinv_U)
```

This yields the result:

L matrix:
```json
[[ 2.          0.          0.          0.        ]
 [ 2.          2.          0.          0.        ]
 [ 1.          1.          2.          0.        ]
 [ 1.          2.33333333 -6.          2.        ]]
```

U matrix:
```json
[[ 1.         -0.5         1.5         1.        ]
 [ 0.          1.5        -1.5         1.        ]
 [ 0.          0.         -1.          0.        ]
 [ 0.          0.          0.         -2.16666667]]
```

## Question 4

The code for solving the augmented matrix using Jacobi method:

```python
import numpy as np


def jacobi_method(A, b, x):
    D = np.diag(A.diagonal())
    Dinv = np.linalg.inv(D)
    R = A - D
    L = np.tril(R)
    U = np.triu(R)
    T = -np.dot(Dinv, R)
    C = np.dot(Dinv, b)

    # accurate to 5 significant digits
    tolerance = 1e-5
    retries = 0
    while abs(A @ x - b).max() > tolerance:
        x = np.dot(T, x) + C
        retries += 1

    return x, retries


if __name__ == "__main__":
    A = np.array(
        [
            [7, -3, 4],
            [-3, 2, 6],
            [2, 5, 3],
        ],
        dtype=float,
    )
    b = np.array([6, 2, -5], dtype=float)
    x = np.array([0, 0, 0], dtype=float)

    # make A diagonally dominant
    A[[1, 2]] = A[[2, 1]]
    b[[1, 2]] = b[[2, 1]]

    x, itr = jacobi_method(A, b, x)
    print(x)
    print(itr)

```

This yields the result `[-0.14332486 -1.37459432  0.71987058]` and `34` iterations were required to achieve the desired accuracy.

## Question 5

The code for solving the augmented matrix using Gauss-Seidel method:

```python
import numpy as np


def gauss_seidel_method(A, b):
    """
    xk+1 = 1/7 * (6 + 3yk - 4zk)
    yk+1 = 1/5 * (-5 - 2xk+1 - 3zk)
    zk+1 = 1/6 * (2 + 3xk+1 - 2yk+1)
    """
    # initial guess
    prev = np.array([float(0) for _ in range(3)])

    # initialize x, y, z as np.array
    ans = np.array([float(0) for _ in range(3)])
    ans[0] = 1 / 7 * (6 + 3 * prev[1] - 4 * prev[2])
    ans[1] = 1 / 5 * (-5 - 2 * ans[0] - 3 * prev[2])
    ans[2] = 1 / 6 * (2 + 3 * ans[0] - 2 * ans[1])

    tolerance = 1e-5
    retries = 1
    diff = np.abs(ans - prev)
    while diff.max() > tolerance:
        prev = ans.copy()
        ans[0] = 1 / 7 * (6 + 3 * prev[1] - 4 * prev[2])
        ans[1] = 1 / 5 * (-5 - 2 * ans[0] - 3 * prev[2])
        ans[2] = 1 / 6 * (2 + 3 * ans[0] - 2 * ans[1])
        retries += 1
        diff = np.abs(ans - prev)

    return ans, retries


if __name__ == "__main__":
    A = np.array(
        [
            [7, -3, 4],
            [-3, 2, 6],
            [2, 5, 3],
        ],
        dtype=float,
    )
    b = np.array([6, 2, -5], dtype=float)

    # make A diagonally dominant
    A[[1, 2]] = A[[2, 1]]
    b[[1, 2]] = b[[2, 1]]

    sol, itr = gauss_seidel_method(A, b)
    print(sol)
    print(itr)

```

This yields the result `[-0.14332299 -1.37459376  0.71986976]` and `14` iterations were required to achieve the desired accuracy, which is `20` iterations less than Jacobi method.

## Question 6

> Note: Tolerance is set to `1e-5` for all methods.

### (a) Use the Jacobi method

The code for solving the augmented matrix using Jacobi method:

```python
import numpy as np


def jacobi_method(A, b, x):
    D = np.diag(A.diagonal())
    Dinv = np.linalg.inv(D)
    R = A - D
    L = np.tril(R)
    U = np.triu(R)
    T = -np.dot(Dinv, R)
    C = np.dot(Dinv, b)

    # accurate to 5 significant digits
    tolerance = 1e-5
    retries = 0
    while abs(A @ x - b).max() > tolerance and retries < 10000:
        x = np.dot(T, x) + C
        retries += 1

    return x, retries


if __name__ == "__main__":
    A = np.array(
        [
            [2, -2],
            [-2, 2],
        ],
        dtype=float,
    )
    b = np.array([0, 0], dtype=float)
    xs = [
        np.array([1, 1], dtype=float),
        np.array([1, -1], dtype=float),
        np.array([-1, 1], dtype=float),
        np.array([2, 5], dtype=float),
        np.array([5, 2], dtype=float),
    ]

    for x in xs:
        sol, itr = jacobi_method(A, b, x)
        print(f"when x is {x}, the solution is {sol} and the number of iterations is: {itr if itr != 10000 else 'inf'}")

```

This yields the result:

```json
when x is [1. 1.], the solution is [1. 1.] and the number of iterations is: 0                                              
when x is [ 1. -1.], the solution is [ 1. -1.] and the number of iterations is: inf                                        
when x is [-1.  1.], the solution is [-1.  1.] and the number of iterations is: inf                                        
when x is [2. 5.], the solution is [2. 5.] and the number of iterations is: inf                                            
when x is [5. 2.], the solution is [5. 2.] and the number of iterations is: inf
```

The solution is accurate when the initial guess is `[1, 1]`, but diverges when the initial guess is `[1, -1]`, `[-1, 1]`, `[2, 5]`, `[5, 2]`.

### (b) Use the Gauss-Seidel method

The code for solving the augmented matrix using Gauss-Seidel method:

```python
import numpy as np


def gauss_seidel_method(A, b, x):
    """
    xk+1 = 1/2 * (0 + 2yk)
    yk+1 = 1/2 * (0 + 2xk+1)
    """
    # initial guess
    prev = x

    # initialize x, y, z as np.array
    ans = np.array([float(0) for _ in range(2)])
    ans[0] = 1 / 2 * (0 + 2 * prev[1])
    ans[1] = 1 / 2 * (0 + 2 * ans[0])

    tolerance = 1e-5
    retries = 1
    diff = np.abs(ans - prev)
    while diff.max() > tolerance:
        prev = ans.copy()
        ans[0] = 1 / 2 * (0 + 2 * prev[1])
        ans[1] = 1 / 2 * (0 + 2 * ans[0])
        retries += 1
        diff = np.abs(ans - prev)

    return ans, retries


if __name__ == "__main__":
    A = np.array(
        [
            [2, -2],
            [-2, 2],
        ],
        dtype=float,
    )
    b = np.array([0, 0], dtype=float)
    xs = [
        np.array([1, 1], dtype=float),
        np.array([1, -1], dtype=float),
        np.array([-1, 1], dtype=float),
        np.array([2, 5], dtype=float),
        np.array([5, 2], dtype=float),
    ]
    
    for x in xs:
        sol, itr = gauss_seidel_method(A, b, x)
        print(f"when x is {x}, the solution is {sol} and the number of iterations is: {itr}")

```

This yields the result:

```json
when x is [1. 1.], the solution is [1. 1.] and the number of iterations is: 1                                              
when x is [ 1. -1.], the solution is [-1. -1.] and the number of iterations is: 2                                          
when x is [-1.  1.], the solution is [1. 1.] and the number of iterations is: 2                                            
when x is [2. 5.], the solution is [5. 5.] and the number of iterations is: 2                                              
when x is [5. 2.], the solution is [2. 2.] and the number of iterations is: 2
```

The gauss-seidel method converges for all initial guesses, and the solution is accurate.

### (c) repeat the above but replace `-2` with `-1.99`

The code for solving the augmented matrix using Jacobi method:

```python
import numpy as np


def jacobi_method(A, b, x):
    D = np.diag(A.diagonal())
    Dinv = np.linalg.inv(D)
    R = A - D
    L = np.tril(R)
    U = np.triu(R)
    T = -np.dot(Dinv, R)
    C = np.dot(Dinv, b)

    # accurate to 5 significant digits
    tolerance = 1e-5
    retries = 0
    while abs(A @ x - b).max() > tolerance and retries < 10000:
        x = np.dot(T, x) + C
        retries += 1

    return x, retries


if __name__ == "__main__":
    A = np.array(
        [
            [2, -1.99],
            [-1.99, 2],
        ],
        dtype=float,
    )
    b = np.array([0, 0], dtype=float)
    xs = [
        np.array([1, 1], dtype=float),
        np.array([1, -1], dtype=float),
        np.array([-1, 1], dtype=float),
        np.array([2, 5], dtype=float),
        np.array([5, 2], dtype=float),
    ]

    for x in xs:
        sol, itr = jacobi_method(A, b, x)
        print(f"when x is {x}, the solution is {sol} and the number of iterations is: {itr if itr != 10000 else 'inf'}")

```

This yields the result:

```json
when x is [1. 1.], the solution is [0.00099547 0.00099547] and the number of iterations is: 1379                           
when x is [ 1. -1.], the solution is [-2.50487904e-06  2.50487904e-06] and the number of iterations is: 2573               
when x is [-1.  1.], the solution is [ 2.50487904e-06 -2.50487904e-06] and the number of iterations is: 2573               
when x is [2. 5.], the solution is [8.30327428e-06 3.32130971e-06] and the number of iterations is: 2655                   
when x is [5. 2.], the solution is [3.32130971e-06 8.30327428e-06] and the number of iterations is: 2655
```

The code for solving the augmented matrix using Gauss-Seidel method:

```python
import numpy as np


def gauss_seidel_method(A, b, x):
    """
    xk+1 = 1/2 * (0 + 1.99yk)
    yk+1 = 1/2 * (0 + 1.99xk+1)
    """
    # initial guess
    prev = x

    # initialize x, y, z as np.array
    ans = np.array([float(0) for _ in range(2)])
    ans[0] = 1 / 2 * (0 + 1.99 * prev[1])
    ans[1] = 1 / 2 * (0 + 1.99 * ans[0])

    tolerance = 1e-5
    retries = 1
    diff = np.abs(ans - prev)
    while diff.max() > tolerance:
        prev = ans.copy()
        ans[0] = 1 / 2 * (0 + 1.99 * prev[1])
        ans[1] = 1 / 2 * (0 + 1.99 * ans[0])
        retries += 1
        diff = np.abs(ans - prev)

    return ans, retries


if __name__ == "__main__":
    A = np.array(
        [
            [2, -1.99],
            [-1.99, 2],
        ],
        dtype=float,
    )
    b = np.array([0, 0], dtype=float)
    xs = [
        np.array([1, 1], dtype=float),
        np.array([1, -1], dtype=float),
        np.array([-1, 1], dtype=float),
        np.array([2, 5], dtype=float),
        np.array([5, 2], dtype=float),
    ]

    for x in xs:
        sol, itr = gauss_seidel_method(A, b, x)
        print(f"when x is {x}, the solution is {sol} and the number of iterations is: {itr}")

```

This yields the result:

```json
when x is [1. 1.], the solution is [0.00098554 0.00098061] and the number of iterations is: 691                            
when x is [ 1. -1.], the solution is [-0.00098554 -0.00098061] and the number of iterations is: 691                        
when x is [-1.  1.], the solution is [0.00098554 0.00098061] and the number of iterations is: 691                          
when x is [2. 5.], the solution is [0.0009909  0.00098595] and the number of iterations is: 851                            
when x is [5. 2.], the solution is [0.00098694 0.000982  ] and the number of iterations is: 760
```

The gauss-seidel method converges faster for all initial guesses compared to Jacobi method.
