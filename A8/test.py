import matplotlib.pyplot as plt


def generateMeshMatrix():
    A = [[0.0] * 19 for i in range(19)]
    for i in range(19):
        A[i][i] = -4
        if (i < 10):
            if (i < 8):
                A[i][i + 5] = 1
            if (i != 0 and i != 5):
                A[i][i - 1] = 1
            if (i + 1 == 5 or i + 1 == 10):
                A[i][i - 1] += 1
            else:
                A[i][i + 1] = 1
            if i > 4:
                A[i][i - 5] = 1

        elif (i < 13):
            A[i][i - 5] = 1
            A[i][i + 3] = 1
            if (i != 10):
                A[i][i - 1] = 1
            if (i != 12):
                A[i][i + 1] = 1
        else:
            A[i][i - 3] = 1
            if (i < 16):
                A[i][i + 3] = 1
            else:
                A[i][i - 3] += 1
            if (i != 15) and (i != 18):
                A[i][i + 1] = 1
            if (i != 13) and (i != 16):
                A[i][i - 1] = 1
    return A

T = generateMeshMatrix()
print(T)
plt.imshow(T)
plt.colorbar()
plt.show()