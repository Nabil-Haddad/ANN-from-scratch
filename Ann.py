import numpy as np

W_input_3X1 = np.array([-2.455127, -1.038431, -1.896615])
input_bias = np.array( [0.7409607, -1.7004427, -0.5459574])

W_hidden_2X3 = np.array([[-1.2329185, -0.4924955, -0.9389248],[-0.9744559, -1.6208865, -1.9251845]])
hidden_bias = np.array( [0.08913168, 1.77876475])

W_ouptut_2X1 = np.array([2.386606, 1.423125])
output_bias = np.array([1.367115])


def sigmoid(u:float)->float:
    return 1/ (1 + np.exp(-u))

if __name__ == "__main__":
    print("This is a simulation of an ANN")
    print(f"This is the results of sigmoid(0) -> {sigmoid(0)}")