import numpy as np
import matplotlib.pyplot as plt

W_input_3X1 = np.array([[-2.455127], [-1.038431], [-1.896615]])
input_bias = np.array( [0.7409607, -1.7004427, -0.5459574])

W_hidden_2X3 = np.array([[-1.2329185, -0.4924955, -0.9389248],[-0.9744559, -1.6208865, -1.9251845]])
hidden_bias = np.array( [0.08913168, 1.77876475])

W_ouptut_1X2 = np.array([[2.386606, 1.423125]]) 
output_bias = np.array([-1.367115])

x = np.array([0.707])

layers = ([W_input_3X1,W_hidden_2X3,W_ouptut_1X2],[input_bias,hidden_bias,output_bias ])

def sigmoid(u):
    return 1/ (1 + np.exp(-u))

def layer(W:np.ndarray, x:np.ndarray ,b:np.ndarray, activation, active :bool = True ,debug :bool = True):
    if debug:
        print(f"The shape of W is {W.shape}")
        print(f"The shape of x is {x.shape}")
    if active :
        return activation(W @ x + b)
    else:
        return W @ x + b

def predict(x:np.ndarray, layers):
    y = x 
    length = len(layers[0])
    for i, (l, b) in enumerate(zip(*layers)):
        if i == 3 -1 :
            activate = False
        else :
            activate = True
        y = layer(l,y,b, sigmoid, active=activate)
    return y


if __name__ == "__main__":
    print("This is a simulation of an ANN")
    print(f"This is the results of sigmoid(0) -> {sigmoid(0)}")
    print(f"This is the results of trying layer 1 -> {layer(W_input_3X1, x, input_bias,sigmoid ,True)}")
    print(f"This is the results of using all layers -> {predict(x,layers)}")

    theta = np.linspace(0, 1.57, 100)
    sin_values = np.sin(theta)
    net_values = np.array([predict(np.array([t]), layers)[0] for t in theta])

    print("max |network - sin| =", np.max(np.abs(net_values - sin_values)))

    plt.figure(figsize=(7, 4.5))
    plt.plot(theta, sin_values, "k--", linewidth=2, label=r"target: $\sin(\theta)$")
    plt.plot(theta, net_values, color="tab:orange", linewidth=3, label="network output")
    plt.scatter([0.707], [predict(x, layers)[0]], color="red", zorder=5, label=r"$\theta=0.707$")
    plt.title("ANN forward pass vs sine")
    plt.xlabel(r"$\theta$ (radians)")
    plt.ylabel("output")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.show()