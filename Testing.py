import json
import numpy as np
import DnnLib
import argparse

# Mapeo de activaciones
ACTIVATION_MAP = {
    "sigmoid": DnnLib.ActivationType.SIGMOID,
    "tanh": DnnLib.ActivationType.TANH,
    "relu": DnnLib.ActivationType.RELU,
    "softmax": DnnLib.ActivationType.SOFTMAX
}


# Cargar modelo
def load_model_from_json(path):
    with open(path, "r") as f:
        config = json.load(f)

    input_shape = config.get("input_shape", [28, 28])
    input_size = int(np.prod(input_shape))

    preprocess_scale = config.get("preprocess", {}).get("scale", 1.0)

    layers = []
    in_features = input_size

    for i, layer_cfg in enumerate(config["layers"]):
        if layer_cfg["type"] != "dense":
            continue  

        units = layer_cfg["units"]
        activation_str = layer_cfg.get("activation", "relu").lower()
        activation = ACTIVATION_MAP.get(activation_str, DnnLib.ActivationType.RELU)

        # Creamos capa
        layer = DnnLib.DenseLayer(in_features, units, activation)

        # Cargamos pesos
        W = layer_cfg.get("W", [])
        b = layer_cfg.get("b", [])
        if W and b:
            W = np.array(W)
            b = np.array(b)

            if W.shape == (units, in_features) and b.shape == (units,):
                layer.weights = W
                layer.bias = b
            elif W.shape == (in_features, units) and b.shape == (units,):
                #Transpuesta
                layer.weights = W.T
                layer.bias = b
            else:
                print(f" Warning: Shapes inválidas en capa {i}. "
                      f"Esperado W=({units},{in_features}), b=({units},). "
                      f"Recibido W={W.shape}, b={b.shape}. Usando inicialización aleatoria.")

        layers.append(layer)
        in_features = units

    return layers, preprocess_scale, input_shape


# Forward de toda la red
def forward_network(layers, X):
    out = X
    for layer in layers:
        out = layer.forward(out)
    return out

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Calculo Precision.")
    parser.add_argument('--dataset',default="/workspace/Datasets/mnist_train.npz",help="Ruta de dataset")
    parser.add_argument('--modelo',default="/workspace/Modelos/modelo.json",help="Ruta de modelo")
    args=parser.parse_args()

    layers, scale, input_shape = load_model_from_json(args.modelo)
    input_size = int(np.prod(input_shape))

    #dataset
    data = np.load(args.dataset)
    images = data["images"]   # (N,28,28)
    labels = data["labels"]   # (N,)
    print("Dataset:", images.shape, labels.shape)

    # Asegurar que las imágenes coincidan con input_shape
    N = images.shape[0]
    X = images.reshape(N, input_size)   # (N, input_size)

    # Preprocesar
    X_scaled = X / scale

    # Hacemos forward
    outputs = forward_network(layers, X_scaled)

    # Predicciones
    y_pred = np.argmax(outputs, axis=1)

    #precisión
    accuracy = np.mean(y_pred == labels)
    print(f"✅ Precisión: {accuracy*100:.2f}%")
