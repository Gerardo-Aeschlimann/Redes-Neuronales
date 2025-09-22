import numpy as np
import matplotlib.pyplot as plt
import DnnLib

# ESTE CODIGO MUESTRA IMAGENES GRAFICAMENTE
data = np.load("/workspace/mnist_test.npz") 
images = data["images"]
labels = data["labels"]

print("Shape imágenes:", images.shape)
print("Shape etiquetas:", labels.shape)

# Tomar 3 ejemplos
X = images[:3]
y = labels[:3]
print("Etiquetas:", y)


for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.imshow(X[i], cmap="gray")
    plt.title(f"Label: {y[i]}")
    plt.axis("off")
plt.show()


X_flat = X.reshape(3, -1)


layer = DnnLib.DenseLayer(784, 10, DnnLib.ActivationType.SOFTMAX)


out = layer.forward(X_flat)
print("Salida de la red (3x10):")
print(out)