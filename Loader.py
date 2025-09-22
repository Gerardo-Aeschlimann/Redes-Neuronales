import numpy as np

def load_mnist_data(npz_file_path, model_config=None, num_classes=None):
    """Cargar y preprocesar datos MNIST"""
    print(f"Cargando datos MNIST desde: {npz_file_path}")
    data = np.load(npz_file_path)
    images = data["images"]
    labels = data["labels"]
    
    print(f"Imágenes: {images.shape}, Etiquetas: {labels.shape}")
    
    # Preprocesamiento
    X = images.reshape(images.shape[0], -1).astype(np.float64)
    
    # Escalar
    if model_config and 'preprocess' in model_config and 'scale' in model_config['preprocess']:
        scale_factor = model_config['preprocess']['scale']
        X = X / scale_factor
        print(f"Imágenes escaladas por {scale_factor}")
    
    # One-hot encoding
    if num_classes:
        y = np.zeros((labels.shape[0], num_classes), dtype=np.float64)
        y[np.arange(labels.shape[0]), labels] = 1.0
    else:
        y = labels
    
    print(f"Datos preprocesados: X {X.shape}, y {y.shape}")
    
    return X, y, labels