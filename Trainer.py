import json
import numpy as np
import DnnLib
from Capa import create_optimizer, create_dense_layer,apply_regularization
from regularization import RegularizationManager
from Loader import load_mnist_data

class MNISTModelTrainer:
    def __init__(self):
        self.model_config = None
        self.layers = []
        self.optimizer = None
        self.reg_manager = RegularizationManager()

    def _get_activation_type(self, layer_config):
        """Mapear string de activación a enum de DnnLib"""
        activation_map = {
            'relu': DnnLib.ActivationType.RELU,
            'sigmoid': DnnLib.ActivationType.SIGMOID,
            'tanh': DnnLib.ActivationType.TANH,
            'softmax': DnnLib.ActivationType.SOFTMAX
        }
        
        activation_str = layer_config.get('activation', 'relu').lower()
        return activation_map.get(activation_str, DnnLib.ActivationType.RELU)

    def _configure_optimizer(self):
        """Configurar el optimizer basado en la configuración del modelo"""
        if not self.model_config or 'optimizer' not in self.model_config:
            # Usar valores por defecto si no hay configuración
            self.optimizer = DnnLib.Adam(learning_rate=0.001)
            print("Optimizador configurado: adam (default)")
            return
            
        opt_config = self.model_config['optimizer']
        opt_params = opt_config.get('params', {})
        
        if opt_config['type'] == 'adam':
            self.optimizer = DnnLib.Adam(
                learning_rate=opt_params.get('learning_rate', 0.001),
                beta1=opt_params.get('beta1', 0.9),
                beta2=opt_params.get('beta2', 0.999),
                epsilon=opt_params.get('epsilon', 1e-8)
            )
        elif opt_config['type'] == 'sgd':
            self.optimizer = DnnLib.SGD(
                learning_rate=opt_params.get('learning_rate', 0.01),
                momentum=opt_params.get('momentum', 0.0)
            )
        elif opt_config['type'] == 'rmsprop':
            self.optimizer = DnnLib.RMSprop(
                learning_rate=opt_params.get('learning_rate', 0.001),
                decay_rate=opt_params.get('decay_rate', 0.9),
                epsilon=opt_params.get('epsilon', 1e-8)
            )
        else:
            # Fallback a Adam
            self.optimizer = DnnLib.Adam(learning_rate=0.001)
            
        print(f"Optimizador configurado: {opt_config['type']}")
        
    def load_model_config(self, json_file_path):
        """Cargar configuración del modelo desde archivo JSON"""
        with open(json_file_path, 'r') as f:
            self.model_config = json.load(f)
        print("Configuración del modelo cargada exitosamente")
        
    def build_model_from_config(self, use_regularization=False, use_dropout=False):
        if not self.model_config:
            raise ValueError("Primero debe cargar la configuración del modelo")
            
        self.layers = []
        in_features = np.prod(self.model_config['input_shape'])
        
        print(f"Dimension inicial: {in_features}")
        
        for i, layer_config in enumerate(self.model_config['layers']):
            print(f"Procesando capa {i}: {layer_config['type']}")
            
            if layer_config['type'] == 'dense':
                # Crear capa densa
                units = layer_config['units']
                activation_type = self._get_activation_type(layer_config)
                
                layer = DnnLib.DenseLayer(in_features, units, activation_type)
                
                # Configurar regularización solo si está activada
                if use_regularization and 'regularizer' in layer_config:
                    reg_config = layer_config['regularizer']
                    reg_type = reg_config['type'].upper()
                    lambda_val = reg_config['lambda']
                    
                    if reg_type == 'L1':
                        layer.set_regularizer(DnnLib.RegularizerType.L1, lambda_val)
                    elif reg_type == 'L2':
                        layer.set_regularizer(DnnLib.RegularizerType.L2, lambda_val)
                    print(f"  + Regularización {reg_type} (λ={lambda_val})")
                
                self.layers.append(layer)
                in_features = units  # Actualizar para siguiente capa
                print(f"Capa densa creada: {in_features} units")
                
            elif layer_config['type'] == 'dropout':
                if use_dropout:
                    # Crear capa dropout solo si está activada
                    rate = layer_config.get('rate', 0.5)
                    dropout_layer = DnnLib.Dropout(dropout_rate=rate)
                    self.layers.append(dropout_layer)
                    print(f"Capa Dropout creada: rate={rate}")
                else:
                    print(f"Capa Dropout omitida (bandera --drop no activada)")
        
        # Configurar optimizer
        self._configure_optimizer()
        
    def load_mnist_data(self, npz_file_path):
        """Cargar y preprocesar datos MNIST"""
        return load_mnist_data(
            npz_file_path, 
            self.model_config,
            self.layers[-1].weights.shape[0] if self.layers else None
        )
    
    def train(self, X, y, true_labels, epochs=10, batch_size=32, validation_split=0.1):
        """Entrenar el modelo con opciones de regularización y dropout"""
        n_samples = X.shape[0]
        val_size = int(n_samples * validation_split)
        train_size = n_samples - val_size
        
        # Dividir en entrenamiento y validación
        indices = np.random.permutation(n_samples)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        true_labels_val = true_labels[val_indices]
        
        print(f"Entrenamiento: {X_train.shape[0]} muestras")
        print(f"Validación: {X_val.shape[0]} muestras")
        
        # Verificar si la última capa usa softmax (caso especial)
        last_layer_uses_softmax = (self.layers[-1].activation_type == DnnLib.ActivationType.SOFTMAX)
        
        # Activar modo training para capas dropout
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = True
        
        # Historial de entrenamiento
        history = {
            'train_loss': [], 'val_loss': [], 'val_accuracy': [],
            'reg_loss': [], 'total_loss': []
        }
        
        for epoch in range(epochs):
            # Entrenamiento por mini-lotes
            epoch_data_loss = 0.0
            epoch_reg_loss = 0.0
            n_batches = 0
            
            # Barajar datos de entrenamiento
            train_indices_shuffled = np.random.permutation(X_train.shape[0])
            X_train_shuffled = X_train[train_indices_shuffled]
            y_train_shuffled = y_train[train_indices_shuffled]
            
            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_train_shuffled[i:i+batch_size]
                y_batch = y_train_shuffled[i:i+batch_size]
                
                # Forward pass
                activation = X_batch
                for layer in self.layers:
                    activation = layer.forward(activation)
                output = activation
                
                # Calcular pérdida de datos
                data_loss = DnnLib.cross_entropy(output, y_batch)
                
                # ✅ CALCULAR PÉRDIDA DE REGULARIZACIÓN CORRECTAMENTE
                reg_loss = 0.0
                for layer in self.layers:
                    if hasattr(layer, 'compute_regularization_loss'):
                        reg_loss += layer.compute_regularization_loss()
                
                total_loss = data_loss + reg_loss
                
                epoch_data_loss += data_loss
                epoch_reg_loss += reg_loss
                
                # Backward pass - CASO ESPECIAL para softmax + cross entropy
                if last_layer_uses_softmax:
                    grad = output - y_batch
                else:
                    grad = DnnLib.cross_entropy_gradient(output, y_batch)
                
                # Propagación hacia atrás
                for layer in reversed(self.layers):
                    grad = layer.backward(grad)
                    # Solo actualizar capas densas (no dropout)
                    if not hasattr(layer, 'training'):
                        self.optimizer.update(layer)
                
                n_batches += 1
            
            avg_data_loss = epoch_data_loss / n_batches
            avg_reg_loss = epoch_reg_loss / n_batches
            avg_total_loss = avg_data_loss + avg_reg_loss
            
            # Validación (desactivar dropout)
            for layer in self.layers:
                if hasattr(layer, 'training'):
                    layer.training = False
            
            val_activation = X_val
            for layer in self.layers:
                val_activation = layer.forward(val_activation)
            val_output = val_activation
            
            val_loss = DnnLib.cross_entropy(val_output, y_val)
            
            # Calcular accuracy
            predicted_classes = np.argmax(val_output, axis=1)
            val_accuracy = np.mean(predicted_classes == true_labels_val)
            
            # Reactivar modo training para próximo epoch
            for layer in self.layers:
                if hasattr(layer, 'training'):
                    layer.training = True
            
            # Guardar historial
            history['train_loss'].append(avg_data_loss)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            history['reg_loss'].append(avg_reg_loss)
            history['total_loss'].append(avg_total_loss)
            
            print(f"Epoch {epoch+1}/{epochs} - "
                f"Val Loss: {val_loss:.4f} - "
                f"Val Accuracy: {val_accuracy:.4f}")
        
        # Dejar en modo inference al finalizar
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = False
        
        return history
    
    def evaluate(self, X, true_labels):
        """Evaluar el modelo en datos de prueba (siempre en modo inference)"""
        print(self.reg_manager.set_training_mode(False))
        
        activation = X
        for layer in self.layers:
            activation = layer.forward(activation)
        output = activation
        
        predicted_classes = np.argmax(output, axis=1)
        accuracy = np.mean(predicted_classes == true_labels)
        
        print(f"Precisión en conjunto de prueba: {accuracy:.4f}")
        return accuracy
    
    def save_trained_model(self, output_file_path):
        if not self.model_config or not self.layers:
            raise ValueError("No hay modelo para guardar")
            
        # Crear una copia de la configuración original
        trained_model_config = self.model_config.copy()
        
        # Contador separado para capas densas en la configuración
        dense_layer_idx = 0
        
        for i, layer in enumerate(self.layers):
            # Solo procesar capas densas (ignorar dropout en la configuración)
            if not hasattr(layer, 'training'):  # Es una capa densa
                if dense_layer_idx < len(trained_model_config['layers']):
                    # Buscar la próxima capa densa en la configuración
                    while (dense_layer_idx < len(trained_model_config['layers']) and 
                        trained_model_config['layers'][dense_layer_idx]['type'] != 'dense'):
                        dense_layer_idx += 1
                    
                    if dense_layer_idx < len(trained_model_config['layers']):
                        # Convertir numpy arrays a listas para JSON
                        trained_model_config['layers'][dense_layer_idx]['W'] = layer.weights.tolist()
                        trained_model_config['layers'][dense_layer_idx]['b'] = layer.bias.tolist()
                        
                        # Agregar información adicional sobre la capa
                        trained_model_config['layers'][dense_layer_idx]['input_dim'] = layer.weights.shape[1]
                        trained_model_config['layers'][dense_layer_idx]['output_dim'] = layer.weights.shape[0]
                        
                        dense_layer_idx += 1
        
        # Agregar metadata de entrenamiento
        trained_model_config['trained'] = True
        trained_model_config['training_info'] = {
            'optimizer_type': self.model_config['optimizer']['type'],
            'learning_rate': self.optimizer.learning_rate
        }
        
        # Guardar el modelo entrenado
        with open(output_file_path, 'w') as f:
            json.dump(trained_model_config, f, indent=2)
            
        print(f"Modelo entrenado guardado en: {output_file_path}")