import DnnLib

class RegularizationManager:
    """Manejador de regularización y dropout"""
    
    def __init__(self):
        self.regularizers = []
        self.dropout_layers = []
    
    def add_regularizer(self, layer):
        """Añadir capa a regularizadores"""
        self.regularizers.append(layer)
    
    def add_dropout_layer(self, dropout_layer):
        """Añadir capa de dropout"""
        self.dropout_layers.append(dropout_layer)
    
    def set_training_mode(self, training=True):
        """Activar/desactivar modo training para dropout"""
        for dropout in self.dropout_layers:
            dropout.training = training
        return f"Modo {'training' if training else 'inference'} para dropout"
    
    def compute_regularization_loss(self):
        """Calcular pérdida total de regularización"""
        total_reg_loss = 0.0
        for layer in self.regularizers:
            total_reg_loss += layer.compute_regularization_loss()
        return total_reg_loss
    
    def create_dropout_layer(self, dropout_config):
        """Crear capa de dropout desde configuración"""
        dropout_rate = dropout_config.get('rate', 0.5)
        dropout_layer = DnnLib.Dropout(dropout_rate=dropout_rate)
        self.add_dropout_layer(dropout_layer)
        return dropout_layer