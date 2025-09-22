import DnnLib

def create_dense_layer(input_dim, layer_config):
    """Factory para crear capas densas con regularización"""
    activation_map = {
        'relu': DnnLib.ActivationType.RELU,
        'sigmoid': DnnLib.ActivationType.SIGMOID,
        'tanh': DnnLib.ActivationType.TANH,
        'softmax': DnnLib.ActivationType.SOFTMAX
    }
    
    activation_type = activation_map.get(
        layer_config.get('activation', 'relu').lower(), 
        DnnLib.ActivationType.RELU
    )
    
    layer = DnnLib.DenseLayer(input_dim, layer_config['units'], activation_type)
    
    if 'regularizer' in layer_config:
        reg_config = layer_config['regularizer']
        reg_type = reg_config['type'].upper() 
        lambda_val = reg_config['lambda']
        
        # Mapear a enum de DnnLib
        if reg_type == 'L1':
            layer.set_regularizer(DnnLib.RegularizerType.L1, lambda_val)
        elif reg_type == 'L2':
            layer.set_regularizer(DnnLib.RegularizerType.L2, lambda_val)
        
        print(f"    Regularizador: {reg_type} (λ={lambda_val})")
    
    print(f"Capa densa creada: {input_dim} -> {layer_config['units']}, "
          f"Activación: {layer_config.get('activation', 'relu')}")
    
    return layer

def apply_regularization(layer, layer_config):
    """Aplicar regularización a una capa si está configurada"""
    if 'regularizer' in layer_config:
        reg_config = layer_config['regularizer']
        reg_type = getattr(DnnLib.RegularizerType, reg_config['type'].upper())
        lambda_val = reg_config.get('lambda', 0.01)
        layer.set_regularizer(reg_type, lambda_val)
        print(f"  + Regularización {reg_config['type']} (λ={lambda_val})")
        return True
    return False

def create_optimizer(opt_config):
    """Factory para crear optimizers"""
    opt_params = opt_config.get('params', {})
    
    if opt_config['type'] == 'adam':
        return DnnLib.Adam(
            learning_rate=opt_params.get('learning_rate', 0.0001),
            beta1=opt_params.get('beta1', 0.9),
            beta2=opt_params.get('beta2', 0.999),
            epsilon=opt_params.get('epsilon', 1e-8)
        )
    elif opt_config['type'] == 'sgd':
        return DnnLib.SGD(
            learning_rate=opt_params.get('learning_rate', 0.01),
            momentum=opt_params.get('momentum', 0.0)
        )
    elif opt_config['type'] == 'rmsprop':
        return DnnLib.RMSprop(
            learning_rate=opt_params.get('learning_rate', 0.001),
            decay_rate=opt_params.get('decay_rate', 0.9),
            epsilon=opt_params.get('epsilon', 1e-8)
        )
    else:
        raise ValueError(f"Optimizer no soportado: {opt_config['type']}")