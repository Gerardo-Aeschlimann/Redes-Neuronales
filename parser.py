import argparse

def create_parser():
    parser = argparse.ArgumentParser(description="Train Mnist Model.")
    
    parser.add_argument('--epochs', type=int, default=10, 
                       help="Número de épocas")
    parser.add_argument('--batchsize', type=int, default=64, 
                       help="Tamaño de batch")
    parser.add_argument('--model', default="/workspace/Modelos/modelo.json", 
                       help="Ruta de modelo config")
    parser.add_argument('--dataset', default="/workspace/Datasets/mnist_train.npz", 
                       help="Ruta de dataset")
    parser.add_argument('--modelgen', default="/workspace/Modelos/modeloGEN.json", 
                       help="Ruta de modelo generado")
    parser.add_argument('--split', type=float, default=0.1,
                       help="Porcentaje de validación")
    parser.add_argument('--reg', action='store_true', help="Usar regularización")

    parser.add_argument('--drop', action='store_true', help="Usar dropout")
    
    return parser