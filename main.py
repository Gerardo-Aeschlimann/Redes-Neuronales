import argparse
from parser import create_parser
from Trainer import MNISTModelTrainer
from Loader import load_mnist_data

def main():
    parser = create_parser()
    args = parser.parse_args()
    
    # Crear y configurar entrenador
    trainer = MNISTModelTrainer()
    trainer.load_model_config(args.model)
    trainer.build_model_from_config(use_regularization=args.reg, use_dropout=args.drop)
    
    # Cargar datos
    X, y, true_labels = load_mnist_data(
        args.dataset, 
        trainer.model_config,
        num_classes=trainer.layers[-1].weights.shape[0] if trainer.layers else None
    )
    
    # Entrenar modelo
    print(f"Batch Size: {args.batchsize}")
    print("\nIniciando entrenamiento...")
    
    history = trainer.train(X, y, true_labels, args.epochs, args.batchsize, args.split)
    
    # Guardar y mostrar resultados
    trainer.save_trained_model(args.modelgen)
    
    print("\nEntrenamiento completado!")
    final_accuracy = history['val_accuracy'][-1]
    final_accuracy*=100
    print(f"Precisión final en validación: {final_accuracy:.4f}%")

if __name__ == "__main__":
    main()