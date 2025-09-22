# Redes-Neuronales
Proyecto de redes
Los siguientes comandos son llamados en terminal utilizando argparse puede usted consultar con --help las opciones.



Comandos disponibles:

1Testing docker run --rm -v ${PWD}:/workspace -w /workspace iderashn/dnn-q32025:latest python Testing.py --modelo /workspace/Modelos/modeloGenerado.json --dataset=/workspace/Datasets/mnist_test.npz



2Entrenamiento docker run --rm -v ${PWD}:/workspace -w /workspace iderashn/dnn-q32025:latest python main.py --epoch=20 --batchsize=64 --model=/workspace/Modelos/vacio.json --dataset=/workspace/Datasets/mnist_train.npz --modelgen=/workspace/Modelos/modeloGenerado.json