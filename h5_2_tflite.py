import argparse
import layers
from tensorflow import lite

def main(args):
    converter = lite.TFLiteConverter.from_keras_model_file("models/"+args.style+".h5",
            custom_objects={
                "InputNormalize": layers.InputNormalize,
                "ReflectionPadding2D": layers.ReflectionPadding2D,
                "Denormalize": layers.Denormalize,
                "UnPooling2D": layers.UnPooling2D,
                })
    tfmodel = converter.convert()

    open("models/"+args.style+".tflite", "wb") .write(tfmodel)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Real-time style transfer')

    parser.add_argument('--style', '-s', type=str, required=True,
                        help='style image file name without extension')

    args = parser.parse_args()
    main(args)
