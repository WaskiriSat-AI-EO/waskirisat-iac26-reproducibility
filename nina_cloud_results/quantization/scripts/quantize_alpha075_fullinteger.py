import argparse
import random
from pathlib import Path

import numpy as np
import tensorflow as tf


IMG_SIZE = 224
SEED = 42


def collect_train_images(train_dir: Path):
    clear_images = sorted((train_dir / "clear").glob("*.png"))
    cloud_images = sorted((train_dir / "cloud").glob("*.png"))

    images = clear_images + cloud_images

    if not images:
        raise RuntimeError(
            f"No se encontraron imágenes PNG en:\n{train_dir}"
        )

    print(
        f"TRAIN: CLEAR={len(clear_images)} "
        f"CLOUD={len(cloud_images)} "
        f"TOTAL={len(images)}"
    )

    return images


def load_image(path: Path):
    """
    Carga la imagen en rango 0..255.

    IMPORTANTE:
    No hacemos /127.5 - 1 aquí porque el modelo
    ya contiene una capa Rescaling interna.
    """

    image = tf.io.read_file(str(path))
    image = tf.io.decode_png(image, channels=3)

    image = tf.image.resize(
        image,
        (IMG_SIZE, IMG_SIZE)
    )

    image = tf.cast(image, tf.float32)

    image = tf.expand_dims(
        image,
        axis=0
    )

    return image


def build_representative_dataset(
    image_paths,
    num_images
):

    rng = random.Random(SEED)

    image_paths = list(image_paths)

    num_images = min(
        num_images,
        len(image_paths)
    )

    selected = rng.sample(
        image_paths,
        num_images
    )

    print(
        f"Representative dataset: "
        f"{len(selected)} imágenes TRAIN B"
    )

    def representative_dataset():

        for image_path in selected:

            image = load_image(
                image_path
            )

            yield [image]

    return representative_dataset


def print_tflite_details(
    tflite_path: Path
):

    interpreter = tf.lite.Interpreter(
        model_path=str(tflite_path)
    )

    interpreter.allocate_tensors()

    input_details = (
        interpreter.get_input_details()
    )

    output_details = (
        interpreter.get_output_details()
    )

    print("\n" + "=" * 70)
    print("DETALLES DEL MODELO TFLITE")
    print("=" * 70)

    print("\nINPUT:")
    print(input_details)

    print("\nOUTPUT:")
    print(output_details)

    print("\nInput dtype:")
    print(input_details[0]["dtype"])

    print("Input quantization:")
    print(input_details[0]["quantization"])

    print("\nOutput dtype:")
    print(output_details[0]["dtype"])

    print("Output quantization:")
    print(output_details[0]["quantization"])


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Full Integer Quantization "
            "MobileNetV2 alpha=0.75"
        )
    )

    parser.add_argument(
        "--model",
        default=(
            "data/cloud/"
            "training_mobilenetv2_alpha075/"
            "mobilenetv2_best.keras"
        ),
    )

    parser.add_argument(
        "--train_dir",
        default=(
            "data/cloud/"
            "dataset_final_repo/train"
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "data/cloud/"
            "training_mobilenetv2_alpha075_fullinteger/"
            "mobilenetv2_alpha075_fullinteger_int8.tflite"
        ),
    )

    parser.add_argument(
        "--num_calibration_images",
        type=int,
        default=300,
    )

    args = parser.parse_args()

    model_path = Path(
        args.model
    ).resolve()

    train_dir = Path(
        args.train_dir
    ).resolve()

    output_path = Path(
        args.output
    ).resolve()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"No existe modelo:\n{model_path}"
        )

    if not train_dir.exists():
        raise FileNotFoundError(
            f"No existe TRAIN:\n{train_dir}"
        )

    print("\n" + "=" * 70)
    print(
        "FULL INTEGER QUANTIZATION "
        "— MobileNetV2 alpha=0.75"
    )
    print("=" * 70)

    print("\nCargando modelo:")
    print(model_path)

    model = tf.keras.models.load_model(
        model_path
    )

    print("\nModelo cargado.")

    train_images = collect_train_images(
        train_dir
    )

    representative_dataset = (
        build_representative_dataset(
            train_images,
            args.num_calibration_images
        )
    )

    print("\nCreando TFLiteConverter...")

    converter = (
        tf.lite.TFLiteConverter
        .from_keras_model(model)
    )

    # --------------------------------------------------
    # Full Integer Quantization
    # --------------------------------------------------

    converter.optimizations = [
        tf.lite.Optimize.DEFAULT
    ]

    converter.representative_dataset = (
        representative_dataset
    )

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]

    # --------------------------------------------------
    # Experimental:
    # desactivar per-channel para capas Dense
    # --------------------------------------------------

    converter._experimental_disable_per_channel_quantization_for_dense_layers = True

    # --------------------------------------------------
    # Interfaz completamente INT8
    # --------------------------------------------------

    converter.inference_input_type = (
        tf.int8
    )

    converter.inference_output_type = (
        tf.int8
    )

    print(
        "\nConvirtiendo modelo "
        "a Full Integer INT8..."
    )

    tflite_model = converter.convert()

    output_path.write_bytes(
        tflite_model
    )

    fp32_size = (
        model_path.stat().st_size
        / (1024 * 1024)
    )

    int8_size = (
        output_path.stat().st_size
        / (1024 * 1024)
    )

    reduction = (
        1.0
        - int8_size / fp32_size
    ) * 100.0

    print("\n" + "=" * 70)
    print("CUANTIZACIÓN COMPLETADA")
    print("=" * 70)

    print(
        f"FP32 : {fp32_size:.3f} MB"
    )

    print(
        f"INT8 : {int8_size:.3f} MB"
    )

    print(
        f"Reducción: {reduction:.2f}%"
    )

    print(
        f"\nArchivo:\n{output_path}"
    )

    print_tflite_details(
        output_path
    )


if __name__ == "__main__":
    main()