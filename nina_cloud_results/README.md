# NINA Cloud Results

Esta carpeta contiene los archivos necesarios para reproducir los resultados del clasificador de nubes IQC de WaskiriSat.

## Estructura

### dataset/
Contiene el dataset final utilizado para entrenamiento, validación y prueba.

- Train: 2653 CLEAR, 2851 CLOUD
- Validation: 468 CLEAR, 530 CLOUD
- Test: 834 CLEAR, 574 CLOUD como tiles únicos
- Total: 7910 tiles únicos

En test, cada tile se encuentra representado mediante los perfiles A, B y C, por lo que existen 4224 archivos físicos.

### training/
Contiene los archivos relacionados con el entrenamiento de MobileNetV2.

- `scripts/`: script de entrenamiento
- `models/`: modelo entrenado
- `reports/`: métricas, reportes y gráficas
- `logs/`: registros de ejecución

Script principal:

`train_movil_v3_16ago.py`

Modelo final:

`mobilenetv2_best.keras`

La arquitectura utilizada corresponde a MobileNetV2 con `alpha = 0.75`.

### evaluation/fp32/
Contiene los resultados de evaluación del modelo antes de cuantización.

Incluye métricas globales y por perfiles de degradación A, B y C, además de matrices de confusión y reportes.

### quantization/
Contiene el proceso de cuantización del modelo.

Script principal:

`quantize_alpha075_fullinteger.py`

Modelo generado:

`mobilenetv2_alpha075_fullinteger_int8.tflite`

La cuantización utilizada corresponde a Full Integer INT8.

### evaluation/int8/
Contiene la evaluación del modelo cuantizado, incluyendo métricas por perfil A, B y C y resultados globales.

### configs/
Contiene los archivos de configuración utilizados para la generación del dataset y simulación del sensor PAG7936.

Archivos principales:

- `config.py`
- `scene_registry.json`

## Flujo de reproducción

El flujo utilizado para reproducir los resultados es:

1. Preparación del dataset final
2. Entrenamiento de MobileNetV2
3. Evaluación del modelo FP32
4. Cuantización Full Integer INT8
5. Evaluación del modelo INT8
6. Comparación y consolidación de resultados

## Dataset

El dataset fue generado a partir de productos Sentinel-2 Level-2A y procesado para aproximar las características del sensor PAG7936 RGB.

Las clases utilizadas son:

- CLEAR: cobertura nubosa menor al 30 %
- CLOUD: cobertura nubosa mayor al 70 %

Las muestras ambiguas fueron excluidas.

## Perfiles de degradación

Se utilizaron tres perfiles:

- A: menor nivel de degradación
- B: condición intermedia
- C: mayor nivel de degradación

El perfil B se utilizó en entrenamiento y validación, mientras que los perfiles A, B y C se utilizaron en test.

## Reproducibilidad

Los scripts, modelos, reportes y configuraciones incluidos en esta carpeta corresponden al flujo utilizado para reproducir los resultados reportados para el clasificador de nubes IQC.