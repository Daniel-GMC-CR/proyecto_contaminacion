# Detector de contaminación por arroz

Proyecto para detectar presencia de granos de arroz en una línea de producción simulada.

## Requisitos

Instalar dependencias:

```bash
pip install -r requirements.txt
```

## Estructura

Fotos sin procesar:
- Arroz (positivas)
- Clip (negativas)

Las imágenes se procesan a:

- Blanco y negro
- 128x128 pixeles
- vectores binarios
- dataset csv

## Generar dataset

Ejecutar:

```bash
python preprocess.py
```

Genera:

dataset/dataset.csv

## Entrenar modelo

```bash
python train_model.py
```

Genera:

models/C24478_Daniel_Madriz.joblib

## Probar inferencia

Editar en predict.py la ruta de la imagen.

Luego:

```bash
python predict.py
```

## Archivos importantes

preprocess.py:
procesamiento de imágenes.

train_model.py:
entrenamiento.

predict.py:
predicción.

## Autor

Daniel Gerardo Madriz Chavarría
Carné C24478