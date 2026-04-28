# Model Card

Nombre:
Detector arroz v1

Uso esperado:
detección de contaminación tipo arroz.

No usar para:
detección industrial real.

Datos:
30 imágenes manualmente recolectadas.

Etiquetado:
1 arroz
0 no arroz

Métricas:
accuracy
precision
recall

Limitaciones:
dataset pequeño
objetos pequeños
ruido visual

Reproducibilidad:

python preprocess.py

python train_model.py