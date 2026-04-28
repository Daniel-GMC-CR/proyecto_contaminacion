import os
import numpy as np
import pandas as pd
from PIL import Image

# ----------------------------
# Rutas
# ----------------------------

RAW_DIR = "Fotos sin procesar"
PROC_DIR = "Fotos procesadas"
CSV_DIR = "dataset"

os.makedirs(CSV_DIR,exist_ok=True)

categorias = {
    "Arroz":1,
    "Clip":0
}

datos=[]

# ------------------------------------
# Procesamiento
# ------------------------------------

for clase,label in categorias.items():

    entrada=os.path.join(RAW_DIR,clase)
    salida=os.path.join(PROC_DIR,clase)

    os.makedirs(salida,exist_ok=True)

    for archivo in os.listdir(entrada):

        ruta=os.path.join(entrada,archivo)

        try:

            img=Image.open(ruta)

            # escala de grises
            img=img.convert("L")

            # tamaño requerido
            img=img.resize((128,128))

            # binarización
            umbral=200

            img_binaria=img.point(
                lambda p:255 if p>umbral else 0
            )

            # guardar procesada
            img_binaria.save(
                os.path.join(salida,archivo)
            )

            # matriz 1 y 0
            matriz=np.array(img_binaria)

            matriz=np.where(
                matriz==255,
                1,
                0
            )

            vector=matriz.flatten()

            fila=list(vector)
            fila.append(label)

            datos.append(fila)

        except Exception as e:
            print(
                f"Error en {archivo}: {e}"
            )


# --------------------------------
# CSV final
# --------------------------------

columnas=[
f'pixel_{i}'
for i in range(16384)
]

columnas.append("label")

df=pd.DataFrame(
    datos,
    columns=columnas
)

df.to_csv(
    "dataset/dataset.csv",
    index=False
)

print(
"Dataset creado correctamente"
)