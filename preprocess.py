import os
import cv2
import numpy as np
import pandas as pd

RAW_DIR="Fotos sin procesar"
PROC_DIR="Fotos procesadas"
CSV_DIR="dataset"

os.makedirs(PROC_DIR,exist_ok=True)
os.makedirs(CSV_DIR,exist_ok=True)

categorias={
    "Arroz":1,
    "Clip":0
}

datos=[]



def vectorizar(img,label):

    matriz=np.where(
        img==255,
        1,
        0
    )

    fila=list(
        matriz.flatten()
    )

    fila.append(label)

    datos.append(fila)



def procesar(img):

    img=cv2.resize(
        img,
        (128,128)
    )


    fondo=cv2.GaussianBlur(
        img,
        (25,25),
        0
    )

    corregida=cv2.divide(
        img,
        fondo,
        scale=255
    )


    _,binaria=cv2.threshold(
        corregida,
        0,
        255,
        cv2.THRESH_BINARY+
        cv2.THRESH_OTSU
    )


    bordes=cv2.Canny(
        corregida,
        50,
        150
    )

    bordes=255-bordes


    combinada=np.minimum(
        binaria,
        bordes
    )


    # kernel 1x1 como usted decidió
    kernel=np.ones(
        (1,1),
        np.uint8
    )

    combinada=cv2.morphologyEx(
        combinada,
        cv2.MORPH_CLOSE,
        kernel
    )


    return combinada



for clase,label in categorias.items():

    entrada=os.path.join(
        RAW_DIR,
        clase
    )

    salida=os.path.join(
        PROC_DIR,
        clase
    )

    os.makedirs(
        salida,
        exist_ok=True
    )



    for archivo in os.listdir(entrada):

        ruta=os.path.join(
            entrada,
            archivo
        )

        img=cv2.imread(
            ruta,
            cv2.IMREAD_GRAYSCALE
        )

        if img is None:
            continue



        base=procesar(img)

        cv2.imwrite(
            os.path.join(
                salida,
                archivo
            ),
            base
        )


        # original
        vectorizar(
            base,
            label
        )


        # rotación 90
        rot1=np.rot90(base)

        vectorizar(
            rot1,
            label
        )


        # rotación 180
        rot2=np.rot90(
            base,
            2
        )

        vectorizar(
            rot2,
            label
        )


        # espejo horizontal
        flip=cv2.flip(
            base,
            1
        )

        vectorizar(
            flip,
            label
        )


        # pequeño desplazamiento
        M=np.float32([
            [1,0,3],
            [0,1,3]
        ])

        shift=cv2.warpAffine(
            base,
            M,
            (128,128),
            borderValue=255
        )

        vectorizar(
            shift,
            label
        )



columnas=[
f"pixel_{i}"
for i in range(16384)
]

columnas.append(
"label"
)

df=pd.DataFrame(
datos,
columns=columnas
)

df.to_csv(
"dataset/dataset.csv",
index=False
)

print(
"Dataset aumentado generado"
)