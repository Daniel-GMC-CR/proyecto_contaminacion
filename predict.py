import cv2
import numpy as np


#----------------------------------
# Umbral calibrado manualmente
# (más sensible para detectar arroz)
#----------------------------------

umbral=15



def preprocesar(ruta):

    img=cv2.imread(
        ruta,
        cv2.IMREAD_GRAYSCALE
    )

    if img is None:
        raise ValueError(
        "No se encontró prueba.jpg"
        )


    img=cv2.resize(
        img,
        (128,128)
    )


    #-------------------------
    # mismo pipeline
    #-------------------------

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



def contar_componentes(img):

    mascara=(
        img==0
    ).astype(
        np.uint8
    )


    num_labels,labels,stats,_=(
        cv2.connectedComponentsWithStats(
            mascara,
            connectivity=8
        )
    )


    contador=0


    for i in range(
        1,
        num_labels
    ):

        area=stats[
            i,
            cv2.CC_STAT_AREA
        ]


        if area>=4:
            contador+=1


    return contador



#-----------------------------
# prueba
#-----------------------------

img=preprocesar(
    "prueba.jpg"
)


# guarda para inspeccionar
cv2.imwrite(
    "prueba_procesada_debug.jpg",
    img
)


componentes=contar_componentes(
    img
)


print(
"Componentes detectados:",
componentes
)

print(
"Umbral:",
umbral
)



if componentes>=umbral:

    print(
        "Hay arroz"
    )

else:

    print(
        "No hay arroz"
    )