import os
import cv2
import numpy as np
import joblib

conteos_arroz=[]
conteos_clip=[]


def contar_componentes(img):

    # objetos negros
    mascara=(img==0).astype(
        np.uint8
    )

    num_labels,labels,stats,cent=(
        cv2.connectedComponentsWithStats(
            mascara,
            connectivity=8
        )
    )

    contador=0

    for i in range(1,num_labels):

        area=stats[
            i,
            cv2.CC_STAT_AREA
        ]

        # ignorar ruido diminuto
        if area>=4:
            contador+=1

    return contador



for clase,label in {
"Arroz":1,
"Clip":0
}.items():

    carpeta=os.path.join(
        "Fotos procesadas",
        clase
    )

    for archivo in os.listdir(
        carpeta
    ):

        img=cv2.imread(
            os.path.join(
                carpeta,
                archivo
            ),
            cv2.IMREAD_GRAYSCALE
        )

        c=contar_componentes(
            img
        )

        if label==1:
            conteos_arroz.append(c)
        else:
            conteos_clip.append(c)



umbral=(
np.mean(conteos_arroz)+
np.mean(conteos_clip)
)/2


joblib.dump(
{
"threshold":umbral
},
"models/C24478_Daniel_Madriz.joblib"
)


print(
"Promedio arroz:",
np.mean(conteos_arroz)
)

print(
"Promedio clip:",
np.mean(conteos_clip)
)

print(
"Umbral:",
umbral
)