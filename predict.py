import joblib
import numpy as np

from PIL import Image


modelo=joblib.load(
"models/C24478_Daniel_Madriz.joblib"
)


def preprocesar(ruta):

    img=Image.open(ruta)

    img=img.convert("L")

    img=img.resize(
        (128,128)
    )

    img=img.point(
        lambda p:
        255 if p>200 else 0
    )

    x=np.array(img)

    x=np.where(
        x==255,
        1,
        0
    )

    return x.flatten().reshape(1,-1)


ruta="prueba.jpg"

x=preprocesar(ruta)

pred=modelo.predict(x)[0]

if pred==1:
    print("Hay arroz")
else:
    print("No hay arroz")