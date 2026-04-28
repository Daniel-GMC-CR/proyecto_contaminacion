import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#---------------------------------
# crear carpeta modelos si no existe
#---------------------------------

os.makedirs(
    "models",
    exist_ok=True
)

#---------------------------------
# cargar dataset
#---------------------------------

df = pd.read_csv(
    "dataset/dataset.csv"
)

X = df.drop(
    columns=["label"]
)

y = df["label"]

#---------------------------------
# dividir entrenamiento/prueba
#---------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

#---------------------------------
# modelo SVM
#---------------------------------

modelo = SVC(
    kernel="linear"
)

modelo.fit(
    X_train,
    y_train
)

#---------------------------------
# evaluación
#---------------------------------

pred = modelo.predict(
    X_test
)

print("\nReporte:\n")

print(
    classification_report(
        y_test,
        pred,
        zero_division=0
    )
)

print(
    "Matriz de confusión:"
)

print(
    confusion_matrix(
        y_test,
        pred
    )
)

#---------------------------------
# exportar modelo
#---------------------------------

joblib.dump(
    modelo,
    "models/C24478_Daniel_Madriz.joblib"
)

print(
"\nModelo guardado correctamente"
)