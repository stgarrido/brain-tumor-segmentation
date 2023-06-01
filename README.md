## Introducción

El objetivo de este proyecto es el procesamiento de imágenes de resonancia magnética (MRI) cerebral en formato NIFTI para la identificación de tumor, a través de un algoritmo desarrollado en Python. Los paquetes utilizados fueron:

- NiBabel
- OpenCV
- NumPy
- Matplotlib

Las técnicas aplicadas fueron:

- K-means clustering
- Erosión y dilatación de imagen
- Detección de bordes Canny

## Desarrollo

Previamente se realiza la elección del slice a trabajar con el software BrainVISA, eligiendo el slice donde el tumor se visualiza con el mayor diámetro posible. A continuación se carga la imagen con NiBabel, se selecciona el slice y con el método k-means se agrupa en 3 grupos según la intensidad de voxels: Fondo negro, materia gris con materia blanca y tumor con corteza cerebral. Luego se crea una máscara binaria aplicando umbralización para aislar el tumor con la corteza cerebral del resto de la imagen. Seguidamente se aplica un proceso de erosión para eliminar la parte de la corteza de la máscara y de dilatación para devolver el tumor a su tamaño original, y se aplica el algoritmo de detección de bordes para identificar el tumor. Finalmente se aplica la máscara al slice original obtenido un slice con el tumor aislado, se calcula la intensidad de voxel promedio que corresponde al tejido del tumor y se genera una nueva imagen NIFTI que conserve solo el tumor.

## Instalación

Es necesario tener instalado Python en su dispositivo. Se debe preparar un entorno virtual con los paquetes necesarios su funcionamiento:

```bash
# Instalación de virtualenv (solo si no lo tiene)
pip install virtualenv
# Creación de entorno virtual
virtualenv env
```

La activación del entorno virtual depende del sistema operativo que se esté utilizando:

```bash
# Comando Linux
source env/Scripts/activate
```
```bash
# Comando Windows
source env/bin/activate
```

Finalmente se deben instalar los paquetes a utilizar:

```bash
pip install -r requirements.txt
```

Para ejecutar el proyecto:

```bash
# Comando Linux
python3 main.py
```
```bash
# Comando Windows
py main.py
```

## Resultados

Se adjuntan imágenes de los resultados obtenidos en las distintas etapas del proyecto

### Imagen agrupada k = 3

<img alt="Etiqueta kmeans" src="results/Etiqueta_kmeans_case_014_2.jpg?raw=true" name="kmeans" width="320"></img>

### Máscara binaria y máscara tumor

<img alt="Máscara binaria" src="results/Mascara_binaria_case_014_2.jpg?raw=true" name="binaryMask" width="320"></img> <img alt="Máscara tumor" src="results/Mascara_tumor_case_014_2.jpg?raw=true" name="tumorMask" width="320"></img>

### Borde del tumor

<img alt="Borde del tumor" src="results/Borde_tumor_case_014_2.jpg?raw=true" name="tumorBorder" width="320"></img>

### Tumor con detección de borde

<img alt="Tumor delineado" src="results/Tumor_delineado_case_014_2.jpg?raw=true" name="sliceTumorBorder" width="320"></img>