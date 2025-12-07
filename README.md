# 🐶 Pipeline de Visión por Computadora para la Detección y Clasificación de Razas de Perros

Este proyecto implementa un pipeline completo de visión por computadora para la clasificación de razas de perros, entrenado para identificar 70 razas distintas utilizando modelos de Deep Learning optimizados. El sistema cuenta con una interfaz visual interactiva desarrollada en Gradio que admite carga múltiple de imágenes, facilitando la detección, clasificación y búsqueda por similitud de manera ágil y precisa.

## Ejecución del Proyecto

1. **Instalación de dependencias**
    - Descarga todas las librerías necesarias ejecutando en la terminal:
      ```
      pip install -r requirements.txt
      ```

2. **Ejecución de las primeras dos etapas y sus métricas**
    - Dirígete al directorio _development/_.
    - Ejecuta los archivos correspondientes a cada etapa:
      - *Etapa 1*: contiene la interfaz de **Gradio** y la métrica **NDCG@10**.
      - *Etapa 2*: incluye las métricas de **recall**, **precisión**, **F1** y las **matrices de confusión**.

3. **Interfaz Gradio de la etapa 2**
    - Para acceder a la interfaz que permite usar los tres modelos (_ResNet50_, _ResNet18_ y la _CNN personalizada_), ejecuta desde el directorio raíz:
      ```
      python gradio-interface/GradioStage2.py
      ```

4. **Interfaz Gradio de la etapa 3**
    - Esta etapa utiliza **YOLOv8** y **ResNet50** para detectar y predecir la raza de perros en imágenes. Para ejecutarla, corre:
      ```
      python gradio-interface/GradioStage3.py
      ```

5. **Ejecución de la etapa 4 - Comparación de modelos**
    - Para comparar el modelo **ResNet50** con su versión cuantificada en **FP16** (evaluando precisión, recall, F1 y tiempo de inferencia en 10 batches), ejecuta:
      ```
      python gradio-interface/GradioStage4-Metrics.py
      ```

6. **Generación de anotaciones en la etapa 4**
    - Para generar anotaciones para cada imagen en _test-images_, creando un directorio _resnet-annotations_ con las anotaciones en formatos **YOLOv5** y **COCO (JSON)**, corre:
      ```
      python gradio-interface/GradioStage4-SAnnotations.py
      ```

---

## Estructura de Directorios

- **_dog-images_**:  
  Contiene las imágenes de las *70 razas de perros*, organizadas en carpetas para _train_, _test_ y _validation_.
- **_test-images_**:  
  Incluye las *10 imágenes* utilizadas para probar manualmente los modelos entrenados mediante la interfaz de _Gradio_.
- **_test-annotations_**:  
  Contiene las *anotaciones manuales* realizadas sobre las 10 imágenes de **_test-images_** para evaluar el modelo.
- **_development_**:  
  Carpeta donde se encuentran los _Jupyter Notebooks_ de las dos primeras etapas, en los cuales se evalúan métricas y se visualizan matrices de confusión.
- **_gradio-interface_**:  
  Aquí se encuentra el *código* y las distintas interfaces de **Gradio** desarrolladas desde la _etapa 2_ (para probar los modelos _ResNet50_, _ResNet18_ y la _CNN custom_) hasta la _etapa 4_.
