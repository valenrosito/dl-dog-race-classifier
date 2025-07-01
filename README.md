## Estructura de Directorios

- **_dog-images_**: Contiene las imágenes de las 70 razas de perros, organizadas en carpetas para _train_, _test_ y _validation_.
- **_test-images_**: Incluye las 10 imagenes, utilizadas para probar manualmente los modelos entrenados mediante la interfaz de _Gradio_.
- **_test-annotations_**: Este directorio contiene las anotaciones manuales realizadas sobre las 10 imagenes de **_test-images_** para poder evaluar al modelo
- **_development_**: Carpeta donde se encuentran los _Jupyter Notebooks_ correspondientes a las dos primeras etapas del trabajo, en los cuales se evaluan métricas y se visualizan matrices de confusión.
- **_gradio-interface_**: Aquí se encuentra el codigo y las distintas interfaces de Gradio desarrollada desde la ***etapa 2***, que permite probar los modelos _ResNet50_, _ResNet18_ y la _CNN custom_, hasta la ***etapa 4***.
