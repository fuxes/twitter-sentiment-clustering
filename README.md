# Server side
No estan subidos los datos, son archivos muy grandes.

preprocessing.py es lo que hace todo el trabajo
Es un pipeline de varias etapas en las que levanta los datos, los vectoriza, clusteriza y genera nuevos archivos para que sean visualizados por el cliente.

## Preguntas
- Como modularizar el pipeline para q sea reusable?
- Como ocultar las keys de github?
- Como enviar las visualizaciones al cliente? (Son archivos de ~100mbs)
- Como se estructurarian las carpetas para el caso de que los workers esten hechos en python y el server en nodejs? (Vale la pena?)
