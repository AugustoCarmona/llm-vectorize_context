# Vectorized Context

> Nota: El proyecto es funcional, pero la documentación del mismo aún se encuentra en proceso. De momento, solo se ha definido el objetivo y alcance del mismo. Sin embargo, siguiendo los comentarios en el código, así como el artículo que dio origen a este repositorio: ['Embeddings and Vector Databases With ChromaDB'](https://realpython.com/chromadb-vector-database/#represent-data-as-vectors) es posible entender su lógica.
> Durante los próximos días terminaré de documentar este README.

![image](https://miro.medium.com/v2/resize:fit:793/0*RTW5byy6eH_eSWTP.png)

El siguiente repositorio es un chatbot de consola que funciona como aplicación técnica de los casos de uso presentados en el artículo ['Embeddings and Vector Databases With ChromaDB'](https://realpython.com/chromadb-vector-database/#represent-data-as-vectors) de realpython.com. Su objetivo es brindar un acceso sencillo al caso de uso explicado, demostrando cómo los conceptos teóricos del Álgebra Vectorial pueden emplearse para vectorizar documentos (en este caso, texto), integrándolos en bases de datos vectoriales como ChromaDB. Esto permite generar comparaciones semánticas que se utilizarán para proporcionar el contexto adecuado al modelo de LLM, GPT-3.5 Turbo de OpenAI, con el fin de obtener respuestas específicas según los datos con los que se alimente el modelo.

El dataset utilizado para generar las colecciones de ChromaDB se obtuvo de [Kaggle](https://www.kaggle.com/datasets/ankkur13/edmundsconsumer-car-ratings-and-reviews) y corresponde a un conjunto de datos que contiene la opinión del consumidor y la calificación en estrellas por fabricante/modelo/tipo de automóvil según el sitio de venta de autos usados edmunds.com. El objetivo de este chatbot es que, según la información de las reviews contenidas en el sitio, pueda resolver dudas específicas sobre modelos, marcas o tipos de vehículos, orientando sus respuestas al consumidor.

En caso de tener alguna sugerencia, o corrección para el proyecto, feel free to contribute.

---

### Requerimientos:

Como en cualquier proyecto, idealmente, ni bien se clone el repositorio es conveniente generar un entorno virtual en el cual se instalarán las dependencias contenidas en el archivo `requirements.txt`. Asimismo, es necesario generar un archivo `config.json`, el cual albergará la secret key utilizada para conectarse a la API de OpenAI.

El archivo `config.json` debe componerse de la siguiente forma:

```json
{
  "openai-secret-key": "tu-secret-key"
}
```

En caso de no contar con créditos disponibles de OpenAI, no hay problema, ya que el fin de este caso de estudio es mostrar cómo funcionan los embeddings para convertir información compleja, como palabras, imágenes o documentos, en vectores en un espacio multidimensional que facilite la comparación y el análisis de datos mediante el cálculo del seno. Dichas funciones se encuentran en los módulos `car_data_etl.py` y `chroma_utils.py` y se explican más en detalle en este README; sin embargo, sin créditos de OpenAI, el contexto generado no se podrá servir al LLM, por lo que la funcionalidad de chat no estará disponible.

---

## Modulos

#### car_data_etl

Este módulo proporciona una función para preparar datos de las revisiones de autos para su indexación en ChromaDB. La función prepare_car_reviews_data toma un archivo de datos de revisiones de autos y devuelve un diccionario que contiene los identificadores únicos, textos de revisión y metadatos asociados para cada revisión de autos de la siguiente forma:

```json
{
  "ids": ["review12", "review13", "review14"],
  "documents": [
    "Last hybrid I by from Kia. 2016 Kia Optima Hybrid Sedan EX 4dr Sedan (2.4L 4cyl gas/electric hybrid 6A). No so high-brid, 25mpg in the city 33 hwy on the 2016 ex hybrid. Taken it in twice stating poor mpg and was told it's just not broken in yet. I have 10k on it when is it going to break in?  Liars!!",
    "A hybrid with good styling. If you are wanting a hybrid, but don't want the ugly as dog poop Prius, this is a car for you to check out.",
    "Love the 2016 Kia Optima Hybrid (even more than th). They changed the interior in the 2017. They did improve the comfort of the seats/headrest, but for me, they went backwards in terms of the interior styling- very bland and generic."
  ],
  "metadatas": [
    { "Author": "Ken", "Rating": 5 },
    { "Author": "Tod Bowermaster", "Rating": 5 },
    { "Author": "Nick", "Rating": 3 }
  ]
}
```

lo que hace la función prepare_car_reviews_data es procesar un archivo CSV con datos de revisiones de autos, filtrar las revisiones según los años de vehículos especificados, y devolver los datos en un formato adecuado para la indexación en ChromaDB generando identificadores únicos, extrayendo y limpiando textos de revisiones asi como dandoles estructura.

#### chroma_utils

---

## Ejecución
