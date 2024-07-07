import pathlib
import polars as pl

def prepare_car_reviews_data(data_path: pathlib.Path, vehicle_years: list[int] = [2017]):
    """Prepara los datos del conjunto de revisiones de autos para su indexación en ChromaDB.

    Args:
        data_path (pathlib.Path): Ruta al archivo de datos de revisiones de autos.
        vehicle_years (list[int], optional): Lista de años de vehículos a incluir. 
            Por defecto, se incluye el ano 2017 para eficientizar la creacion de la coleccion a efectos practicos.

    Returns:
        dict: Un diccionario con las claves 'ids', 'documents' y 'metadatas'. 
            'ids' es una lista de identificadores únicos para cada revisión.
            'documents' es una lista de textos de revisión.
            'metadatas' es una lista de diccionarios con metadatos asociados a cada revisión.
    """

    # define un esquema para asegurar la interpretacion de los datatypes del conjunto
    # y luego escanea el conjunto de reviews aplicando dicho esquema
    dtypes = {
        "": pl.Int64,
        "Review_Date": pl.Utf8,
        "Author_Name": pl.Utf8,
        "Vehicle_Title": pl.Utf8,
        "Review_Title": pl.Utf8,
        "Review": pl.Utf8,
        "Rating": pl.Float64,
    }
    car_reviews = pl.scan_csv(data_path, dtypes=dtypes)

    # extrae vehiculo titulo y ano como nuevas columnas filtrando por los anos seleccionados
    car_review_db_data = (
        car_reviews.with_columns(
            [
                (pl.col("Vehicle_Title").str.split(by=" ").list.get(0).cast(pl.Int64)).alias("Vehicle_Year"),
                (pl.col("Vehicle_Title").str.split(by=" ").list.get(1)).alias("Vehicle_Model"),
            ]
        )
        .filter(pl.col("Vehicle_Year").is_in(vehicle_years))
        .select(["Review_Title", "Review", "Rating", "Vehicle_Year", "Vehicle_Model"])
        .sort(["Vehicle_Model", "Rating"])
        .collect()
    )

    # da el formato esperado por chromadb para las claves de id, documentos y metadata
    ids = [f"review{i}" for i in range(car_review_db_data.shape[0])]
    documents = car_review_db_data["Review"].to_list()
    metadatas = car_review_db_data.drop("Review").to_dicts()

    return {"ids": ids, "documents": documents, "metadatas": metadatas}

""" Ejemplo:
{
    'ids': ['review12', 'review13', 'review14'],
    'documents': [
        "Last hybrid I by from Kia. 2016 Kia Optima Hybrid Sedan EX 4dr Sedan (2.4L 4cyl gas/electric hybrid 6A). No so high-brid, 25mpg in the city 33 hwy on the 2016 ex hybrid. Taken it in twice stating poor mpg and was told it's just not broken in yet. I have 10k on it when is it going to break in?  Liars!!",
        "A hybrid with good styling. If you are wanting a hybrid, but don't want the ugly as dog poop Prius, this is a car for you to check out.",
        "Love the 2016 Kia Optima Hybrid (even more than th). They changed the interior in the 2017. They did improve the comfort of the seats/headrest, but for me, they went backwards in terms of the interior styling- very bland and generic."
    ],
    'metadatas': [
        {'Author': 'Ken', 'Rating': 5},
        {'Author': 'Tod Bowermaster', 'Rating': 5},
        {'Author': 'Nick', 'Rating': 3}
    ]
}

"""