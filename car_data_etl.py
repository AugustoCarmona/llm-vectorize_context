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
