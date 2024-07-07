import pathlib
import chromadb
from chromadb.utils import embedding_functions
from more_itertools import batched

def build_chroma_collection(
    chroma_path: pathlib.Path,
    collection_name: str,
    embedding_func_name: str,
    ids: list[str],
    documents: list[str],
    metadatas: list[dict],
    distance_func_name: str = "cosine",
):
    """
    Crea una colección en ChromaDB utilizando un modelo de embedding específico para indexar documentos.

    Args:
        chroma_path (pathlib.Path): Ruta donde se almacenará ChromaDB.
        collection_name (str): Nombre de la colección a crear.
        embedding_func_name (str): Nombre del modelo de embedding a utilizar para convertir documentos en vectores.
        ids (list[str]): Lista de identificadores únicos para cada documento.
        documents (list[str]): Lista de documentos a indexar.
        metadatas (list[dict]): Lista de metadatos asociados a cada documento.
        distance_func_name (str, opcional): Función de distancia para calcular similitudes entre documentos.

    Returns:
        None

    Raises:
        ChromaDBError: Si hay un problema al interactuar con ChromaDB.
    """
    chroma_client = chromadb.PersistentClient(chroma_path)

    # modelo que transformara los documentos en vectores
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=embedding_func_name
    )

    # para calcular la distancia entre vectores se utilizara distancia de coseno (valor por defecto)
    # al representar cada vector de embedding una review de cliente, cuanto menor sea la distancia
    # sinoidal entre dos vectores, mayor sera la relacion entre los documentos, por lo que el contexto
    # sera relevante
    collection = chroma_client.create_collection(
        name=collection_name,
        embedding_function=embedding_func,
        metadata={"hnsw:space": distance_func_name},
    )

    document_indices = list(range(len(documents)))

    # itera los indices de los documentos en lotes de 166 elementos
    # agregando los documentos del lote actual a la coleccion
    for batch in batched(document_indices, 166):
        start_idx = batch[0]
        end_idx = batch[-1]

        collection.add(
            ids=ids[start_idx:end_idx],
            documents=documents[start_idx:end_idx],
            metadatas=metadatas[start_idx:end_idx],
        )