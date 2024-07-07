import chromadb
from chromadb.utils import embedding_functions
from car_data_etl import prepare_car_reviews_data
from chroma_utils import build_chroma_collection

DATA_PATH = "data/archive/*"
# DATA_PATH = "data/archive/*.csv"
CHROMA_PATH = "car_review_embeddings"
EMBEDDING_FUNC_NAME = "multi-qa-MiniLM-L6-cos-v1"
COLLECTION_NAME = "car_reviews"

def eliminar_coleccion_si_existe(nombre_coleccion):
    try:
        db = chromadb.PersistentClient(CHROMA_PATH)
        db.delete_collection(nombre_coleccion)
        print(f"La colección '{nombre_coleccion}' ha sido eliminada correctamente.")
    except Exception as e:
        print(f"Error al intentar eliminar la colección '{nombre_coleccion}': {e}")

def make_query():
    client = chromadb.PersistentClient(CHROMA_PATH)
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_FUNC_NAME
        )
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_func)

    great_reviews = collection.query(
        query_texts=["Find me some positive reviews that discuss the car's performance"],
        n_results=5,
        include=["documents", "distances", "metadatas"]
    )

    print(great_reviews["documents"][0][0])

def main():
    
    
    eliminar_coleccion_si_existe(COLLECTION_NAME)

    chroma_car_reviews_dict = prepare_car_reviews_data(DATA_PATH)

    print("Data consolidada")


    build_chroma_collection(
        CHROMA_PATH,
        COLLECTION_NAME,
        EMBEDDING_FUNC_NAME,
        chroma_car_reviews_dict["ids"],
        chroma_car_reviews_dict["documents"],
        chroma_car_reviews_dict["metadatas"]
    )

    print("Coleccion creada")

    make_query()

if __name__ == "__main__":
    main()