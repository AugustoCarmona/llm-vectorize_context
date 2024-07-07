import chromadb
from chromadb.utils import embedding_functions
from chromadb.db.base import UniqueConstraintError
import json
import openai

from car_data_etl import prepare_car_reviews_data
from chroma_utils import build_chroma_collection

DATA_PATH = "data/archive/*"
CHROMA_PATH = "car_review_embeddings"
EMBEDDING_FUNC_NAME = "multi-qa-MiniLM-L6-cos-v1"
COLLECTION_NAME = "car_reviews"

def make_question(question):
    client = chromadb.PersistentClient(CHROMA_PATH)
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_FUNC_NAME
    )

    collection = client.get_collection(
        name=COLLECTION_NAME, embedding_function=embedding_func
    )

    context = """
    You are a customer success employee at a large
    car dealership. Use the following car reviews
    to answer questions: {}
    """

    good_reviews = collection.query(
        query_texts=[question],
        n_results=10,
        include=["documents"],
        where={"Rating": {"$gte": 3}},
    )

    reviews_str = ",".join(good_reviews["documents"][0])

    good_review_summaries = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": context.format(reviews_str)},
            {"role": "user", "content": question},
        ],
        temperature=0,
        n=1,
    )

    print(good_review_summaries["choices"][0]["message"]["content"])

def main():
    try:
        # como parametro opcional se puede pasar una lista con los anos que se desee agregar a la coleccion [2016,2017,2018]
        # por defecto solo se incluye el ano 2017
        chroma_car_reviews_dict = prepare_car_reviews_data(DATA_PATH)
        build_chroma_collection(
            CHROMA_PATH,
            COLLECTION_NAME,
            EMBEDDING_FUNC_NAME,
            chroma_car_reviews_dict["ids"],
            chroma_car_reviews_dict["documents"],
            chroma_car_reviews_dict["metadatas"]
        )
        print("Colección creada")
    except UniqueConstraintError:
        print(f"Warning: La colección {COLLECTION_NAME} ya existe.")

    # secret key
    with open("config.json", mode="r") as json_file:
        config_data = json.load(json_file)
    openai.api_key = config_data.get("openai-secret-key")

    while True:
        question = input("\nAsk me: ")
        make_question(question + "\n")

if __name__ == "__main__":
    main()