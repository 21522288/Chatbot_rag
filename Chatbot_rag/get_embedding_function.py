__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
# from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
# from dotenv import load_dotenv
# import os
# load_dotenv()


def get_embedding_function():
   # embeddings = HuggingFaceInferenceAPIEmbeddings(
   #  api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN"), model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
   # )
   embeddings = FastEmbedEmbeddings()
   return embeddings