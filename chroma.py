from langchain.embeddings import HuggingFaceEmbeddings
from chromadb.config import Settings
import chromadb
import pdfplumber
import os

PDF_FOLDER = "/home/alfonsocalero/Documentos/IA/WAS/ai-in-browser/docs"

chroma_client = chromadb.Client(Settings(chroma_server_host="localhost", chroma_server_http_port=8000))
collection = chroma_client.create_collection(name="my_collection")

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

collection_name = "mi_coleccion_pdf"

def extract_text_from_pdf(pdf_path):
    """Extrae texto de un PDF usando pdfplumber."""
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def process_and_store_pdfs(folder_path):
    """Procesa todos los PDFs de la carpeta y almacena los embeddings en ChromaDB."""
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, file_name)
            print(f"Procesando {pdf_path}...")
            
            text = extract_text_from_pdf(pdf_path)
            
            if text.strip():
                embeddings = embedding_model.embed_documents([text])
                
                collection.add(
                    documents=[text],
                    metadatas=[{"file_name": file_name}],
                    ids=[file_name]
                )
                print(f"Texto del archivo {file_name} almacenado en la base de datos.")
            else:
                print(f"No se encontró texto en {file_name}.")

process_and_store_pdfs(PDF_FOLDER)

# results = collection.query(
#     query_texts=["y recepción del mensaje de datos y de documentos electrónicos transferibles, emitir certificados en relación"],
#     n_results=2
# )
# print(results)
