from flask import Flask, request
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


app = Flask(__name__)
model = HuggingFaceEmbeddings(model_name = "mixedbread-ai/mxbai-embed-large-v1")

def generate_emb(name):
    splitter = RecursiveCharacterTextSplitter(chunk_size = 400, chunk_overlap = 50)
    list = os.listdir(name)
    for el in list:
        with open(name+'\\'+el, 'r') as input_file:
            data = input_file.read()
    
            chunks = splitter.split_text(data)
            meta = []
            for ch in chunks:
             meta.append({"name":name,"url": el, "id": ch.index})
            save(chunks, meta)

def save(chunks, meta):
    vector_store = FAISS.from_texts(chunks, model, meta)
    vector_store.save_local("local")

@app.route('/query', methods=['GET'])
def find():
    query = request.args.get('text')
    vector_store = FAISS.load_local("local",model,allow_dangerous_deserialization=True)
    ret = vector_store.as_retriever(k=3, lambda_mult= 0.5)  
    res = ret.invoke(query)

    return res

if __name__ == '__main__':
    load_dotenv()
    generate_emb('task_2\knowledge_base')
 
    app.run(host="0.0.0.0", port=8082, debug=True)
