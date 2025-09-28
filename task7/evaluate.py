from flask import Flask, request
import os, random, re, datetime, json 
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import PromptTemplate
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext

from langchain_huggingface import HuggingFacePipeline

log_name = "logs.jsonl"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, filename="logs.jsonl",filemode="w", format="%(asctime)s %(levelname)s %(message)s")
# 
model_name = "mixedbread-ai/mxbai-embed-large-v1"
model = HuggingFaceEmbeddings(model_name = model_name)

def openJson(name):
    with open(name, 'r') as input_file:
        return json.load(input_file)
    
golden_questions = openJson('task7\\golden_questions.json')

def getMetadata(docs):
    comb_text = ""
    for doc in docs:
        for k, v in doc.metadata.items():
            if k == "url":
                comb_text += "<a href=" +v+">"
    return comb_text

def process_llm_response(llm_response):  
    # Заменить символы новой строки с пользовательскими заполнителями  
    processed_response = llm_response["result"].replace("\n", "")[:1500]
    # Отправить обработанный ответ  
    return processed_response.join(getMetadata(llm_response["source_documents"]))  

def find(question: str):
        vector_store = FAISS.load_local("local",model,allow_dangerous_deserialization=True)
        ret = vector_store.as_retriever(k=3, lambda_mult= 0.5)  
        #docs = ret.invoke(question)

        llm = HuggingFacePipeline.from_model_id(
            model_id="gpt2",
            task="text-generation"
        )
        template = "System: {system3}\nContext: {context}\nQuestion: {question}\nAnswer:"
    #  prompt = prompt_template.format_prompt(query= [HumanMessage(content=query)]).to_string
        QA_CHAIN_PROMPT = PromptTemplate(input_variables=["system3", "context", "question"], template=template, validate_template=True)
        formatted_prompt = QA_CHAIN_PROMPT.format(system3="You are a helpful assistant ", context="Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.", question=question)

        qa_chain = RetrievalQA.from_llm(
        llm=llm,
        verbose=True,
        retriever=ret,
        #prompt = formatted_prompt,
        #chain_type="stuff",
        return_source_documents=True
        #chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )
        llm_response = qa_chain.invoke({"query":formatted_prompt}) #qa_chain.invoke({"system":"You are a helpful assistant ", "context":"Provide full answer", "question":query1, "query":query1})
        
        #logger.info(f"Response {res} ")
        return llm_response

def compare(res: str, answers: str)->bool:
    text = res["result"]
    index = text.find("Answer:")
    print(re.search("Answer:", text))
    print(text[index::])
    array = answers.split(",")
    b1 = any(re.search(p, text, re.IGNORECASE) for p in array)

    #b2 = any(re.search(p, array, re.IGNORECASE) for p in text)
    return b1 #& b2

if __name__ == '__main__':

    for q,a in golden_questions.items():
        logger.info(f"Question {q}")
        res = find(q)
        logger.info(f"Response lenght {len(res)}")
        res1 = process_llm_response(res)

        logger.info(f"Response {res1} ")
        docs = res["source_documents"]
        logger.info(f"Doc count {len(docs)}")
        meta = getMetadata(docs)
        logger.info(f"Doc count {meta}")
        b = compare(res, a)
        logger.info(f"Answer is correct: {b}")

