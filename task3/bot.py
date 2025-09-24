from flask import Flask, request
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext

from langchain_huggingface import HuggingFacePipeline

#app = Flask(__name__)

#model_name = "sentence-transformers/all-mpnet-base-v2"


model_name = "mixedbread-ai/mxbai-embed-large-v1"

model = HuggingFaceEmbeddings(model_name = model_name)
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

logger = logging.getLogger(__name__)
#logging.basicConfig(level=logging.INFO, filename="bot_log.log",filemode="w")

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


#@app.route('/query', methods=['GET'])
async def find(update: Update, context: CallbackContext):
    wait_anw = await update.message.reply_text("Wait a minute! Thinking...")
    await context.bot.send_chat_action(action = "typing", chat_id = update.message.chat_id)
    #query = request.args.get('text')
    query = update.message.text
    vector_store = FAISS.load_local("local",model,allow_dangerous_deserialization=True)
    ret = vector_store.as_retriever(k=3, lambda_mult= 0.5)  
    docs = ret.invoke(query)

    comb_text = "\n".join(doc.page_content for doc in docs) 

    llm = HuggingFacePipeline.from_model_id(
        model_id="gpt2",
        task="text-generation"
    )
    prompt_template = ChatPromptTemplate([
        ("system", "You are a helpful assistant"),
        MessagesPlaceholder("query")
    ])

  #  prompt = prompt_template.format_prompt(query= [HumanMessage(content=query)]).to_string
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context"], template=" Context: {context} User")

    
    qa_chain = RetrievalQA.from_llm(
    llm=llm,
    verbose=True,
    retriever=ret,
    prompt = QA_CHAIN_PROMPT
    #type = 
    #chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    res = qa_chain.invoke({"System": "You are a helpful assistant ","context": "Only name return", "query": query})
    await update.message.reply_text(res["result"], parse_mode="HTML")

async def start(update: Update, context: CallbackContext):
    await update.message.reply_text("Hello, My name is The Oracle")

async def unknown_handler(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text("Unknown commnd")

async def error_handler(update: Update, context: CallbackContext) -> None:
    logger.error(context.error)
    await update.message.reply_text("Error")

def main() -> None:
    telApp = Application.builder().token("8476077399:AAFk41SL-_kKoMuFiq0ob-wLk8bsWhfb1_Y").build()
    telApp.add_handler(CommandHandler("start",start))
    telApp.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, find))
    telApp.add_handler(MessageHandler(filters.COMMAND, unknown_handler))
    telApp.add_error_handler(error_handler)
    telApp.run_polling(timeout=10)



if __name__ == '__main__':

    #generate_emb('task_2\knowledge_base')
    try:
        main()
    except Exception as e:
        logger.error(e)
    #app.run(host="0.0.0.0", port=8082, debug=True)
