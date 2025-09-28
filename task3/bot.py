from flask import Flask, request
import os, random, re, datetime
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


model_name = "mixedbread-ai/mxbai-embed-large-v1"

model = HuggingFaceEmbeddings(model_name = model_name)
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

logger = logging.getLogger(__name__)
#logging.basicConfig(level=logging.INFO, filename="bot_log.log",filemode="w", format="%(asctime)s %(levelname)s %(message)s")

def generate_emb(name):
    logger.info(f"Generate Embeddings")
    splitter = RecursiveCharacterTextSplitter(chunk_size = 400, chunk_overlap = 50)
    
    list = os.listdir(name)
    for el in list:
        with open(name+'\\'+el, 'r') as input_file:
            data = input_file.read()
            logger.info(f"Split file")
            chunks = splitter.split_text(data)
            meta = []
            for ch in chunks:
                logger.info(f"add meta")
                meta.append({"name":name,"url": el, "id": ch.index})
            save(chunks, meta)

def save(chunks, meta):
    vector_store = FAISS.from_texts(chunks, model, meta)
    logger.info(f"index updated at {datetime.datetime.now}, {len(chunks)} files added, 0 errors")
    vector_store.save_local("local")

def getMetadata(docs):
    comb_text = ""
    for doc in docs:
        for k, v in doc.metadata.items():
            if k == "url":
                comb_text += "<a href=" +v+">"
    return comb_text

def process_llm_response(llm_response):  
    # Заменить символы новой строки с пользовательскими заполнителями  
    index = llm_response["result"].find("Answer:")
    processed_response = llm_response["result"][index::].replace("\n", "")
    # Отправить обработанный ответ  
    return processed_response.join(getMetadata(llm_response["source_documents"]))  

def simple_question(question: str)->bool:
    array = [
        r"\b(hello|hi)\b",
        r"\b(what`s up)\b",
        r"\b(thank)\b",
    ]
    return any(re.search(p, question, re.IGNORECASE) for p in array)

async def find(update: Update, context: CallbackContext):

    query1 = update.message.text
    user_id = update.effective_user.id
    logger.info(f"new question {query1} from user {user_id}")
    if simple_question(query1):
        res = random.choice(["The problem is choice", "Can I help you?", "Let me tell about something"])
        await update.message.reply_text(res)
    else:
        wait_anw = await update.message.reply_text("Wait a minute! Thinking...")
        await context.bot.send_chat_action(action = "typing", chat_id = update.message.chat_id)
        vector_store = FAISS.load_local("local",model,allow_dangerous_deserialization=True)
        ret = vector_store.as_retriever(k=2, lambda_mult= 0.7)  
        #docs = ret.invoke(query1)
        #logger.info(f"Doc count {len(docs)} for user {user_id}")
        #comb_text = "\n".join(doc.page_content for doc in docs) 

        llm = HuggingFacePipeline.from_model_id(
            model_id="gpt2",
            task="text-generation"
        )
        template = "System: {system3}\nContext: {context}\nQuestion: {question}\nAnswer:"
    #  prompt = prompt_template.format_prompt(query= [HumanMessage(content=query)]).to_string
        QA_CHAIN_PROMPT = PromptTemplate(input_variables=["system3", "context", "question"], template=template, validate_template=True)
        formatted_prompt = QA_CHAIN_PROMPT.format(system3="You are a assistant ", context="Provide a short answer. Do not return context in answer", question=query1)

        qa_chain = RetrievalQA.from_llm(
        llm=llm,
        verbose=True,
        retriever=ret,
        #prompt = QA_CHAIN_PROMPT,
        #chain_type="stuff",
        return_source_documents=True
        #chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )
        res = qa_chain.invoke({"query":formatted_prompt}) #qa_chain.invoke({"system":"You are a helpful assistant ", "context":"Provide full answer", "question":query1, "query":query1})
       
        logger.info(f"Response {res} for user {user_id}")
        answer =process_llm_response(res)[::2000]
        await update.message.reply_text(answer, parse_mode="HTML")

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
