from flask import Flask, request, jsonify
import requests
from bs4 import BeautifulSoup
import json, os, logging, datetime
import schedule
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

def openJson(name):
    with open(name, 'r') as input_file:
        return json.load(input_file)
    
urls = openJson('task_2\\file_map.json')

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, filename="generate_files_log.log",filemode="w", format="%(asctime)s %(levelname)s %(message)s")

model_name = "mixedbread-ai/mxbai-embed-large-v1"

model = HuggingFaceEmbeddings(model_name = model_name)

app = Flask(__name__)

def generate_emb(name):
    logger.info(f"Generate Embeddings")
    splitter = RecursiveCharacterTextSplitter(chunk_size = 400, chunk_overlap = 50)
    all_chunks = []
    list = os.listdir(name)
    for el in list:
        try:
            with open(name+'\\'+el, 'r') as input_file:
                data = input_file.read()
                logger.info(f"Split file {el}")
                chunks = splitter.split_text(data)
                logger.info(f"Chunk count = {len(chunks)} ")
                meta = []
                for ch in chunks:
                    logger.info(f"add meta: id={ch.index}, name={name} ")
                    meta.append({"name":name,"url": el, "id": ch.index})
                all_chunks.extend(chunks)    
            save(all_chunks, meta)
        except Exception as e:
            logger.error(e)        

def save(chunks, meta):
    vector_store = FAISS.from_texts(chunks, model, meta)
    logger.info(f"index updated at {datetime.datetime.now}, {len(chunks)} files added, 0 errors")
    vector_store.save_local("local")

def parse_user_datafile_bs(text, rename):
    logger.info(f"Start renaming")
    soup = BeautifulSoup(text, 'html.parser')
    
    text1 = soup.get_text(separator="", strip=False)
    text1.join(e for e in text1 if e.isalnum())
    cleaned_text = "".join([char for char in text1 if char != "\n"])
    cleaned_text2 = "".join([char for char in cleaned_text if char != "\t"])
    #replace_text = cleaned_text2.translate(str.maketrans(rename))
    for k,v in rename.items():
        cleaned_text2 = cleaned_text2.replace(k,v)
        #name = soup.find('span', {'class': 'mw-page-title-main'}).next.replace(k,v)
        #role = soup.find('p', {'class': 'pull-quote__text'}).next.replace(k,v)
#.find('div', {'class': 'mw-body-content'}

       # script = soup.find("script", text=re.compile('<p>'))
        #data = json.loads(re.search("data\s+=\s+(\[.*?\])", script).group(1))
        #role2 = body.next.replace(k,v)
    #text1 = [item.strip() for item in text1 if str(item)]  
    logger.info(f"Renaming is succsses")
    return cleaned_text2 

@app.route('/scrapy', methods=['GET'])
def scrapy():
    logger.info(f"Start scraping")
    rename = openJson('task_2\\terms_map.json')
        
    headers = {
            'User-Agent': 'PostmanRuntime/7.45.0'
        }
    for k,v in urls.items():
        try:
            logger.info(f"Load by url {v}")
            r = requests.get(v, headers)
            filename = 'task_2\knowledge_base\\' + k + '.txt'
            with open(filename, 'w', encoding="utf-8") as output_file:
                logger.info(f"Start generate {filename}")
                t = parse_user_datafile_bs(r.text, rename)
                logger.info(f"Parsing is succsses")
                output_file.write(t)
                logger.info(f"Save is succsses")
            return rename    
        except Exception as e:
            logger.error(e)        
    

def check()->bool:
    logger.info(f"Find a new item in file_map")
    cpt = sum([len(files) for r, d, files in os.walk("task_2\knowledge_base")])
    url2 = openJson('task_2\\file_map.json')
    return len(url2) != cpt


@app.route('/check', methods=['GET'])
def check22():

    if check():
        logger.info('new files is present')
        scrapy()
        generate_emb('task_2\knowledge_base')
    return 200


if __name__ == '__main__':

    #check()
    app.run(host="0.0.0.0", port=8084, debug=True)


    