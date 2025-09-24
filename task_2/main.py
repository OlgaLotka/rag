from flask import Flask, request, jsonify
import requests
from bs4 import BeautifulSoup
import json 
import os
import logging
import schedule
from crontab import CronTab

def openJson(name):
    with open(name, 'r') as input_file:
        return json.load(input_file)
    
urls = openJson('task_2\\file_map.json')

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, filename="py_log.log",filemode="w")

app = Flask(__name__)


def parse_user_datafile_bs(text, rename):

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
    return cleaned_text2 

@app.route('/scrapy', methods=['GET'])
def scrapy():
    rename = openJson('task_2\\terms_map.json')
        
    headers = {
            'User-Agent': 'PostmanRuntime/7.45.0'
        }
    for k,v in urls.items():
        r = requests.get(v, headers)
        filename = 'task_2\knowledge_base\\' + k + '.txt'
        with open(filename, 'w', encoding="utf-8") as output_file:
            print(filename)
            t = parse_user_datafile_bs(r.text, rename)
            output_file.write(t)

    return rename

def check():
    cpt = sum([len(files) for r, d, files in os.walk("task_2\knowledge_base")])
    if (len(urls) != cpt):
        logger.info('new files is present')
        #scrapy()
    return cpt

@app.route('/check', methods=['GET'])
def check22():
    count = check()
    return jsonify(count)


if __name__ == '__main__':
    schedule.every().minute.do(check)
    app.run(host="0.0.0.0", port=8084, debug=True)
    
    while True:
        schedule.run_pending()

    