from flask import Flask, request
import requests
from bs4 import BeautifulSoup
import json 


urls = {
    "Neo":"https://matrix.fandom.com/wiki/Neo",
    "Trinity":"https://matrix.fandom.com/wiki/Trinity"
    }

app = Flask(__name__)

def parse_user_datafile_bs(text,k, rn):
    results = []
    soup = BeautifulSoup(text)
    name = soup.find('span', {'class': 'mw-page-title-main'}).next.replace(k,rn)
    role = soup.find('p', {'class': 'pull-quote__text'}).next.replace(k,rn)
      
    results.append({
                'name': name,
                'role': role
            })
    return results

@app.route('/scrapy', methods=['GET'])
def scrapy():
    with open('task_2\\terms_map.json', 'r') as input_file:
        rename = json.load(input_file)
        
    headers = {
            'User-Agent': 'PostmanRuntime/7.45.0'
        }
    for k,v in urls.items():
        rn = rename.get(k)
        r = requests.get(v, headers)
        filename = 'task_2\knowledge_base\\' + k + '.txt'
        with open(filename, 'w', encoding="utf-8") as output_file:
            t = parse_user_datafile_bs(r.text,k, rn)
            output_file.write(str(t))

    return rename 

if __name__ == '__main__':

    app.run(host="0.0.0.0", port=8084, debug=True)