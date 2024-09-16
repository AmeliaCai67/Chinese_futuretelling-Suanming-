from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask import Blueprint, send_from_directory
from openai import OpenAI
import os
import sqlite3
import logging
import traceback
import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

static_bp = Blueprint('static', __name__, static_folder='static')

# 定义一个字典存储不同模型对应的API_KEY和base_url
MODEL_CONFIG = {
    "glm-4": {
        "api_key": "YOUR_API_key",
        "base_url": "https://open.bigmodel.cn/api/paas/v4/"
    },
    "abab6.5s-chat": {
        "api_key": "YOUR_API_key",
        "base_url": "https://api.minimax.chat/v1"
    }
}

# 定义全局变量来存储当前选择的模型
current_model = "glm-4"  # 默认为GLM模型

def generate_prompt_template(question, client_prompt, history=None):
    try:
        # Integrate all rag_content into the prompt
        results = search_similar_records(question, r"csv\suanming_database.db")
        rag_content = str(results)  # Convert results to string for storage
        # Include conversation history
        if history:
            history_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
            prompt_template = f"""Use the following pieces of context to answer the question at the end.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.

            Previous conversation:
            {history_context}

            Additional context:
            {rag_content}

            Question: {question}
            Client instruction: {client_prompt}
            Helpful Answer:"""
        else:
            prompt_template = f"""Use the following pieces of context to answer the question at the end.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.

            {rag_content}

            Question: {question}
            Client instruction: {client_prompt}
            Helpful Answer:"""

    except Exception as e:
        logging.error(f"Error in generate_prompt_template: {str(e)}")
        prompt_template = f"""
        Question: {question}
        Client instruction: {client_prompt}
        Helpful Answer:
        """

    return rag_content, prompt_template

def search_similar_records(query, database_path, top_k=10):
    # 初始化模型
    model = SentenceTransformer('shibing624/text2vec-base-chinese')
    
    # 清除缓存
    try:
        torch.cuda.empty_cache()
        print("清空缓存成功")
    except Exception:
        print("清空缓存失败")
        
    # 对查询进行向量化
    query_vector = model.encode(query)
    
    # 连接数据库
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    
    results = {}
    
    # 获取所有表格
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    
    for table in tables:
        table_name = table[0]
        cursor.execute(f"SELECT chapter_id, segment_id, content, keywords, length, keywords_embeddings FROM {table_name}")
        records = cursor.fetchall()
        
        similarities = []
        for record in records:
            keywords_vector = np.frombuffer(record[5], dtype=np.float32)
            similarity = cosine_similarity([query_vector], [keywords_vector])[0][0]
            similarities.append((similarity, record))
        
        # 按相似度排序并获取前top_k个结果
        similarities.sort(reverse=True)
        top_results = similarities[:top_k]
        
        # 根据length筛选并整理结果
        filtered_results = []
        for _, record in top_results:
            if record[4] > 500:
                filtered_results.append(record[3])  # keywords
            else:
                filtered_results.append(record[2])  # content
        
        results[table_name] = filtered_results
    
    conn.close()
    return results


@app.route('/change_model', methods=['POST'])
def change_model():
    data = request.json
    new_model = data.get('model')
    new_api_key = data.get('api_key')
    
    if new_model not in MODEL_CONFIG:
        return jsonify({"success": False, "error": f"Model '{new_model}' not supported."}), 400
    
    # 更新MODEL_CONFIG中的API_KEY
    MODEL_CONFIG[new_model]['api_key'] = new_api_key
    
    # 更新当前模型
    global current_model
    current_model = new_model
    
    return jsonify({"success": True, "current_model": current_model, "updated_api_key": new_api_key})


@app.route('/history', methods=['GET'])
def get_history():
    conn = sqlite3.connect('res_database.db')
    c = conn.cursor()
    c.execute(f"SELECT query_time, query FROM {current_table_name} ORDER BY query_time DESC LIMIT 50")
    history = [{"query_time": row[0], "query": row[1]} for row in c.fetchall()]
    conn.close()
    return jsonify({"success": True, "history": history})

@app.route('/open_folder', methods=['GET'])
def open_folder():
    folder_path = r"csv"
    try:
        os.startfile(folder_path)
        return jsonify({"success": True})
    except:
        return jsonify({"success": False})

@app.route('/new_conversation', methods=['POST'])
def new_conversation():
    global current_table_name
    current_table_name = init_res_db()
    return jsonify({"success": True})


def init_res_db():
    conn = sqlite3.connect('res_database.db')
    c = conn.cursor()
    
    # Create a new table for queries with a timestamp-based name
    table_name = f"queries_{int(time.time())}"
    c.execute(f'''CREATE TABLE {table_name}
                 (query_id INTEGER PRIMARY KEY AUTOINCREMENT,
                  query_time DATETIME,
                  query TEXT,
                  client_prompt TEXT,
                  rag_content TEXT,
                  response TEXT)''')
    
    conn.commit()
    conn.close()
    
    return table_name

# Global variable to store the current table name
current_table_name = init_res_db()

def save_query_to_database(query, client_prompt, rag_content, response):
    conn = sqlite3.connect('res_database.db')
    c = conn.cursor()
    
    c.execute(f'''INSERT INTO {current_table_name}
                  (query_time, query, client_prompt, rag_content, response)
                  VALUES (?, ?, ?, ?, ?)''',
              (time.strftime('%Y-%m-%d %H:%M:%S'), query, client_prompt, rag_content, response))
    
    conn.commit()
    conn.close()

@app.route('/ask', methods=['POST'])
def ask():
    start_time = time.time()
    try:
        data = request.json
        question = data.get('question')
        top_k = data.get('top_k', 5)
        top_p = data.get('top_p', 0.8)
        model = data.get('model', current_model)
        history = data.get('history', [])
        client_prompt = '你是一位德高望重、庙算无遗的风水大师，请严格按照人设作答，并保持谦虚，尊重客户。'
        rag_content, prompt_template = generate_prompt_template(question, client_prompt, history)

        print(prompt_template)

        # 获取对应模型的API_KEY和base_url
        model_config = MODEL_CONFIG.get(model)
        if not model_config:
            raise ValueError(f"Model '{model}' not supported.")

        client = OpenAI(
            api_key=model_config["api_key"],
            base_url=model_config["base_url"]
        ) 

        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt_template},
                {"role": "user", "content": question}
            ],
            top_p=top_p,
            temperature=0.9
        ) 

        answer = completion.choices[0].message.content

        # 保存查询信息到数据库
        save_query_to_database(question, client_prompt, rag_content, answer)

        return jsonify({'success': True, 'answer': answer, 'status': f"滴答，你的答案已送达！耗时 {time.time() - start_time:.2f} 秒"})
    except Exception as e:
        logging.error(f"Error in ask: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e), 'status': '哎呀，失败了呢~'}), 500


# 添加新的路由来提供HTML文件
@app.route('/')
def serve_html():
    return send_from_directory('static', 'updated-fortune-telling-app.html')

# 创建一个Blueprint来处理静态文件
@static_bp.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

# 注册Blueprint以使其生效
app.register_blueprint(static_bp)

if __name__ == "__main__":
    # 使得应用在本地网络中可访问
    app.run(host='0.0.0.0', port=5000, debug=True)
