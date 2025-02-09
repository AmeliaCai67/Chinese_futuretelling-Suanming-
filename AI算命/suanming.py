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
import requests
import json
import openai
from datetime import datetime
from zhdate import ZhDate

# 配置日志
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

static_bp = Blueprint('static', __name__, static_folder='static')

# 定义全局变量来存储当前选择的模型
MODEL_CONFIG = ''
current_model = ''

# 配置文件路径
CONFIG_PATH = './config/model_config.json'

def generate_prompt_template(question, client_prompt, history=None):
    try:
        # 获取当前日期时间
        current_time = datetime.datetime.now()
        chinese_weekday = ["一", "二", "三", "四", "五", "六", "日"][current_time.weekday()]
        
        # 获取农历日期
        lunar = ZhDate.from_datetime(current_time)
        lunar_date = f"{lunar.chinese()}"
        
        # 格式化日期时间
        datetime_info = f"""当前时间：{current_time.strftime('%Y年%m月%d日 %H:%M')}
            星期{chinese_weekday}
            农历：{lunar_date}"""
        
        # 初始化 rag_content 为空字符串
        rag_content = ""
        
                
        try:
            # 尝试从本地加载模型
            results = search_similar_records(question, r"AI算命/csv/suanming_database.db", top_k=1)
            rag_content = str(results)
        except Exception as e:
            logging.warning(f"Search similar records failed: {str(e)}")
            # 如果检索失败,使用空的上下文继续
            results = {}
            rag_content = str(results)

        # 构建 prompt
        if history:
            history_context = "\n".join([
                f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" 
                for msg in history if isinstance(msg, dict)
            ])
            prompt_template = f"""请根据以下信息回答问题。
            如果不知道答案，请直接说不知道，不要试图编造答案。

            {datetime_info}

            历史对话：
            {history_context}

            参考资料：
            {rag_content}

            问题：{question}
            角色设定：{client_prompt}
            请回答："""
        else:
            prompt_template = f"""请根据以下信息回答问题。
            如果不知道答案，请直接说不知道，不要试图编造答案。

            {datetime_info}

            参考资料：
            {rag_content}

            问题：{question}
            角色设定：{client_prompt}
            请回答："""

    except Exception as e:
        logging.error(f"Error in generate_prompt_template: {str(e)}")
        rag_content = ""
        prompt_template = f"""
        问题：{question}
        角色设定：{client_prompt}
        请回答：
        """

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    return rag_content, prompt_template

def search_similar_records(query, database_path, top_k=10):
    try:
        # 使用相对于 AI算命 目录的路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "embedding", "text2vec-base-chinese")
        if not os.path.exists(model_path):
            logging.error(f"Model path not found: {model_path}")
            return {}
        model = SentenceTransformer(model_path)
        print('成功加载模型')

    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        return {}
    
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

def load_model_config():
    """从文件加载模型配置"""
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    except Exception as e:
        print(f"Error loading config: {str(e)}")
        return {}

def save_model_config(config):
    """保存模型配置到文件"""
    try:
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"Error saving config: {str(e)}")
        return False

def call_model(question, model_config, prompt_template):
    """统一模型调用入口"""
    try:
        if model_config['type'] == 'local':
            return _call_local(question, model_config, prompt_template)
        elif model_config['type'] == 'remote':
            return _call_remote_api(question, model_config, prompt_template)
        raise Exception("不支持的模型类型")
    except Exception as e:
        logging.error(f"模型调用失败: {str(e)}")
        raise

def _call_local(question, config, prompt_template):
    """统一本地模型调用"""
    url = f"{config['url']}/v1/chat/completions"
    payload = {
        "model": config['model_name'],
        "messages": [
            {"role": "system", "content": prompt_template},
            {"role": "user", "content": question}
        ],
        "temperature": 0.7,
        "max_tokens": -1,
        "stream": False
    }
    return _send_request(url, config.get('api_key'), payload)

def _call_remote_api(question, config, prompt_template):
    """统一远程API调用"""
    provider = config['provider']
    if provider == 'minimax':
        return _call_minimax(question, config, prompt_template)
    elif provider == 'glm':
        return _call_glm(question, config, prompt_template)
    raise Exception("不支持的API提供商")

def _call_minimax(question, config, prompt_template):
    """MiniMax专用调用逻辑"""
    url = f"{config['base_url']}/text/chatcompletion_pro?GroupId={config['group_id']}"
    payload = {
        "model": config['model_name'],
        "messages": [
            {
                "sender_type": "BOT",
                "sender_name": "算命大师",
                "text": prompt_template
            },
            {
                "sender_type": "USER",
                "sender_name": "用户",
                "text": question
            }
        ]
    }
    return _send_request(url, config['api_key'], payload)

def _call_glm(question, config, prompt_template):
    """GLM专用调用逻辑"""
    client = openai.OpenAI(
        api_key=config['api_key'],
        base_url=config['base_url']
    )
    return client.chat.completions.create(
        model=config['model_name'],
        messages=[
            {"role": "system", "content": prompt_template},
            {"role": "user", "content": question}
        ]
    )

def _send_request(url, api_key, payload):
    """统一HTTP请求处理"""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"API请求失败: {response.text}")
    return response.json()

# 示例配置文件结构 (config/model_config.json):
"""
{
    "local-model-1": {
        "name": "本地模型1",
        "type": "local",
        "url": "http://localhost:1234",
        "model_name": "deepseek-r1-distill-llama-8b"
    },
    "minimax-1": {
        "name": "MiniMax",
        "type": "remote",
        "provider": "minimax",
        "base_url": "https://api.minimax.chat/v1",
        "api_key": "your-api-key",
        "group_id": "your-group-id",
        "model_name": "abab6.5s-chat"
    },
    "glm-1": {
        "name": "GLM-4",
        "type": "remote",
        "provider": "glm",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "api_key": "your-api-key",
        "model_name": "glm-4"
    }
}
"""

'''
===========================================
路由
===========================================
'''
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/api/models', methods=['GET'])
def get_models():
    """获取所有模型配置"""
    try:
        config = load_model_config()
        # 如果配置文件为空，初始化默认配置
        if not config:
            config = {
                "minimax-1": {
                    "name": "MiniMax",
                    "type": "remote",
                    "provider": "minimax",
                    "base_url": "https://api.minimax.chat/v1",
                    "api_key": "",
                    "group_id": "1810530961268413114",
                    "model_name": "abab6.5s-chat"
                },
                "glm-1": {
                    "name": "GLM-4",
                    "type": "remote",
                    "provider": "glm",
                    "base_url": "https://open.bigmodel.cn/api/paas/v4",
                    "api_key": "",
                    "model_name": "glm-4"
                }
            }
            save_model_config(config)
        return jsonify(config)
    except Exception as e:
        logging.error(f"Error loading model config: {str(e)}")
        return jsonify({"error": "无法加载模型配置"}), 500

@app.route('/api/models', methods=['POST'])
def update_models():
    """更新模型配置"""
    try:
        new_config = request.json  # 从请求中获取新配置
        if not isinstance(new_config, dict):
            return jsonify({"error": "无效的配置格式"}), 400
            
        # 验证每个模型的必要字段
        for model_id, config in new_config.items():
            if not isinstance(config, dict):
                return jsonify({"error": f"模型 {model_id} 配置无效"}), 400
            required_fields = ["name", "type", "model_name"]
            
            if config["type"] == "local":
                required_fields.append("url")
                # 移除名称一致性校验，仅保留必要字段验证
                # 添加模型名称存在性校验
                if not config.get("model_name"):
                    return jsonify({
                        "error": f"本地模型 {model_id} 必须提供 model_name 字段",
                        "invalid_field": "model_name"
                    }), 400
                    
            elif config["type"] == "remote":
                required_fields.extend(["provider", "base_url", "api_key"])
                
            for field in required_fields:
                if field not in config:
                    return jsonify({"error": f"模型 {model_id} 缺少必要字段: {field}"}), 400

        if save_model_config(new_config):
            return jsonify({"status": "success"})
        else:
            return jsonify({"error": "保存配置失败"}), 500
    except Exception as e:
        logging.error(f"Error saving model config: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/ask', methods=['POST'])
def ask():
    start_time = time.time()
    try:
        data = request.json
        question = data.get('question')
        model_id = data.get('model_id')
        history = data.get('history', [])
        
        if not question:
            return jsonify({"error": "问题不能为空"}), 400
            
        if not model_id:
            return jsonify({"error": "未指定模型"}), 400
            
        # 加载模型配置
        configs = load_model_config()
        model_config = configs.get(model_id)
        
        if not model_config:
            return jsonify({"error": "模型未找到"}), 404
            
        # 生成提示模板
        client_prompt = '你是一位德高望重、庙算无遗的风水大师，请严格按照人设作答，并保持谦虚，尊重客户。'
        rag_content, prompt_template = generate_prompt_template(question, client_prompt, history)

        # 调用模型
        response = call_model(question, model_config, prompt_template)
        
        # 处理响应
        if model_config['type'] == 'local':
            answer = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        elif model_config['provider'] == 'minimax':
            answer = response.get("reply", "")
        elif model_config['provider'] == 'glm':
            answer = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        else:
            answer = "未知模型类型"

        # 保存查询信息到数据库
        save_query_to_database(question, client_prompt, rag_content, answer)

        response_data = {
            "success": True,
            "answer": {
                "role": "assistant",
                "content": answer,
                "timestamp": datetime.now().isoformat()
            },
            "status": {
                "text": f"耗时 {time.time() - start_time:.2f} 秒",
                "code": 200
            }
        }
        return jsonify(response_data)
        
    except Exception as e:
        logging.error(f"Error in ask: {str(e)}", exc_info=True)
        error_response = {
            "success": False,
            "error": {
                "code": 500,
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            },
            "status": {
                "text": "哎呀，失败了呢~",
                "code": 500
            }
        }
        return jsonify(error_response), 500


@app.route('/new_conversation', methods=['POST'])
def new_conversation():
    global current_table_name
    current_table_name = init_res_db()
    return jsonify({"success": True})

@app.route('/chat')
def chat():
    return send_from_directory('static', 'updated-fortune-telling-app.html')

# 创建一个Blueprint来处理静态文件
@static_bp.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

# 注册Blueprint以使其生效
app.register_blueprint(static_bp)

if __name__ == "__main__":
    # 使得应用在本地网络中可访问
    app.run(host='0.0.0.0', port=8000, debug=True)
