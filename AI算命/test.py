import os
import logging
import sqlite3
import numpy as np
import time
import datetime
import requests
import openai
from sentence_transformers import SentenceTransformer
import torch
from sklearn.metrics.pairwise import cosine_similarity

from zhdate import ZhDate



# 定义一个字典存储不同模型对应的API_KEY和base_url
MODEL_CONFIG = {
    "glm-4": {
        "api_key": "YOUR_API_key",
        "base_url": "https://open.bigmodel.cn/api/paas/v4/"
    },
    "abab6.5s-chat": {
        "api_key": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJHcm91cE5hbWUiOiLolKHlub_nj4oiLCJVc2VyTmFtZSI6IuiUoeW5v-ePiiIsIkFjY291bnQiOiIiLCJTdWJqZWN0SUQiOiIxODEwNTMwOTYxMjcyNjA3NDgzIiwiUGhvbmUiOiIxMzEyMjQxMDE3NiIsIkdyb3VwSUQiOiIxODEwNTMwOTYxMjY4NDEzMTE0IiwiUGFnZU5hbWUiOiIiLCJNYWlsIjoiIiwiQ3JlYXRlVGltZSI6IjIwMjUtMDItMDMgMTI6MTY6MzMiLCJUb2tlblR5cGUiOjEsImlzcyI6Im1pbmltYXgifQ.MVKluuSS5qgHbZt2ijTr5rtcJaUC-cXN0Mh090XK2d8UtofXgmU8YND3cduinlVpYFDAoDTI2Esv1tzsWAx_7XStI-1-yWTTqNhh2sIc9a844_cRzNdOIaQIF3UjgK5cPQ_yZPvvjCGqvE8imhpRqHtv8wkVZXBamIsyElEspXJuT0MT6YSILDi4HUJ7azxJugeSnh6jsFKQzuKcF21ceO7AshrktMKKK3VCy3WngrRwh9N34WMisNH0Lwk-qDW-Sj0JTh5uQp0TIJoLCFrsUzTyo3s946La3C15ePogA1Es_qFxKZlxuDXyraMo_FDRjg9_Dmfl1_RoI40lEieE3g",  # 你的 MiniMax API key
        "base_url": "https://api.minimax.chat/v1"
    },
    "local-model": {
        "api_key": "",  # 本地模型不需要 API key
        "base_url": "http://localhost:1234/v1",
        "model_name": "deepseek-r1-distill-llama-8b"  # 本地模型名称
    }
}

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
    # print(f'成功向量化为：{query_vector}')

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



def view_first_ten_records(db_file):
    """
    此函数用于查看 SQLite 数据库文件中的前 10 条记录。
    :param db_file: 数据库文件的路径
    :return: 无
    """
    try:
        # 连接数据库
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        # 获取数据库中所有表的名称
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        for table in tables:
            table_name = table[0]
            print(f"Table: {table_name}")
            # 从表中获取前 10 条记录
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 10")
            records = cursor.fetchall()
            for record in records:
                print(record)
        # 关闭数据库连接
        conn.close()
    except sqlite3.Error as e:
        print(f"Error occurred: {e}")

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
            # 确保 history 中的每个消息都有 'role' 和 'content' 字段
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
        # 确保即使发生错误也返回有效值
        rag_content = ""
        prompt_template = f"""
        问题：{question}
        角色设定：{client_prompt}
        请回答：
        """

    # 设置 tokenizers 并行处理
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    return rag_content, prompt_template

def ask(question: str, current_model: str):
    start_time = time.time()
    try:
        client_prompt = '你是一位德高望重、庙算无遗的风水大师，请严格按照人设作答，并保持谦虚，尊重客户。'
        rag_content, prompt_template = generate_prompt_template(question, client_prompt)

        print("问题:", question)
        print("提示模板:", prompt_template)

        if current_model == "abab6.5s-chat":
            # MiniMax API 调用逻辑保持不变
            group_id = "1810530961268413114"
            url = f"https://api.minimax.chat/v1/text/chatcompletion_pro?GroupId={group_id}"
            headers = {
                "Authorization": f"Bearer {MODEL_CONFIG[current_model]['api_key']}", 
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "abab6.5s-chat",
                "tokens_to_generate": 8192,
                "reply_constraints": {
                    "sender_type": "BOT",
                    "sender_name": "算命大师"
                },
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
                ],
                "bot_setting": [
                    {
                        "bot_name": "算命大师",
                        "content": client_prompt
                    }
                ],
            }
            
            response = requests.post(url, headers=headers, json=payload)
            
        elif current_model == "local-model":
            # 本地模型 API 调用
            url = f"{MODEL_CONFIG[current_model]['base_url']}/chat/completions"
            headers = {"Content-Type": "application/json"}
            
            payload = {
                "model": MODEL_CONFIG[current_model]["model_name"],
                "messages": [
                    {"role": "system", "content": prompt_template},
                    {"role": "user", "content": question}
                ],
                "temperature": 0.7,
                "max_tokens": -1,
                "stream": False
            }
            
            response = requests.post(url, headers=headers, json=payload)
            
        else:  # GLM-4 或其他模型
            client = openai.OpenAI(
                api_key=MODEL_CONFIG[current_model]["api_key"],
                base_url=MODEL_CONFIG[current_model]["base_url"]
            )

            completion = client.chat.completions.create(
                model=current_model,
                messages=[
                    {"role": "system", "content": prompt_template},
                    {"role": "user", "content": question}
                ],
                temperature=0.9
            )
            return completion.choices[0].message.content

        # 处理响应
        if response.status_code != 200:
            raise Exception(f"API request failed: {response.text}")
            
        result = response.json()
        print("API响应:", result)
        
        # 根据不同模型处理响应格式
        if current_model == "abab6.5s-chat":
            answer = result.get("reply", "")
        elif current_model == "local-model":
            answer = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        else:
            answer = "未知模型类型"
            
        print("最终答案:", answer)
        return answer

    except Exception as e:
        logging.error(f"Error in ask: {str(e)}", exc_info=True)
        return f"错误: {str(e)}"

'''
=================================================

测试函数

=================================================
'''

def test_search_similar_records():
    query = '今天适合吃肯德基吗？'
    results = search_similar_records(query, r"AI算命/csv/suanming_database.db")
    print(f'检索结果：{results}')

def test_generate_prompt_template():
    """测试生成提示模板的函数"""
    print("\n=== 测试 generate_prompt_template ===")
    
    # 测试场景1：基本问题，无历史记录
    question1 = "今天适合吃肯德基吗？"
    client_prompt1 = "你是一位德高望重、庙算无遗的风水大师，请严格按照人设作答，并保持谦虚，尊重客户。"
    print("\n测试1 - 基本问题:")
    rag_content1, prompt1 = generate_prompt_template(question1, client_prompt1)
    print(f"RAG内容: {rag_content1[:200]}...")  # 只打印前200个字符
    print(f"生成的提示模板: {prompt1}")

    # 测试场景2：带历史记录的问题
    question2 = "那明天呢？"
    history = [
        {"role": "user", "content": "今天适合吃肯德基吗？"},
        {"role": "assistant", "content": "根据您的问题，我需要谨慎考虑..."}
    ]
    print("\n测试2 - 带历史记录:")
    rag_content2, prompt2 = generate_prompt_template(question2, client_prompt1, history)
    print(f"RAG内容: {rag_content2[:200]}...")
    print(f"生成的提示模板: {prompt2}")

def test_ask():
    """测试ask函数"""
    print("\n=== 测试 ask ===")
    
    # 测试不同的问题
    test_questions = [
        "请分析下面的八字：乙卯 壬申 乙巳 乙酉。2021~2030走辛未大运。请分析该八字2025年的学业运并说明原因。",
    ]
    
    # 测试不同的模型
    test_models = ["local-model"]
    
    for model in test_models:
        print(f"\n测试模型: {model}")
        for question in test_questions:
            print(f"\n问题: {question}")
            try:
                ask(question, model)
            except Exception as e:
                print(f"错误: {str(e)}")
            print("-" * 50)

def test_local_model():
    print("\n=== 测试本地模型 ===")
    question = "今天适合吃肯德基吗？"
    try:
        answer = ask(question, "local-model")
        print(f"问题: {question}")
        print(f"回答: {answer}")
    except Exception as e:
        print(f"测试失败: {str(e)}")

def main():
    """主测试函数"""
    try:
        # 测试提示模板生成
        test_generate_prompt_template()
        
        # 测试ask函数
        test_ask()
        
        # # 测试本地模型
        # test_local_model()
        
    except Exception as e:
        print(f"测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()