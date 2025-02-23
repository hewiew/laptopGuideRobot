from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
import numpy as np
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from typing import List
from langchain.schema import Document
import base64
import matplotlib.pyplot as plt
import re
import check
import api__key
from llama_cpp import Llama
api_key = api__key.api_key
os.environ["NVIDIA_API_KEY"] = api_key
refuse_model = "meta/llama-3.3-70b-instruct"

# 查看当前可以使用的模型
# In[3]:
chart_reading = ChatNVIDIA(model="qwen/qwen2.5-coder-32b-instruct")
embedding_model = NVIDIAEmbeddings(model="baai/bge-m3")

def extract_python_result(text):
    if not isinstance(text, str):
        raise ValueError("Expected string or bytes-like object, but got type {}".format(type(text)))
    pattern = r'###\s*(.*?)\s*###'
    matches = re.findall(pattern, text, re.DOTALL)
    c = [match.strip() for match in matches]
    # print(c)
    return c[0]

def extract_python_code(text):
    if not isinstance(text, str):
        raise ValueError("Expected string or bytes-like object, but got type {}".format(type(text)))
    pattern = r'###python\s*(.*?)\s*###'
    matches = re.findall(pattern, text, re.DOTALL)
    #print('-' * 50, matches)
    c = [match.strip() for match in matches]

    return c[0]

# 执行由大模型生成的代码
def execute_and_return(x):
    code = extract_python_code(x)
    print("代码：", str(code))
    try:
        result = exec(str(code))
        # print("exec result: "+result)
    except:
        print("The code is not executable, don't give up, try again!")
    return x


class RAGCarAdvisor:
    def __init__(self, llm_model="deepseek-ai/deepseek-r1", embedding_model="ai-embed-qa-4", vector_db_path="../aws_hackathon_demo/vdb"):
        """
        初始化 RAG 机器人
        :param llm_model: 用于优化查询和生成回答的 LLM
        :param embedding_model: 用于生成嵌入向量的模型
        :param vector_db_path: 向量数据库的存储路径（FAISS）
        """
        self.llm = ChatNVIDIA(model_name=llm_model)
        self.embedding_model = NVIDIAEmbeddings(model=embedding_model)
        self.vector_db = FAISS.load_local(vector_db_path, self.embedding_model, allow_dangerous_deserialization=True)
        # self.local_llm = Llama("/home/nvidia/aws_hackathon_demo/model/Llama-3.1-8b-chinese-Q4_K_M.gguf", flash_attn=True, n_gpu_layers=-1)

    def optimize_query(self, query: str) -> str:
        """
        使用 LLM 进行查询优化，使其更清晰、可检索
        :param query: 用户输入的原始查询
        :return: 优化后的查询
        """
        prompt_template = PromptTemplate(
            template="请优化以下查询，使其更清晰、更适合检索数据库：\n用户查询：{query}\n优化查询：",
            input_variables=["query"]
        )
        optimized_query = self.llm.invoke(prompt_template.format(query=query))
        return optimized_query.content

    def check_bounds(self, query: str) -> str:
        """
        使用 LLM 进行查询优化，使其更清晰、可检索
        :param query: 用户输入的原始查询
        :return: 优化后的查询
        """
        prompt_template = PromptTemplate(
            template="""判断以下用户提问是否与国补（国家补贴）相关，国补可以给用户金钱补贴。如果相关，输出###1###，不相关，输出###0###。注意，只输出结果，不要输出任何无关内容。
                            query:{query}""",
            input_variables=["query"]
        )
        optimized_query = self.llm.invoke(prompt_template.format(query=query))
        result = extract_python_result(optimized_query.content)
        return result

    def retrieve_documents(self, query: str, top_k: int = 5) -> List[Document]:
        """
        从向量数据库中检索与查询最相关的文档
        :param query: 经过优化的查询文本
        :param top_k: 返回的文档数量
        :return: 检索到的文档列表
        """
        return self.vector_db.similarity_search(query, k=top_k)

    def generate_response(self, query: str, retrieved_docs: List[Document]) -> str:
        """
        结合检索到的文档和 LLM 生成最终回答
        :param query: 用户的查询
        :param retrieved_docs: 检索到的相关文档
        :return: 最终回答
        """
        context = "\n".join([doc.page_content for doc in retrieved_docs])
        prompt_template = PromptTemplate(
            template="你是一名笔记本销售顾问，用户正在寻找购买笔记本的建议。请基于以下信息回答问题：\n\n{context}\n\n用户的问题：{query}\n\n直接给出回答，不要说出思考逻辑，使用中文回答问题：",
            input_variables=["context", "query"]
        )
        response = self.llm.invoke(prompt_template.format(context=context, query=query))
        return response.content

    def generate_chart(self, retrieved_docs: List[Document], save_path) -> str:
        """
        结合检索到的文档和 LLM 生成最终回答
        :param query: 用户的查询
        :param retrieved_docs: 检索到的相关文档
        :return: 最终回答
        """
        context = "\n".join([doc.page_content for doc in retrieved_docs])
        prompt_template = PromptTemplate(
            template="""基于以下给出的消费补助使用流程信息，生成一段python代码，代码可生成一张简单流程图描述消费补助使用方法.
            仅输出生成的代码,设定字体是SimSun，dot.attr('node', fontname='SimSun')。生成的图片保存到 ”{save_path}“下, 生成的代码以###python开始，以###结束.
            注意：dot = Digraph(comment='消费补助使用流程', format='png')是唯一正确的形式。{context}""",
            input_variables=["context", "save_path"]
        )
        response = self.llm.invoke(prompt_template.format(context=context, save_path=save_path))
        print('*'*50, response)
        execute_and_return(response.content)
        #return response.content

    def ask(self, api_key, refuse_model, query: str, save_path) -> str:
        """
        用户询问汽车相关问题，完整执行 RAG 流程
        :param query: 用户原始查询
        :return: 生成的答案
        """

        res = check.query_check(query, refuse_model)
        # print('-------res:', res)

        if res != 1:
            return res

        # if self.check_bounds(query) == '1':
        #     # 1. 优化查询
        #     # optimized_query = self.optimize_query(query)
        #     # print(f"优化后的查询: {optimized_query}")

        #     # 2. 检索相关文档
        #     retrieved_docs = self.retrieve_documents(query)
        #     print('绘图相关', retrieved_docs)
        #     self.generate_chart(retrieved_docs, save_path)

        # 1. 优化查询
        optimized_query = self.optimize_query(query)
        #print(f"优化后的查询: {optimized_query}")

        # 2. 检索相关文档
        retrieved_docs = self.retrieve_documents(optimized_query)

        # 3. 生成回答
        response = self.generate_response(optimized_query, retrieved_docs)

        #check.answer_check()

        return response

# def jpg2text(jpg):
#
#     with open(jpg, "rb") as f:
#         image_b64 = base64.b64encode(f.read()).decode()
#
#     chart_reading = ChatNVIDIA(model="microsoft/phi-3-vision-128k-instruct")
#     result = chart_reading.invoke(
#         f'Generate describe of the figure below, : <img src="data:image/png;base64,{image_b64}" />')
#
#     return result
def main(query,apikey, save_path='/home/nvidia/aws_hackathon_demo/process_chart', llm_model="deepseek-ai/deepseek-r1", embedding_model="ai-embed-qa-4", vector_db_path="/home/nvidia/aws_hackathon_demo/vdb"):

    a = RAGCarAdvisor(llm_model=llm_model, embedding_model=embedding_model, vector_db_path=vector_db_path)
    p1 = "给我推荐一款能玩3A大作，帧率高的电脑，比如Dying Light 2 ，Shadow of the Tomb Raider"
    p2 = "我想要买一台6000元左右的游戏本，显卡我想要NVIDA的GTX 4080， CPU不要AMD的，AMD的太垃圾了，差劲的让想我打人"
    p3 = "我想购买游戏本电脑，国补,我想用国补购买笔记本，怎样申请，怎样购买"
    response = a.ask(apikey, refuse_model, query, save_path=save_path)
    print(response)
    return response
p3 = "我想购买游戏本电脑，国补,我想用国补购买笔记本，怎样申请，怎样购买"
#main(p3, apikey=api__key.api_key, save_path='../aws_hackathon_demo/process_chart', llm_model="deepseek-ai/deepseek-r1", embedding_model="ai-embed-qa-4",vector_db_path='Y:/nvidia')
main(p3, apikey=api__key.api_key, save_path='/home/nvidia/aws_hackathon_demo/process_chart', llm_model="deepseek-ai/deepseek-r1", embedding_model="ai-embed-qa-4",vector_db_path='/home/nvidia/aws_hackathon_demo/vdb')