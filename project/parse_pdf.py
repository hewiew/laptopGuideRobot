from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
import base64
import load_raw_info
import api__key
import os


def write_txt_files(folder_path, info):
    """读取指定文件夹下的所有 .txt 文件"""
    txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]

    for file in txt_files:
        file_path = os.path.join(folder_path, file)

        with open(file_path, "a", encoding="utf-8") as f:
            f.write("\n\n" + info)



os.environ["NVIDIA_API_KEY"] = api__key.api_key
#os.environ["NVIDIA_API_KEY"] = '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'

#chart_reading = ChatNVIDIA(model="deepseek-ai/deepseek-r1")
#embedding_model = NVIDIAEmbeddings(model="ai-embed-qa-4")
class FAISSBuilder:
    def __init__(self, embedding_model="text-embedding-ada-002", llm_model="qwen/qwen2.5-7b-instruct"):
        """
        初始化 FAISS 向量数据库构建器
        :param embedding_model: 用于生成嵌入向量的模型
        """
        self.embedding_model = NVIDIAEmbeddings(model=embedding_model)
        self.llm = ChatNVIDIA(model_name=llm_model)

    def process_document(self, raw_text: str) -> str:
        """
        使用 LLM 处理文档，使其更适合 RAG 调用
        :param raw_text: 原始文档内容
        :return: 处理后的文档
        """
        prompt_template = PromptTemplate(
            template=(
                "请对以下文本进行清理和优化，使其适合用于 RAG 数据库存储：\n"
                "1. 过滤掉无关内容（如噪音、无意义段落）。\n"
                "2. 结构化内容，确保清晰易读。\n"
                "3. 提取摘要和关键词，以便后续检索。\n\n"
                "原始文本：{raw_text}\n\n"
                "注意：只输出中文结果，不要输出自己修改内容的思路，只输出原文！"
            ),
            input_variables=["raw_text"]
        )

        processed_text = self.llm.invoke(prompt_template.format(raw_text=raw_text))
        #print('processed_text', processed_text)
        return processed_text.content.strip()

    def split_document(self, text, chunk_size=250, chunk_overlap=20):
        """
        使用 CharacterTextSplitter 将文本分割为不超过 chunk_size 的片段。

        参数：
            text (str): 需要分割的原始文本。
            chunk_size (int): 每个文本块的最大字符长度。
            chunk_overlap (int): 每个块之间的重叠部分，防止语义断裂。

        返回：
            List[str]: 分割后的文本列表。
        """
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return text_splitter.split_text(text)

    def split_text_by_blank_line(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # 按空行切分
        paragraphs = [para.strip() for para in content.split('\n\n') if para.strip()]

        return paragraphs

    def pre_process(self, path):
        info = load_raw_info.concat_infos_extract_figures(path)
        write_txt_files(path, info)

    def build_faiss_index(self, folder_path, faiss_db_path="vector_store"):
        """
        读取 TXT 文件，使用 LLM 处理文本，并存入 FAISS 向量数据库
        :param file_path: 文本文件路径
        :param faiss_db_path: FAISS 存储路径
        """
        self.pre_process(folder_path)

        documents = []
        txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]

        for file in txt_files:
            file_path = os.path.join(folder_path, file)
            paragraphs = self.split_text_by_blank_line(file_path)

            documents.extend(paragraphs)

        d_c = []

        for idx, line in enumerate(documents):
            d_c.append(Document(
                page_content=line,
                metadata={"source": f"line_{idx}"}
            ))

        vector_db = FAISS.from_documents(d_c, self.embedding_model)
        vector_db.save_local(faiss_db_path)
        print(f"✅ FAISS 向量数据库已保存到: {faiss_db_path}")

    def load_faiss_index(self, faiss_db_path="vector_store"):
        """
        加载 FAISS 向量数据库
        :param faiss_db_path: FAISS 向量数据库路径
        :return: FAISS 数据库对象
        """
        return FAISS.load_local(faiss_db_path, self.embedding_model, allow_dangerous_deserialization=True)

def main(folder_path='Y:/shao/nvidia/wei-test-data/Bright16 Air', faiss_db_path='Y:/shao/nvidia/vdb'):
    # 1. 初始化 FAISS 构建器
    faiss_builder = FAISSBuilder(embedding_model="baai/bge-m3")

    # 2. 加载 TXT 文件数据
    # documents = faiss_builder.load_text_data("Y:/shao/cars.txt")  # 这里是你的文本文件路径

    # 3. 构建 FAISS 向量数据库
    faiss_builder.build_faiss_index(folder_path=folder_path, faiss_db_path=faiss_db_path)

    print('db success')

#main()

    # # 4. 加载 FAISS 数据库
    # vector_db = faiss_builder.load_faiss_index(faiss_db_path='Y:/nvidia')
# ========================== 使用示例 ==========================

# print(jpg2text("Y:/英伟达比赛/aws_hackathon_demo/聊一个可能“改变行业”的设计/v3.jpg", llm_model="deepseek-ai/deepseek-r1"))
# 1. 初始化 FAISS 构建器
# faiss_builder = FAISSBuilder(embedding_model="baai/bge-m3")
#
# # 2. 加载 TXT 文件数据
# #documents = faiss_builder.load_text_data("Y:/shao/cars.txt")  # 这里是你的文本文件路径
#
# # 3. 构建 FAISS 向量数据库
# faiss_builder.build_faiss_index(folder_path="Y:/shao/rag_try3", faiss_db_path='Y:/nvidia')
#
# # 4. 加载 FAISS 数据库
# vector_db = faiss_builder.load_faiss_index(faiss_db_path='Y:/nvidia')
#
# # 5. 测试查询
# query = "Total War: Warhammer 3,  Dying Light 2,Forza Horizon 5"
# retrieved_docs = vector_db.similarity_search(query, k=3)  # 检索最相关的3个文档
# for doc in retrieved_docs:
#     print("相关文档:", doc.page_content)