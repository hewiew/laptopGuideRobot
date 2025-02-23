# -*- coding: utf-8 -*-
import sys
import uuid
from datetime import datetime
from PIL import Image
import streamlit as st
# import ollama
import pickle
import base64
from io import BytesIO
import logging
import time
# from chenyj import *

from RAG_TEST import *
path = '/home/lab/krame'
class LLM_model():
    def __init__(self, model_name="gemma2"):
        super().__init__()
        self.model_name = model_name

    def single_chat_response(self, txt_content, question_content):

        response = ollama.chat(model=self.model_name, messages=[
            {
                'role': 'user',
                'content': txt_content + '**' + '请用中文回答，' + question_content + '**',
            },
        ])
        return response['message']['content']

    def chat_bot_response(self, question_content):
        response = ollama.chat(model="gemma2", messages=[
            {
                'role': 'user',
                'content': question_content,
            },
        ])
        return response['message']['content']

    def list_response(self, txt_content, question_content):
        response = ollama.chat(model="gemma2", messages=[
            {
                'role': 'user',
                'content': txt_content + '**' + '只返回给定列表中的元素' + question_content + '**',
            },
        ])
        return response['message']['content']

    def multi_modal_response(self, question, image_base64):
        response = ollama.chat(model="llama3.2-vision", messages=[
            {
                'role': 'user',
                'content': question,
                'images': [image_base64]
            },
        ])
        return response['message']['content']
# ollama_model = LLM_model(model_name="gemma2")

def display_existing_messages(recovery_list):
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    for message in recovery_list:
        with st.chat_message(message["role"]):
            if message["type"] == 'pic':
                st.image(message['content'], caption='上传的图片', use_container_width =True)
            else:
                st.markdown(message["content"])


def add_user_message_to_session(prompt, recovery_list):
    if prompt:
        role = 'user'
        if isinstance(prompt, Image.Image):
            my_type = 'pic'
            with st.chat_message("user"):
                st.image(prompt, caption='上传的图片', use_container_width =True)
        else:
            st.session_state["messages"].append({"role": "user", "content": prompt})
            my_type = 'string'
            with st.chat_message("user"):
                st.markdown(prompt)
        recovery_list.append({'role': role, 'type': my_type, 'content': prompt})


def add_assi_message_to_session(prompt, recovery_list):
    if prompt:
        role = 'assistant'
        if isinstance(prompt, Image.Image):
            my_type = 'pic'
            with st.chat_message("assistant"):
                st.image(prompt, caption='上传的图片', use_container_width =True)
        else:
            st.session_state["messages"].append({"role": "assistant", "content": prompt})
            my_type = 'string'
            with st.chat_message("assistant"):
                st.markdown(prompt)
        recovery_list.append({'role': role, 'type': my_type, 'content': prompt})


def generate_assistant_response(query, recovery_list):
    # add_user_message_to_session 显示消息的时候做了处理，所以这里不需要再次添加最新提问
    print('history-->')
    # 历史对话记录
    history = st.session_state["messages"]
    print(history)
    role = 'assistant'
    with st.chat_message(role):
        message_placeholder = st.empty()

        # 模型处理
        # 模型输出为message_text
        
        message_text = main(query, apikey=api__key.api_key, save_path=path+'/nvidia/process_chart', llm_model="deepseek-ai/deepseek-r1", embedding_model="baai/bge-m3",vector_db_path=path+'/nvidia/vdb')

        # 多模态处理在此处
        message_placeholder = st.empty()
        full_response = ""
        # 模拟流式输出，逐字添加字符
        # for char in predict(query, history):
        for char in message_text:
            full_response += char
            # 更新占位符的内容
            message_placeholder.markdown(full_response)
            # 添加延迟，控制输出速度
            time.sleep(0.05)

        if isinstance(full_response, Image.Image):
            my_type = 'pic'
            st.image(full_response, caption='上传的图片', use_container_width =True)
        else:
            message_placeholder.markdown(full_response)
            st.session_state["messages"].append({"role": role, "content": full_response})
            my_type = 'string'
        recovery_list.append({'role': role, 'type': my_type, 'content': full_response})
    return full_response


def hide_streamlit_header_footer():
    hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            #root > div:nth-child(1) > div > div > div > div > section > div {padding-top: 0rem;}
            </style>
            """
    st.markdown(hide_st_style, unsafe_allow_html=True)


def background_set():
    # 定义自定义 CSS 样式，设置背景颜色为浅蓝色
    custom_css = """
    <style>
        /* 改变整个body的背景颜色 */
        body {
            background-color: #f0f2f6; /* 选择你喜欢的颜色 */
        }

        /* 如果需要，也可以单独设置主内容区的背景颜色 */
        .main {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
        }
    </style>
    """
    # 注入自定义 CSS 样式
    st.markdown(custom_css, unsafe_allow_html=True)


def logo_display():
    # 打开本地图片文件
    image_path = path+"/nvidia/icon/icon.png"  # 请替换为实际的图片路径
    image = Image.open(image_path)
    # 在侧边栏显示图片
    st.sidebar.image(image, caption='卡拉米', use_container_width =True)


# 将PIL图像转换为Base64编码字符串
def convert_to_base64(pil_image):
    buffered = BytesIO()
    # 将图像转换为RGB模式
    pil_image = pil_image.convert("RGB")
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


# 从指定路径加载图像并转换为Base64编码字符串
def load_image(file_path):
    pil_image = Image.open(file_path)
    return convert_to_base64(pil_image)


def generate_unique_time_string_with_uuid():
    # 获取当前时间，并格式化为字符串
    time_string = datetime.now().strftime('%Y%m%d%H%M%S%f')
    # 生成 UUID
    unique_id = uuid.uuid4().hex
    # 合并时间字符串和 UUID
    unique_time_string = 0
    return str(time_string)


def pic_recognize_response(pil_instant):

    return ollama_model.multi_modal_response("**请解释这张图，用中文回答**", convert_to_base64(pil_instant))

