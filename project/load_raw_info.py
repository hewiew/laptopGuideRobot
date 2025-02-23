from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda
from langchain.schema.runnable.passthrough import RunnableAssign
from langchain_core.runnables import RunnableBranch
from langchain_core.runnables import RunnablePassthrough

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

from PIL import Image

import os
import base64
import matplotlib.pyplot as plt
import numpy as np

os.environ["NVIDIA_API_KEY"] = "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"


# 单张图片内容识别相关

def display_image(image_file):
    display(Image.open(image_file))


def image2b64(image_file):
    with open(image_file, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()
        return image_b64


# 获取指定文件夹内所有图片的路径
# 写得比较糙，只用来处理全是图片的文件夹
def get_image_file_lists_in_file(file_name):
    pass


# 简单试了一下，得要先把图片类型分类一下再分别进行具体识别效果比较好，不然识别结果没法看。
def judge_image_type(image_b64):
    judge_prompt = f'''
    Based on the figure <img src="data:image/png;base64,{image_b64}" />, follwoing below instructions output the figure type in[A, side, C, performance], and only output the figure type:
    "A": A photo of a laptop's back cover with a can of beverage included as a size reference.
    "side": treat one laptop as a cuboid, define the laptop's Lid as back side. If the figure is a photo of a laptop's with left/right side facing the viewer, the figure type is "side".
    "C": A photo of one laptop's keyboard.
    "performance": A statistical chart in the form of a line graph, bar chart, etc. 
    '''

    chart_reading = ChatNVIDIA(model="ai-phi-3-vision-128k-instruct")
    result = chart_reading.invoke(judge_prompt)
    # print(result.content)

    return result.content


def extract_info_from_image(image_b64, image_type):
    # 笔记本A面，屏幕后壳
    if image_type == "A":
        extract_prompt = f'''
    extract information from the figure: <img src="data:image/png;base64,{image_b64}" />, follow instructions below:
    1. extract what color the laptop's back over is.
    '''
    # 笔记本侧面
    elif image_type == "side":
        extract_prompt = f'''
    based on the figure taken from one laptop's side: <img src="data:image/png;base64,{image_b64}" />, describe the following data cable ports on the side of the laptop you can see:
    1.Describe is there any Network cable interface.
    2.Describe is there any type-C interface.
    3.Output one single sentence, in the form like 'the laptop has ... interface'.
    '''
    # 笔记本C面，键盘
    elif image_type == "C":
        extract_prompt = f'''
    based on the figure of one laptop's keyboard: <img src="data:image/png;base64,{image_b64}" />, answering following questions:
    1.Is there a numeric keypad? 
    '''
    # 性能评测数据
    elif image_type == "performance":
        extract_prompt = f'''
    based on the figure of the statistical chart : <img src="data:image/png;base64,{image_b64}" />,generate data table.
    If there are other extra information in the statistical, like the test setting, test environment, etc, also output these extra information.
    '''

        # 仅针对fps测试那张图编的prompt
        # extract_prompt = f'Generate two tables based on the figure below, one is the hardware configurations of two laptops, one is the fps performance(except the table, also output the test setting): <img src="data:image/png;base64,{image_b64}" />'

    chart_reading = ChatNVIDIA(model="ai-phi-3-vision-128k-instruct")
    result = chart_reading.invoke(extract_prompt)
    print(result.content)
    return result.content


# 获取一个文件夹内所有jpg图片的列表
def get_jpg_files_list(folder_path):
    files_and_dirs = os.listdir(folder_path)
    file_names = [file for file in files_and_dirs if os.path.isfile(os.path.join(folder_path, file))]
    file_names = [file for file in file_names if file.endswith('.jpg')]

    return file_names


# 对图片们进行多模态分析，输出一个完整的
def concat_infos_extract_figures(folder_path):
    file_names = get_jpg_files_list(folder_path)
    print(file_names)
    info_list = []
    for image_file in file_names:
        image_b64 = image2b64(folder_path + "/" + image_file)
        image_type = judge_image_type(image_b64)
        print(image_type)
        info = extract_info_from_image(image_b64, image_type)
        info_list.append(info)

    # return info_list
    return "\n".join(info_list)

#print(concat_infos_extract_figures("D:/python/nvidia/faiss_index"))