# 先检查是不是符合topic，再检查内容是否安全。
# 用户输入prompt OK： 返回1
# 不OK： 返回相应拒绝文本

from openai import OpenAI


from api__key import api_key
import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import HumanMessage, SystemMessage

os.environ["NVIDIA_API_KEY"] = api_key

# api_key = "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"
refuse_model = "meta/llama-3.3-70b-instruct"


def judge_query_topic(query):
    system_prompt = f"""
You will be acting as a laptop salesperson. You will patiently answer any questions customers may have regarding the selection and purchase of laptops.
Your role is to ensure that you respond only to relevant queries and adhere to the following guidelines:
1.Only providing users with factual information related to the laptop, such as laptop hardware and software, laptop price, laptop performance.
2.If a user asks about topics irrelevant to laptop, politely redirect the conversation or end the interaction.
3.Your responses should be professional, accurate, and polite, focusing solely on providing transparent, up-to-date information about laptop purchasing.
"""

    # completion = client.chat.completions.create(
    #     model="nvidia/llama-3.1-nemoguard-8b-topic-control",
    #     messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content":
    #         query}],
    #     temperature=0.5,
    #     top_p=1,
    #     max_tokens=1024
    # )
    # return completion.choices[0].message.content

    llm = ChatNVIDIA(
        model="nvidia/llama-3.1-nemoguard-8b-topic-control",
        temperature=0.5,
        top_p=1,
        max_tokens=1024
    )
    # messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content":
    #         query}],
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query)]
    completion = llm.invoke(messages)
    return completion.content


def judge_query_safety(query):
    # completion = client.chat.completions.create(
    #     model="nvidia/llama-3.1-nemoguard-8b-content-safety",
    #     messages=[{"role": "user", "content": query}],
    #     temperature=0.5,
    #     top_p=1,
    #     max_tokens=1024
    # )
    # return completion.choices[0].message.content

    llm = ChatNVIDIA(
        model="nvidia/llama-3.1-nemoguard-8b-content-safety",
        temperature=0.5,
        top_p=1,
        max_tokens=1024
    )
    completion = llm.invoke(query)
    return completion.content


def refuse_base_on_bad_query_type(bad_query_info, refuse_model):
    if bad_query_info == 'off-topic ':
        return "抱歉，作为一个导购机器人，您的提问并不在我的专业范畴内。我很乐意为您解答笔记本电脑相关的问题。"

    # client = OpenAI(
    #     base_url="https://integrate.api.nvidia.com/v1",
    #     api_key=api_key
    # )
    # unsafe_type = bad_query_info.content.split(": ")[-1].split('}')[0]
    unsafe_type = bad_query_info.split(": ")[-1].split('}')[0]
    # print(unsafe_type)
    refuse_prompt = f'''
    Because some unsafe content has been detected in user's query, you are going to politely refuse answeing. Follow instructions below:
    1.Answer in Chinese.
    2.Begin with "非常抱歉，根据我国相关人工智能安全标准，我无法回答您的这个问题", then explain in details with following guides.
    3.The use's query is about {unsafe_type}, ploitely point out this, and tell the user to avoid related query next.
    '''
    # completion = client.chat.completions.create(
    #     model=refuse_model,
    #     messages=[{"role": "user", "content": refuse_prompt}],
    #     temperature=0,
    #     top_p=1,
    #     max_tokens=1024
    # )
    # return completion.choices[0].message.content
    llm = ChatNVIDIA(
        model=refuse_model,
        temperature=0,
        top_p=1,
        max_tokens=1024
    )
    completion = llm.invoke(refuse_prompt)
    return completion.content


# LLM输出的内容安全检测
# 下面输出（answer）审核的两个函数，目前没用到，也没有修改成LangChain格式
# def judge_answer_safety(answer, client):
#     completion = client.chat.completions.create(
#       model="nvidia/llama-3.1-nemoguard-8b-content-safety",
#       messages=[{"role":"assistant","content":answer}],
#       temperature=0.5,
#       top_p=1,
#       max_tokens=1024
#     )
#     print(completion.choices[0].message)
#     return completion.choices[0].message.content

# def answer_check(api_key, output):
#     client = OpenAI(
#   base_url = "https://integrate.api.nvidia.com/v1",
#   api_key = api_key
# )
#     if 'unsafe' in judge_answer_safety(output, client):
#         return 0
#     else:
#         return 1




def query_check(input_prompt, refuse_model="meta/llama-3.3-70b-instruct"):
    # client = OpenAI(
    #     base_url="https://integrate.api.nvidia.com/v1",
    #     api_key=api_key
    # )
    # 检查话题
    topic_check_res = judge_query_topic(input_prompt)
    # print('话题检查：', topic_check_res)
    if topic_check_res == 'off-topic ':
        return refuse_base_on_bad_query_type(topic_check_res, refuse_model)
    # 检查内容安全
    safety_check_res = judge_query_safety(input_prompt)
    # print("安全审核返回值", safety_check_res)
    # print("返回值类型", type(safety_check_res))
    # if type(safety_check_res) != str and 'unsafe' in safety_check_res.content:
    if 'unsafe' in safety_check_res:
        # print(safety_check_res)
        return refuse_base_on_bad_query_type(safety_check_res, refuse_model)
    return 1
