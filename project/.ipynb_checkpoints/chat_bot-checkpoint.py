
# -*- coding: utf-8 -*-

# 请输入以下命令启动前端
# export PATH="$PATH:/home/nvidia/.local/bin"
# streamlit run chat_bot.py --server.port 5000
# 外网端口：http://36.150.110.74:9519/

from ui_methods import *
import parse_pdf

path = '/home/nvidia/aws_hackathon_demo/krame'

# 设置日志级别为 ERROR，这样只会显示错误信息，忽略警告
logging.getLogger("streamlit").setLevel(logging.ERROR)
allow_multi_modal = True

# 初始化会话状态
if 'counter' not in st.session_state:
    st.session_state.counter = 0
if st.session_state.counter == 0:
    recovery_list = []
    with open('data.pkl', 'wb') as f:
        pickle.dump(recovery_list, f)
    st.session_state.counter = 1
else:
    try:
        with open('data.pkl', 'rb') as f:
            recovery_list = pickle.load(f)  # {'role':'user/assistant','type':'string/pic',content:''}
    except FileNotFoundError:
        st.error('未找到 data.pkl 文件')
        recovery_list = []

# 初始化主界面
st.title("卡拉米のDEMO")
st.write("我的第一个专属机器人，它可以回答你的问题，也可以和你聊天。")

logo_display(path)
background_set()
hide_streamlit_header_footer()
display_existing_messages(recovery_list)

llm_model="deepseek-ai/deepseek-r1"
embedding_model="baai/bge-m3"
vector_db_path="/home/nvidia/aws_hackathon_demo/vdb"

bot = RAGCarAdvisor(llm_model=llm_model, embedding_model=embedding_model, vector_db_path=vector_db_path)
# 初始化侧边栏
st.sidebar.header("其他功能")

st.sidebar.subheader("向量数据库建库")

if st.sidebar.button('执行', help='向量数据库建库'):
    # **在下面添加向量库建库函数**
    parse_pdf.main(folder_path=path+'/nvidia/wei-test-data/Bright16 Air', faiss_db_path=path+'/nvidia/vdb')
    message_text = "向量数据库建库执行成功！"
    with st.chat_message('assistant'):
        message_placeholder = st.empty()
        full_response = ""
        # 模拟流式输出，逐字添加字符
        for char in message_text:
            full_response += char
            # 更新占位符的内容
            message_placeholder.markdown(full_response)
            # 添加延迟，控制输出速度
            time.sleep(0.05)

    recovery_list.append({'role': 'assistant', 'type': "string", 'content': full_response})

uploaded_file = st.sidebar.file_uploader("向机器人传图", type=['png', 'jpg', 'jpeg', 'txt'])

# 上传图片功能
if st.sidebar.button('上传', help='提交图片'):
    image = Image.open(uploaded_file)
    add_user_message_to_session(image, recovery_list)
    if allow_multi_modal:

        # save_path = /home/lab/nvidia/wei-test-data/Bright16 Air/'
        save_path = path+'/nvidia/wei-test-data/Bright16 Air/'
        save_name = generate_unique_time_string_with_uuid() + '.jpg'
        image.save(save_path + save_name)

        # 此处添加返回信息
        message_text = "收到图片啦！"
        with st.chat_message('assistant'):
            # response = pic_recognize_response(image)

            message_placeholder = st.empty()
            full_response = ""
            # 模拟流式输出，逐字添加字符
            for char in message_text:
                full_response += char
                # 更新占位符的内容
                message_placeholder.markdown(full_response)
                # 添加延迟，控制输出速度
                time.sleep(0.05)

        recovery_list.append({'role': 'assistant', 'type': "string", 'content': full_response})

IMAGE_FOLDER = "/home/lab/krame/nvidia/process_chart"  # 修改为你的图片文件夹路径

# 侧边栏上传按钮
if st.sidebar.button('加载图片', help='从文件夹加载图片'):
    if os.path.exists(IMAGE_FOLDER):
        images = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('png', 'jpg', 'jpeg', 'gif'))]

        if images:
            for img_file in images:
                img_path = os.path.join(IMAGE_FOLDER, img_file)
                image = Image.open(img_path)

                # 显示图片在聊天框
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = f"加载图片：{img_file} 📷\n"

                    # 逐字模拟流式输出
                    for char in full_response:
                        message_placeholder.markdown(full_response[:full_response.index(char) + 1])
                        time.sleep(0.05)  # 控制输出速度

                    st.image(image, caption=img_file)  # 显示图片

# 输入框
query = st.chat_input("你可以问我任何你想问的问题")
if query:
    add_user_message_to_session(query, recovery_list)  # 存储问句
    response = generate_assistant_response(query, recovery_list, bot)  # 回答生成

# 存储对话
with open('data.pkl', 'wb') as f:
    pickle.dump(recovery_list, f)
