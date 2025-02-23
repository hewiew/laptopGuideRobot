from llama_cpp import Llama

llm = Llama("/home/nvidia/aws_hackathon_demo/model/Llama-3.1-8b-chinese-Q4_K_M.gguf", flash_attn=True, n_gpu_layers=-1)
def predict(message, history):
    messages = []

    for user_message, assistant_message in history:
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": assistant_message})

    messages.append({"role": "user", "content": message})

    text = ""

    # response = [{"choices": [{"delta": {"content": "text1"}}]}, {"choices": [{"delta": {"content": "text2"}}]}]
    response = llm.create_chat_completion(
         messages=messages, stream=True
    )
    for chunk in response:
        content = chunk["choices"][0]["delta"]
        if "content" in content:
            text = content["content"]
            yield text


# message="你是一只可爱的小猫咪"

# history = [["", ""]]

# for out in predict(message, history):
#     print(out, end='')