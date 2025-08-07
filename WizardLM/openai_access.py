import openai
from openai import OpenAI, AsyncOpenAI
import time
from Utilis.utilis import img2base64
# api_key = "sk-85caf631866d4b0fb79eaeeb34f8f96e"
# model = "qvq-max-2025-03-25"
# qwen_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# model = "qvq-max-2025-03-25"
ark_url = "https://ark.cn-beijing.volces.com/api/v3"
api_key = "ecb26efc-05e7-4d58-8d40-0dca61ccb4e9" # my own api
model = "doubao-1-5-thinking-vision-pro-250428"
#model = "doubao-lite-128k-240828"
openai.api_key = api_key
async def get_oai_completion(prompt, image_path=None):
    if image_path!= None:
        img_encoded_format = img2base64(image_path)
        messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text":  "你是Qwen, 是阿里开发的大模型助手"}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
              
            ]+[ {"type": "image_url", "image_url": {"url": img_encoded_format}} ] if image_path is not None else []
        }
    ]
    try: 
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=ark_url,
        )
       
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=1,
            max_tokens=2048,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            stream=True,
        )
        #res = response["choices"][0]["message"]["content"]
        answer = ""
        reasoning_content = ""
        async for chunk in response:
            if not chunk.choices:
                print("\nUsage:")
                print(chunk.usage)
            else:
                delta = chunk.choices[0].delta
                # 打印思考过程
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
                    reasoning_content += delta.reasoning_content
                else:
                    answer += delta.content
       
       
        return answer, reasoning_content
    except Exception as e:
        raise e
       
async def call_chatgpt(ins,image_path=None):
    success = False
    re_try_count = 15
    ans = ''
    while not success and re_try_count >= 0:
        re_try_count -= 1
        try:
            ans = await get_oai_completion(ins, image_path=image_path)
            success = True
        except Exception as e:
            raise e
    return ans