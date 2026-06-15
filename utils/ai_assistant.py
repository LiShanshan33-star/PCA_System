from openai import OpenAI

from config import *


client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL
)


def ask_ai(
        question,
        context=""
):

    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {
                "role":"system",
                "content":
                "你是一位六西格玛和过程能力分析专家。"
            },
            {
                "role":"user",
                "content":
                f"""
                分析结果：

                {context}

                用户问题：

                {question}
                """
            }
        ]
    )

    return (
        response
        .choices[0]
        .message
        .content
    )