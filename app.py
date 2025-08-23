import openai
import os
from dotenv import load_dotenv

load_dotenv()

client = openai.OpenAI()

response = client.responses.create(
    model="gpt-4o",
    input=[
        {"role": "user", "content": "Hello! How can I use the OpenAI API in Python?"}
    ],
    temperature=0.7
)

print(response.choices[0].message.content)
