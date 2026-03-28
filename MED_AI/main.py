from openai import OpenAI

client = OpenAI(
    api_key=OPEN_AI_API,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
user_query=input("user:")
response = client.chat.completions.create(
    model="gemini-2.5-flash",
    response_format={"type": "json_object"},
    messages=[
        
        {
            "role": "user",
            "content": user_query
        }
    ]
)

print(response.choices[0].message.content)
