from openai import OpenAI
md = OpenAI(
            api_key= "sk-rfCIGhxrzcdsMV4jC17e406bE56c47CbA5416068A62318D3",
            base_url=f"http://ipxx.chat.gpt:3006/v1"
        )
response = md.chat.completions.create(
                model="gemini-2.5-flash",
                messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello"},
                ]
            }],
            temperature=0
            )
print(response)