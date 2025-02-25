import openai

def query_llm(pdf_text, user_query):
    openai.api_key = "YOUR_OPENAI_API_KEY"
    response = openai.Completion.create(
        engine="davinci",
        prompt=f"PDF Text: {pdf_text}\n\nUser Query: {user_query}\n\nResponse:",
        max_tokens=150
    )
    return response.choices[0].text.strip()