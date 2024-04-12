import sys
from openai import OpenAI
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load API key from .env file
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# inspritation -> https://cloud.google.com/blog/topics/partners/build-intelligent-apps-with-neo4j-and-google-generative-ai
CYPHER_GENERATION_TEMPLATE = """You are an expert Neo4j Cypher translator who understands the text in english and converts it to Cypher strictly based on the Neo4j Schema provided and following the instructions below:
1. Generate Cypher query compatible ONLY for Neo4j Version 5.
2. Please do not use the same variable names for different nodes and relationships in the query.
3. Do not create Nodes and relationships that are already existing.
4. Always enclose the Cypher output inside 3 backticks.
5. Cypher is NOT SQL. So, do not mix and match the syntaxes.
6. Every Cypher query always starts with a CREATE keyword.
7. If no context provided, assume the people are fishermen.
8. Use similar labels for nodes if exist.
"""

def openai_convert_text_to_cypher(text, cypher_queries):
    # Set up OpenAI API client
    client = OpenAI()

    print('Waiting for response from OpenAI GPT...')
    # Make a call to OpenAI's chat API
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": CYPHER_GENERATION_TEMPLATE },
            {"role": "system", "content": "These are the existing Nodes and relationships: " + cypher_queries },
            {"role": "user", "content": text}
        ],
    )

    # Get the generated response
    reply = response.choices[0].message.content
    #print(reply)

    return get_queries(reply)

def get_queries(text):
    # Get the text between 3 backticks
    cypher_output = text.split("```")[1:-1]
    # Split grouped queries between backticks and get an array of queries
    queries = []
    for group in cypher_output:
        for query in group.splitlines():
            stripped_query = query.strip()
            if stripped_query.startswith("CREATE"):
                queries.append(stripped_query)

    return queries

def google_convert_text_to_cypher(text, cypher_queries):
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-pro')

    print('Waiting for response from Google Gemini...')
    prompt = CYPHER_GENERATION_TEMPLATE + f"""
    These are the existing Nodes and relationships: {cypher_queries}

    Convert the following text to Cypher queries:
    {text}
    """
    response = model.generate_content(prompt)
    try:
        return get_queries(response.text)
    except Exception as e:
        print(f'{type(e).__name__}: {e}')
        print(response.prompt_feedback)


def main():
    if len(sys.argv) < 4:
        print("Please use the following command: text_to_cypher.py [openai|google] [text.txt] [database.txt]")
        sys.exit(1)

    ai = sys.argv[1]
    inputTextFile = sys.argv[2]
    databaseTextFile = sys.argv[3]

    try:
        print('Reading text files....')
        with open(inputTextFile, 'r') as f1, open(databaseTextFile, 'r') as f2:
            text1 = f1.read()
            text2 = f2.read()
            if (ai == 'openai'):
                queries = openai_convert_text_to_cypher(text1, text2)
            elif (ai == 'google'):
                queries = google_convert_text_to_cypher(text1, text2)
            else:
                print('Ai argument must be openai or google')
                return
            
            print('\n\n======= Generated Cypher Queries ======')
            for query in queries:
                print(query)

    except FileNotFoundError:
        print("One or both of the files could not be found.")
        sys.exit(1)


if __name__ == "__main__":
    main()