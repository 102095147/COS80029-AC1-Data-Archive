import sys
from openai import OpenAI
import os
from dotenv import load_dotenv
import tokenizer
import json
from datetime import datetime

# Load API key from .env file
load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')

GENERATION_TEMPLATE = """You are a helpful assistant who classifies the relation between two entities in a sentence and follows the instructions below:
1. Return JSONL valid text in the format: {"sentence": "", "head":"", "tail": "", "relation": ""}
2. Every sentence must be included
3. Classify the relation between two entities for each sentence from the list of relations
4. If no relevant relation exists for classification, use NO_RELATION instead
5. If a sentence is incomplete, complete the sentence with context extrapolated from the text
6. There must not be any other text except JSONL

Relations: 
"""

def openai_generate_relations(text, relations, filename):
    # Set up OpenAI API client
    client = OpenAI()

    print('Waiting for response from OpenAI GPT...')
    prompt = GENERATION_TEMPLATE
    prompt = prompt + '[' + ', '.join([f'`{rel.strip()}`' for rel in relations]) + ', `UNKNOWN`]'
    #print(prompt)
    # Make a call to OpenAI's chat API
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": prompt },
            {"role": "user", "content": text}
        ],
    )

    # Get the generated response
    reply = response.choices[0].message.content

    file_path = 'outputs/' + datetime.now().strftime("%d-%m-%Y_%H.%M.%S")

    with open(f'{file_path}_{filename}_relations.txt', 'w') as f1, open(f'{file_path}_{filename}_no-relations.txt', 'w') as f2:
        for json_entry in reply.split('\n'):
            print('Generated JSON entry: ', json_entry)

            # Format response into JSONL structure supported by OpenNRE
            relation_entry = json.loads(json_entry)
            sentence_tokens = get_token_list(relation_entry['sentence'])
            head_tokens = get_token_list(relation_entry['head'])
            tail_tokens = get_token_list(relation_entry['tail'])
            head_pos = get_indices(sentence_tokens, head_tokens)
            tail_pos = get_indices(sentence_tokens, tail_tokens)

            relation_dict = {
                'token': sentence_tokens,
                'h': {
                    'name': relation_entry['head'].lower(),
                    'pos': head_pos
                },
                't': {
                    'name': relation_entry['tail'].lower(),
                    'pos': tail_pos
                },
                'relation': relation_entry['relation']
            }

            # Store sentences without relations in a seperate file
            if relation_entry['relation'] != 'NO_RELATION':
                f1.write(json.dumps(relation_dict) + '\n')
            else:
                f2.write(json.dumps(relation_dict) + '\n')

        print(f'\n\n==== Created JSONL file {file_path}_relations.txt with generated relations training set ====\n')


def get_token_list(sentence):
    tokens = []
    for token in tokenizer.tokenize(sentence):
        kind, txt, val = token
        if txt != '':
            tokens.append(txt)
    return tokens

def get_indices(sentence_tokens, entity_tokens):
    sub_len = len(entity_tokens)
    for i in range(len(sentence_tokens)):
        if sentence_tokens[i:i+sub_len] == entity_tokens:
            return (i, i+sub_len-1)
    return (-1, -1)  # return (-1, -1) if entity is not in sentence

def main():
    try:
        print('Reading relation file....')
        relationListTextFile = 'relations.txt'
        with open(relationListTextFile, 'r') as f:
            relation_list = f.readlines()
        
        directory = 'inputs'
        print('Looping through text input files....')
        for filename in os.listdir(directory):
            if filename.endswith('.txt'):
                with open(os.path.join(directory, filename), 'r') as f:
                    print(f'Extracting relations from file {filename}')
                    openai_generate_relations(f.read(), relation_list, filename)

    except FileNotFoundError:
        print('The text input file could not be found.')
        sys.exit(1)


if __name__ == '__main__':
    main()