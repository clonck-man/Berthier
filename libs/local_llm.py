import json
import requests


def generate(user_input, context, model="mistral-nemo"):
    r = requests.post(
        'http://127.0.0.1:11434/api/generate',
        json={
            'model': model,
            'prompt': user_input,
            'context': context
        },
        stream=True)
    r.raise_for_status()

    for line in r.iter_lines():
        body = json.loads(line)
        respond_part = body.get('response', '')
        print(respond_part, end='', flush=True)

        if 'error' in body:
            raise Exception(['error'])
        if body.get('done', False):
            return body['context']


def get_embedding(context, model="mistral-nemo"):
    r = requests.post(
        'http://127.0.0.1:11434/api/embed',
        json={
            'model': model,
            'input': context
        },
        stream=True)
    r.raise_for_status()

    for line in r.iter_lines():
        body = json.loads(line)
        return body['embeddings']


def main():
    context = []
    while True:
        user_input = input(">")
        context = generate(user_input, context)
        print()


if __name__ == "__main__":
    main()
