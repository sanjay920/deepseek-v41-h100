"""Seven small chat checks against a running server."""
import argparse
import json
import urllib.request


def run(url, model):
    def request(messages, maximum=64):
        body = {'model': model, 'messages': messages, 'temperature': 0, 'max_tokens': maximum}
        req = urllib.request.Request(
            url.rstrip('/') + '/v1/chat/completions', data=json.dumps(body).encode(),
            headers={'Content-Type': 'application/json'},
        )
        with urllib.request.urlopen(req, timeout=600) as response:
            return json.load(response)

    cases = [
        ('arithmetic', 'What is 17 times 19? Return only the integer.', '323'),
        ('word_transform', 'Reverse the letters of stressed. Return only the reversed word.', 'desserts'),
        ('code_reasoning', 'What does this Python print?\nprint([x*x for x in range(7) if x%2])\nReturn only the printed list.', [1, 9, 25]),
        ('json_extraction', 'Mira packed 7 boxes. Return only a JSON object with keys name and count. The count must be a number.', {'name': 'Mira', 'count': 7}),
        ('reasoning', 'Five boxes each hold six pens. Seven pens are given away. How many pens remain? Return only the integer.', '23'),
    ]
    results = []
    for name, prompt, expected in cases:
        response = request([{'role': 'user', 'content': prompt}])
        text = response['choices'][0]['message']['content'].strip()
        actual = text
        if isinstance(expected, (dict, list)):
            try:
                actual = json.loads(text)
            except ValueError:
                actual = None
        results.append({'test': name, 'passed': actual == expected, 'output': text})

    response = request([
        {'role': 'user', 'content': 'Remember this session code: cobalt-7319.'},
        {'role': 'assistant', 'content': 'I will remember it.'},
        {'role': 'user', 'content': 'What was the session code? Return only the code.'},
    ])
    text = response['choices'][0]['message']['content'].strip()
    results.append({'test': 'multiturn_recall', 'passed': text == 'cobalt-7319', 'output': text})

    lines = [f'Inventory record {i:03d}: item pine-{i:03d} contains {i%13+1} sealed packages.' for i in range(100)]
    lines.insert(40, 'The override password for this inventory is VXQ-4827-LIME.')
    prompt = 'Read the inventory below.\n' + '\n'.join(lines) + '\nWhat is the override password? Return only the password.'
    response = request([{'role': 'user', 'content': prompt}], maximum=32)
    text = response['choices'][0]['message']['content'].strip()
    results.append({
        'test': 'retrieval', 'output': text,
        'passed': text == 'VXQ-4827-LIME' and response['usage']['prompt_tokens'] > 1024,
    })
    return {'passed': all(item['passed'] for item in results), 'checks': results}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:30000')
    parser.add_argument('--model', default='/model')
    args = parser.parse_args()
    result = run(args.url, args.model)
    print(json.dumps(result, indent=2))
    if not result['passed']:
        raise SystemExit(1)
