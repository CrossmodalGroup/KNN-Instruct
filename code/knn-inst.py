from tqdm import tqdm
import json
import time
import openai
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
import torch

system_template_knninst = '''You are a helpful assistant designed to interpret and analyze user queries that have actually been proposed to AI assistant ChatGPT. Based on your analysis, you are capable of further crafting new queries.'''

template_gen_query_knninst_fewshot = """\n\n\nThe passage above showcases several dialogs between a human user and the AI assistant, ChatGPT. As can be seen, real-world user queries are usually meaningful, diversified and grounded, reflecting practical needs, critical thinking or intriguing ideas. Now perform the following task.

### Task: Position yourself as the user in question, and craft a new, high-quality query. Keep the following in mind:
1. Relevance: Incorporate your previous analysis, fully utilize these informative prior, ensuring that the new query aligns well with this user.
2. Originality: The new query should be distinguished to existing ones instead of naive imitation or transfer, so try your best in CREATIVITY;
3. Standalone: The new query should be self-contained and not depend on prior conversations.
4. Format: You should simply return a string as the new query."""

API_ERROR_OUTPUT = "$ERROR$"
API_MAX_RETRY = 10
API_RETRY_SLEEP = 10

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(line)
                print(f'Error: Could not parse line as JSON: {str(e)}')
    return data

def embed(doc, model, tokenizer_):
    with torch.no_grad():
        inputs = tokenizer_(doc, max_length=tokenizer.model_max_length, padding=True, truncation=True, return_tensors="pt").to(model.device)
        embedding = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
    return embedding

def assemble_demos_fewshot(demos):
    
    prompt = ""
    for i in range(len(demos)):

        conv = demos[i]['conversations']

        prompt += f"[Exemplar Dialog {i+1}]"

        for turn in conv:
            if turn['from'] == 'user':
                prompt += f"\n[Human User]: {turn['value']}"
            else:
                if len(tokenizer.encode(prompt) + tokenizer.encode(turn['value'])) > 256:
                    lengthy_response = turn['value'].split(" ")
                    lengthy_response = " ".join(lengthy_response[:int(len(lengthy_response) / 3)])
                    prompt += f"\n[AI Assistant]: {lengthy_response} ... ... (Omit)"
                    break
                else:
                    prompt += f"\n[AI Assistant]: {turn['value']}"

        prompt += f"\n[Exemplar Dialog {i+1} Ends]\n"

    return prompt

    
def make_dialog_prompt(demos):

    messages = [{
        'role': 'system', 'content': system_template_knninst
    }]

    convs = "\"\"\"" + assemble_demos_fewshot(demos) + "\"\"\""
    messages.append({
        'role': 'user',
        'content': convs + template_gen_query_knninst_fewshot
    })

    return messages

def to_openai_msg(prompt):
    return [{
        'role': 'user', 'content': prompt
    }]

def query_openai(messages):
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )
            output = response.choices[0].message.content
            break
        except Exception as e:
            print(f"Error encountered: {e}")
            time.sleep(API_RETRY_SLEEP)

    return output


seed_data = load_jsonl("./data/seeds-pool.jsonl")

# Clustered
tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-roberta-large")
model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-large").to("cuda:0")
embeddings = []

for item in tqdm(seed_data, total = len(seed_data)):
    embeddings.append(embed(item['conversations'][0]['value'], model, tokenizer))

embeddings = torch.cat(embeddings, dim=0)

for i in tqdm(range(len(seed_data))):

    query_embed = embed(seed_data[i]['conversations'][0]['value'], model, tokenizer)
    cosine_similarities = F.cosine_similarity(query_embed, embeddings)
    cosine_distances = 1 - cosine_similarities
    top3_scores, top3_indices = cosine_distances.topk(3, largest=False)
    demo_indices = top3_indices[1:]

    demos = [seed_data[i]] + [seed_data[j] for j in demo_indices]

    messages = make_dialog_prompt(demos)

    new_query = query_openai(messages)

    new_response = query_openai(to_openai_msg(new_query))

    sample = {
        'id': i + len(seed_data), 
        'conversations': [{
            'from': 'user', 
            'value': new_query
        }, {
            'from': 'assistant', 
            'value': new_response
        }]
    }

    with open("./data/seeds-pool.jsonl", 'a') as f:
        f.write(json.dumps(sample, ensure_ascii=False) + '\n')