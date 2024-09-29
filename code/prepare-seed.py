import openai
import json
import time
from tqdm import tqdm

API_ERROR_OUTPUT = "$ERROR$"
API_MAX_RETRY = 10
API_RETRY_SLEEP = 10

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
                messages=messages,
            )
            output = response.choices[0].message.content
            break
        except Exception as e:
            print(f"Error encountered: {e}")
            time.sleep(API_RETRY_SLEEP)
    return output


with open('./data/3k-prompts-cleaned.jsonl', "r") as file:
    prompts = file.readlines()


for prompt in tqdm(prompts, desc="Processing prompts", unit="prompt"):
    prompt_data = json.loads(prompt)
    query = prompt_data["prompt"]
    
    answer = query_openai(to_openai_msg(query))

    user_msg = {
        "from":"user", 
        "value":query
    }
    bot_msg = {
        "from":"assistant", 
        "value":answer
    }
    seed = {
        "id":prompt_data["id"], 
        "conversations":[user_msg, bot_msg]
    }
    with open('./data/seeds-pool.jsonl', "a") as file:
        file.write(json.dumps(seed) + "\n")