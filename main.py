
import time
import numpy as np
from transformers import T5ForConditionalGeneration, AutoTokenizer, BertTokenizer, BertForSequenceClassification
import torch
from tqdm.auto import tqdm, trange
import gc
from gigachat_helper import *
import json


from googletrans import Translator


translator = Translator()

def cleanup():
    """
    A helpful function to clean all cached batches.
    """
    gc.collect()
    torch.cuda.empty_cache()

base_model_name = 'ai-forever/ruT5-base'
model_name = 's-nlp/ruT5-base-detox'
crime_classifier = 'Skoltech/russian-sensitive-topics'
tokenizer = AutoTokenizer.from_pretrained(base_model_name, cache_dir="./models")
model = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir="./models")
model.to('cuda')


classifier_tokenizer = BertTokenizer.from_pretrained(crime_classifier, cache_dir="./models")
classifier_model = BertForSequenceClassification.from_pretrained(crime_classifier, cache_dir="./models")


with open("id2topic.json") as f:
    target_vaiables_id2topic_dict = json.load(f)

def paraphrase(text, model, n=None, max_length='auto', temperature=0.0, beams=3):
    texts = [text] if isinstance(text, str) else text
    inputs = tokenizer(texts, return_tensors='pt', padding=True)['input_ids'].to(model.device)
    if max_length == 'auto':
        max_length = int(inputs.shape[1] * 1.2) + 10
    result = model.generate(
        inputs,
        num_return_sequences=n or 1,
        do_sample=False,
        temperature=temperature,
        repetition_penalty=3.0,
        max_length=max_length,
        bad_words_ids=[[2]],  # unk
        num_beams=beams,
    )
    texts = [tokenizer.decode(r, skip_special_tokens=True) for r in result]
    if not n and isinstance(text, str):
        return texts[0]
    return texts

system_prompt = "Я хочу, чтобы ты выступил в роли ДенВота, веселого и остроумного русского бота, созданного Atm4x. Твоя главная цель — распространять позитив и вызывать смех. Отвечай пользователям остроумными, добрыми и смешными замечаниями, всегда избегая негатива. Используй простые русские фразы, когда это уместно, чтобы усилить свою привлекательность. Помни, что твой создатель Atm4x задумал тебя, чтобы ты вызывал у людей улыбку и прогонял печаль! Когда пользователь выражает негатив, умело переводи разговор на шутку и доброту на русском языке."

def check_for(text, is_pred = False):
    tokenized = classifier_tokenizer.batch_encode_plus([text], max_length=512,
                                            pad_to_max_length=True,
                                            truncation=True,
                                            return_token_type_ids=False)
    tokens_ids, mask = torch.tensor(tokenized['input_ids']), torch.tensor(tokenized['attention_mask'])
    with torch.no_grad():
        model_output = classifier_model(tokens_ids, mask)
        for y_c in model_output['logits']:
            index = str(int(np.argmax(y_c)))
            y_c = target_vaiables_id2topic_dict[index]
        return y_c

messages = [Message(content=system_prompt, role='system').to_json()]

r2 = translator.detect('Привет')
inp = ''
while inp != 'exit':
    user = input('Имя пользователя: ')
    inp = input('Что вы хотите сказать пупсу: ')
    start_time = time.time()
    r1 = translator.detect(inp)
    inp = translator.translate(inp, src=r1.lang, dest=r2.lang).text
    user_translated = translator.translate(inp, src=r1.lang, dest=r2.lang).text
    result = paraphrase([user_translated, inp], model, temperature=50.0, beams=10)

    message = "User: " + result[1]

    if result[1] != inp:
        print("detoxified: ", message)

    checked = check_for(message, True)
    print('Обнаружено: ', checked)
    if checked != 'none':
        continue

    messages.append(Message(content=message, role='user').to_json())
    res = request(messages)
    print(res.json())
    answer = res.json()['choices'][0]['message']['content']
    if res.json()['choices'][0]['finish_reason'] == 'blacklist':
        messages = [Message(content=system_prompt, role='system').to_json()]
    print("DenVot: ", answer)
    messages.append(Message(content=answer, role='assistant').to_json())
    end_time = time.time()
    print("time:", round(end_time-start_time, 2), "seconds")