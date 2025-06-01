from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse
from tqdm import tqdm
from torch.nn.functional import softmax
import math
import re
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def create_messages_one_pass(topics, instruction_type='refine_labelTokenProbs', topic_n=2, words_n=10):
    all_messages = []

    for topic in topics:
        sys_role = 'You are a helpful assistant in understanding topics and words.'

        ############## instruction ##############
        #########################################
        if instruction_type in ['refine_labelTokenProbs', 'refine_wordIntrusion', 'refine_seqLike']:
            instruction = '''Analyze step-by-step and provide the final answer. \nStep 1. Given a set of words, summarize a topic (avoid using proper nouns as topics) by %s words that covers most of those words. (Note, only the topic, no other explanations.)\nStep 2. Remove irrelevant words about the topic from the given word list. (Note, only the removed words, no other explanations.)\nStep 3. Add new relevant words (maximun 10 words) about the topic to the word list up to %s words. (Note, only the added words, no other explanations.)\nStep 4. Provide your answer in json format as {'Topic': '<%s Word Topic>', 'Words': '<Refined %s Word List>'}. Note, only %s refined words allowed for the topic, and no follow up explanations. Do use [] for the word list.''' % (topic_n, words_n, topic_n, words_n, words_n)

        elif instruction_type == 'refine_askConf':
            instruction = '''Analyze step-by-step, provide the final answer and your confidence of solving the problem. \nStep 1. Given a set of words, summarize a topic (avoid using proper nouns as topics) by %s words that covers most of those words. (Note, only the topic, no other explanations.)\nStep 2. Remove irrelevant words about the topic from the given word list. (Note, only the removed words, no other explanations.)\nStep 3. Add new relevant words (maximum 10 words) about the topic to the word list up to %s words. (Note, only the added words, no other explanations.)\nStep 4. Provide your answer in json format as {'Topic': '<%s Word Topic>', 'Words': '<Refined %s Word List>', 'Confidence': '<Your confidence of solving the problem, numeric value from 0-100>'}. Note, only %s refined words allowed for the topic, and no follow up explanations. Do use [] for the word list.''' % (topic_n, words_n, topic_n, words_n, words_n)
        #########################################
        #########################################

        messages = []
        messages.append({"role": "system", "content": sys_role + ' ' + instruction})
        messages.append({"role": "user", "content": ', '.join(topic)})
        all_messages.append(messages)

    return all_messages


def create_messages_two_steps(topics, reference_answers, instruction_type='refine_twoStep_Score'):
    all_messages = []

    for topic, answer in zip(topics, reference_answers):
        sys_role = 'You are a helpful assistant in understanding topics and words.'

        if instruction_type == 'refine_twoStep_Score':
            instruction = '''Analyze the topic and words then provide the answer in JSON format. \nGiven a set of words and a reference topic, how well the reference topic cover the words in the list? Provide a numeric score range from 0-100 to show the coverage of the topic to the words. Higher socre indictaes better coverage. \nFormat your answer in json format as {'Coverage Score': '<Numeric value from 0-100>'}. No follow up explanations.'''

        elif instruction_type == 'refine_twoStep_Boolean':
            instruction = '''Read the question and provide the answer in JSON format without explanation. \nGiven a set of words and a reference topic, does the reference topic cover most the words in the list? \nProvide your answer in json format as {'Answer': '<YES or NO>'}. Only use 'YES' or 'NO' for your answer, no follow up explanations.'''

        messages = []
        messages.append({"role": "system", "content": sys_role + ' ' + instruction})
        messages.append({"role": "user", "content": "Words: [" + ', '.join(topic) + "]. Reference Topic: " + answer + "."})
        all_messages.append(messages)

    return all_messages


def extract_text_between_strings(text, string1, string2):
    pattern = re.escape(string1) + r'(.*?)' + re.escape(string2)
    matches = re.findall(pattern, text, re.DOTALL)
    return matches


def parse_answer(answer, instruction_type):
    answer = answer.replace('"', "'")
    try:
        topic = extract_text_between_strings(answer, "'Topic':", ',')[0].strip()
        if topic.startswith("'"):
            topic = topic[1:]
        if topic.endswith("'"):
            topic = topic[0:-1]
        topic = topic.replace(".", "")

        words = extract_text_between_strings(answer, "'Words': [", ']')[0].strip()
        words = words.replace("'", '').split(',')
        words = [w.strip() for w in words]

        conf = None
        if instruction_type == 'refine_askConf':
            conf = extract_text_between_strings(answer, "'Confidence':", '}')[0].strip()
            conf = conf.replace("'", '')

        ans_dict = {'Topic': topic, 'Words': words, 'Conf': conf}

    except:
        print('Error when parsing answer:')
        print(answer)
        ans_dict = None

    return ans_dict


def get_column_by_index(matrix, col_index):
    return [row[col_index] for row in matrix]


def batch_generator(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def get_phrase_probability(token_probs, target_phrase):
    # Remove spaces and lowercase both for comparison
    target_clean = target_phrase.replace(" ", "").lower()
    results = []
    n = len(token_probs)

    for start in range(n):
        phrase = ""
        prob = 1.0
        for end in range(start, n):
            phrase += token_probs[end]['token']
            prob *= float(token_probs[end]['prob'])  # make sure prob is a float
            phrase_clean = phrase.replace(" ", "").lower()
            if phrase_clean == target_clean:
                results.append({'start': start, 'end': end, 'prob': prob})
                break
            if not target_clean.startswith(phrase_clean):
                break  # early stop
    return results


def generate_one_pass(model, tokenizer, topics, voc, instruction_type='refine_labelTokenProbs', topic_n=2, batch_size=5, max_new_tokens=300):
    messages = create_messages_one_pass(topics, instruction_type, topic_n)
    message_batches = batch_generator(messages, batch_size)

    n_batches = math.ceil(len(messages) / batch_size)
    topic_probs_list = []
    word_probs_list = []

    print('Running LLM Feedback ...')
    for messages in tqdm(message_batches, total=n_batches):
        encodeds = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", truncation=False, padding=True).cuda()
        generated_outputs = model.generate(encodeds,
                                 pad_token_id=tokenizer.pad_token_id,
                                 max_new_tokens=max_new_tokens,
                                 do_sample=False, # greedy sampling
                                 num_return_sequences=1,
                                 temperature=None,
                                 #temperature=temperature,  # work only when do sampling is true
                                 top_p=None,
                                 output_scores=True,
                                 return_dict_in_generate=True)

        # generated id and score
        batch_generated_ids = generated_outputs.sequences.cpu().numpy()
        batch_scores = generated_outputs.scores

        n_opt = batch_generated_ids.shape[0]
        input_length = encodeds.shape[1]

        for i in range(n_opt):
            # generated text
            generated_sequence = batch_generated_ids[i]
            new_tokens = generated_sequence[input_length:]  # Exclude the input tokens
            decoded_sequence = tokenizer.decode(new_tokens, skip_special_tokens=True)

            # parse answer
            ans_dict = parse_answer(decoded_sequence, instruction_type)
            if ans_dict is not None:
                topic = ans_dict['Topic']
                words_suggested = ans_dict['Words']
                words_suggested = list(set(words_suggested))

                # word removal as conf
                original_words = messages[i][1]['content'].split(', ')
                count = 0
                for w in original_words:
                    if not w in words_suggested:
                        count += 1
                removal_conf = 1 - count / len(original_words)

                # keep only word in voc
                if voc is not None:
                    words = []
                    for word in words_suggested:
                        try:
                            voc[word.strip().lower()]
                            words.append(word)
                        except:
                            pass
                else:
                    words = words_suggested

                if len(words) == 0:
                    print('all suggested words are oov')
                    topic_prob_dict = None
                    word_probs_dict = None
                    topic_probs_list.append(topic_prob_dict)
                    word_probs_list.append(word_probs_dict)
                    continue

                if instruction_type == 'refine_askConf':
                    topic_prob = float(ans_dict['Conf'])*0.01
                elif instruction_type == 'refine_wordIntrusion':
                    topic_prob = float(removal_conf)
                else:
                    # seq probabilies
                    scores = get_column_by_index(batch_scores, i)
                    seq_probs = []
                    for j in range(len(new_tokens)):
                        token_id = new_tokens[j].item()
                        token = tokenizer.decode(token_id)

                        # get logits and probabilies
                        score = scores[j].reshape(-1) # logits over V
                        token_probs= softmax(scores[j], dim=-1).reshape(-1) # probs over V
                        token_logits = score[token_id].cpu().numpy() # token logit
                        token_prob = token_probs[token_id].cpu().numpy() # token prob

                        id_token_prob = {'pos': j, 'id': token_id, 'token':token, 'logits': token_logits, 'prob': token_prob}
                        seq_probs.append(id_token_prob)

                    if instruction_type == 'refine_seqLike':
                        topic_prob = 0
                        n_token = len(seq_probs)
                        for item in seq_probs:
                            topic_prob += -math.log(item['prob'])
                        topic_prob = topic_prob / n_token
                    else:
                        results = get_phrase_probability(seq_probs, topic)
                        try:
                            topic_prob = results[0]['prob']
                        except:
                            topic_prob_dict = None
                            word_probs_dict = None
                            topic_probs_list.append(topic_prob_dict)
                            word_probs_list.append(word_probs_dict)
                            continue

                topic_prob_dict = {topic: topic_prob}
                word_prob = 1/len(words)
                word_probs_dict = {w: word_prob for w in words}  # assign uniform

            # fail to get a answer
            else:
                topic_prob_dict = None
                word_probs_dict = None

            topic_probs_list.append(topic_prob_dict)
            word_probs_list.append(word_probs_dict)

    return topic_probs_list, word_probs_list


def generate_two_step(model, tokenizer, topics, voc, instruction_type='refine_twoStep_Score', topic_n=2, batch_size=5, max_new_tokens=300):
    # get step one answer
    step_one_topics, step_one_words = generate_one_pass(model, tokenizer, topics, voc, topic_n=topic_n, batch_size=batch_size, max_new_tokens=max_new_tokens)
    none_idxs = [i for i in range(len(step_one_topics)) if step_one_topics[i] is None]
    reference_answers = [list(item.keys())[0] for item in step_one_topics if item is not None]

    # create step two message
    messages = create_messages_two_steps(topics, reference_answers, instruction_type)
    message_batches = batch_generator(messages, batch_size)
    n_batches = math.ceil(len(messages) / batch_size)
    topic_probs_list = []

    print('Running LLM Feedback ...')
    for messages in tqdm(message_batches, total=n_batches):
        encodeds = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", truncation=False, padding=True).cuda()
        generated_outputs = model.generate(encodeds,
                                 pad_token_id=tokenizer.pad_token_id,
                                 max_new_tokens=max_new_tokens,
                                 do_sample=False, # greedy sampling
                                 num_return_sequences=1,
                                 temperature=None,
                                 #temperature=temperature,  # work only when do sampling is true
                                 top_p=None,
                                 output_scores=True,
                                 return_dict_in_generate=True)

        # generated id and score
        batch_generated_ids = generated_outputs.sequences.cpu().numpy()
        batch_scores = generated_outputs.scores

        n_opt = batch_generated_ids.shape[0]
        input_length = encodeds.shape[1]

        for i in range(n_opt):
            # generated text
            generated_sequence = batch_generated_ids[i]
            new_tokens = generated_sequence[input_length:]  # Exclude the input tokens
            decoded_sequence = tokenizer.decode(new_tokens, skip_special_tokens=True)

            # verbalized conf
            if instruction_type == 'refine_twoStep_Score':
                answer = decoded_sequence.replace('"', "'")
                conf = extract_text_between_strings(answer, "'Coverage Score':", '}')[0].strip()
                conf = conf.replace("'", '')
                topic_prob = float(conf)*0.01
            # pTrue conf
            elif instruction_type == 'refine_twoStep_Boolean':
                # seq probabilies
                scores = get_column_by_index(batch_scores, i)
                seq_probs = []
                for j in range(len(new_tokens)):
                    token_id = new_tokens[j].item()
                    token = tokenizer.decode(token_id)

                    # get logits and probabilies
                    score = scores[j].reshape(-1) # logits over V
                    token_probs= softmax(scores[j], dim=-1).reshape(-1) # probs over V
                    token_logits = score[token_id].cpu().numpy() # token logit
                    token_prob = token_probs[token_id].cpu().numpy() # token prob

                    id_token_prob = {'pos': j, 'id': token_id, 'token':token, 'logits': token_logits, 'prob': token_prob}
                    seq_probs.append(id_token_prob)

                result_true = get_phrase_probability(seq_probs, 'yes')
                result_false = get_phrase_probability(seq_probs, 'no')
                if len(result_true) > 0:
                    topic_prob = result_true[0]['prob']
                elif len(result_false) > 0:
                    topic_prob = 1 - result_false[0]['prob']

            topic_probs_list.append(topic_prob)

    # update topics probs and put none answers back
    for idx in none_idxs:
        topic_probs_list.insert(idx, None)
    if len(step_one_topics) != len(topic_probs_list):
        print('something wrong!')
        quit()
    else:
        for i in range(len(step_one_topics)):
            if step_one_topics[i] is not None:
                topic = list(step_one_topics[i].keys())[0]
                step_one_topics[i][topic] = topic_probs_list[i]

    return step_one_topics, step_one_words


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='llama3-8b') # ['llama3-8b']
    parser.add_argument('--inference_bs', type=int, default=5) # the number of topics pass to the LLM at once, increase this number based on your GPU memory
    parser.add_argument('--max_new_tokens', type=int, default=300)
    parser.add_argument('--temperature', type=float, default=0.5) # higher, deverse, scaling logits, only work when do sampling=true
    args = parser.parse_args()

    # model

    # model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
    # model_name = 'mistralai/Mistral-7B-Instruct-v0.3'
    # model_name = '01-ai/Yi-1.5-9B-Chat'
    # model_name = 'microsoft/Phi-3-mini-128k-instruct'
    model_name = 'Qwen/Qwen1.5-32B-Chat'

    # load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 trust_remote_code=True,
                                                 torch_dtype=torch.float16
                                                 ).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    # example topics
    topic1 = ['book', 'university', 'bank', 'science', 'vote', 'gordon', 'surrender', 'intellect', 'skepticism', 'shameful']
    topic2 = ['game', 'team', 'hockey', 'player', 'season', 'year', 'league', 'nhl', 'playoff', 'fan']
    topic3 = ['written', 'performance', 'creation', 'picture', 'chosen', 'clarify', 'second', 'appreciated', 'position', 'card']
    topics = [topic1, topic2, topic3]

    # vocabulary set
    voc = None

    # Difference confidence
    # 'refine_labelTokenProbs': label token prob
    # 'refine_wordIntrusion': word intrusion
    # refine_askConf: ask for confidence
    # 'refine_seqLike': length normalized sequence likelihood
    # 'refine_twoStepBoolean': self-reflective
    # 'refine_twoStep_Boolean': p(True)

    inference_bs = 5
    instruction_types = ['refine_labelTokenProbs', 'refine_wordIntrusion', 'refine_askConf', 'refine_seqLike',
                         'refine_twoStep_Score', 'refine_twoStep_Boolean']
    # generate topics
    for instruction_type in instruction_types:
        print(model_name)
        print(instruction_type)
        if instruction_type in ['refine_labelTokenProbs', 'refine_wordIntrusion', 'refine_askConf', 'refine_seqLike']:
            topic_probs, word_prob = generate_one_pass(model,
                                                       tokenizer,
                                                       topics,
                                                       voc=voc,
                                                       batch_size = inference_bs,
                                                       instruction_type=instruction_type)

        elif instruction_type in ['refine_twoStep_Score', 'refine_twoStep_Boolean']:
            topic_probs, word_prob = generate_two_step(model,
                                                       tokenizer,
                                                       topics,
                                                       voc=voc,
                                                       batch_size=inference_bs,
                                                       instruction_type=instruction_type)

        print(topic_probs)
        print(word_prob)