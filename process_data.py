from tqdm import tqdm
import json


def process_raw_data(args, tokenizers, max_length):
    with open(args.raw_file_path, 'r', encoding='utf8') as fp:
        data = fp.read()
    train_tokens = []
    for turn in tqdm(data.split('\n\n'), desc='covert to train token'):
        token_ids = [tokenizers.convert_tokens_to_ids('[CLS]')]
        for i, utter in enumerate(turn.split('\n')):
            token = tokenizers.encode(utter, add_special_tokens=False) + [tokenizers.convert_tokens_to_ids("[SEP]")]
            token_ids.extend(token)
        if max_length:
            token_ids = token_ids[:max_length]
        train_tokens.append(token_ids)

    with open(args.train_tokenized_path, 'w', encoding='utf8') as fp:
        json.dump(train_tokens, fp)


def load_train_data(args):
    with open(args.train_tokenized_path, 'r', encoding='utf8') as fp:
        data = json.load(fp)
    return data
