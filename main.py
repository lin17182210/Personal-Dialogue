import os
import random
from datetime import datetime
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import train_test_split
from transformers import GPT2Config, GPT2LMHeadModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from process_data import process_raw_data, load_train_data
from data_set import DialogueDataset
import logging

pad_id = 0


def get_logger():
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    console = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    log.addHandler(console)
    return log


def config_parse():
    parse = argparse.ArgumentParser()
    # data
    data_arg = parse.add_argument_group('Data')
    data_arg.add_argument('--raw_file_path', type=str, default='', help='原始数据集 eg: ./data/resource/train.txt',
                          required=False)
    data_arg.add_argument('--train_tokenized_path', type=str, default='./data/train_tokenized.txt', required=False,
                          help='原始语料tokenizer之后')
    data_arg.add_argument('--vocab_path', type=str, default='', help='词典文件 eg: ./vocab/vocab.txt', required=False)
    data_arg.add_argument('--tensorboard_summary', type=str, default='./tensorboard_summary', help='Tensorboard路径')
    data_arg.add_argument('--model_config', type=str, default='./data/resource/model_config.json', help='模型配置json文件')

    # model
    model_arg = parse.add_argument_group('Model')
    model_arg.add_argument('--use_cuda', type=bool, default=False, help='是否使用cuda加速')
    model_arg.add_argument('--pretrained_model', type=str, default='./model', required=False, help='预训练模型 eg: ./model ')
    model_arg.add_argument('--pretrained_tokenizer_model', type=str, default='bert-base-chinese', required=False,
                           help='预训练tokenizer模型')
    model_arg.add_argument('--output_path', type=str, default='./model', help='模型保存位置')

    # train
    train_arg = parse.add_argument_group('Train')
    train_arg.add_argument('--seed', type=int, default=None, help='训练随机数种子', required=False)
    train_arg.add_argument('--epoch', type=int, default=2, help='训练批次', required=False)
    train_arg.add_argument('--batch_size', type=int, default=16, help='每批次训练样本数量', required=False)
    train_arg.add_argument('--lr', type=float, default=1e-5, help='学习率', required=False)
    train_arg.add_argument('--gradient_accumulation', type=int, default=1, help='梯度积累')
    train_arg.add_argument('--max_gradient_norm', type=float, default=1.0)
    train_arg.add_argument('--num_warmup_steps', type=int, default=500, help='warm up step')
    train_arg.add_argument('--do_train', type=bool, required=False, help='Train model', default=True)
    train_arg.add_argument('--do_eval', type=bool, required=False, help='Evaluate model', default=False)
    train_arg.add_argument('--log_step', type=int, default=1, help='更新tensorboard频率')
    config = parse.parse_args()
    return config


def set_random_seed(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)


def check_model_parameters(model):
    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    logger.info(f'{model.__class__.__name__} total parameters: {num_parameters}')


def load_pretrained_model(args):

    if args.pretrained_model:
        logger.info(f'loading pretrained model from {args.pretrained_model}')
        model = GPT2LMHeadModel.from_pretrained(args.pretrained_model)
    else:
        logger.info('init pretrained model...')
        config = GPT2Config.from_json_file(args.model_config)
        model = GPT2LMHeadModel(config)
    return model, model.config.to_dict().get("n_ctx")


def calculate_loss_and_accuracy(predict, label, device):
    logits = predict[0]
    shift_logits = logits[..., : -1, :].contiguous()
    shift_labels = label[..., 1:].contiguous().to(device)
    loss_function = CrossEntropyLoss(ignore_index=pad_id, reduction='sum')
    loss = loss_function(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    _, pred = shift_logits.max(dim=-1)
    not_ignore = shift_labels.ne(pad_id)  # 非运算
    num_target = not_ignore.long().sum().item()  # 非padding的数量
    correct = (shift_labels == pred) & not_ignore
    accuracy = correct.float().sum() / num_target
    loss = loss / num_target
    return loss, accuracy


def collate_fn(batch):
    global pad_id
    input_ids = []
    max_input_len = max([len(_) for _ in batch])
    for ipt in batch:
        input_len = len(ipt)
        ipt.extend([pad_id] * (max_input_len - input_len))
        input_ids.append(ipt)
    return torch.tensor(input_ids, dtype=torch.long)


def train(model, device, data, args):
    dataset = DialogueDataset(data)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size, collate_fn=collate_fn)
    total_steps = int(len(dataloader) * args.epoch / args.batch_size)
    model.to(device)
    model.train()
    opt = AdamW(model.parameters(), lr=args.lr, correct_bias=True)
    opt.zero_grad()
    scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=args.num_warmup_steps,
                                                num_training_steps=total_steps)
    summary = SummaryWriter(log_dir=args.tensorboard_summary)
    oom_time = 0
    run_loss = 0
    overall_step = 0
    epoch_step = int(total_steps / args.epoch)
    logger.info(f"starting train model: {datetime.now()}")
    for epoch in range(args.epoch):
        start_time = datetime.now()
        for batch_index, batch in enumerate(dataloader):
            input_idx = batch.to(device)
            try:
                outputs = model.forward(input_idx)
                loss, accuracy = calculate_loss_and_accuracy(outputs, input_idx, device)
                loss.backward()
                # 防止梯度爆炸
                torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=args.max_gradient_norm)
                if (batch_index + 1) % args.gradient_accumulation == 0:
                    run_loss += loss
                    opt.step()
                    opt.zero_grad()
                    scheduler.step()
                    overall_step += 1
                    # 更新tensorboard
                    if overall_step % args.log_step == 0:
                        logger.info(f'batch: {batch_index + 1}/{epoch_step}, epoch: {epoch}/{args.epoch}, '
                                    f'loss: {loss}, accuracy: {accuracy}')
                        summary.add_scalar('loss', loss.item(), overall_step)

            except RuntimeError as e:
                if "out of memory" in str(e):
                    oom_time += 1
                    logger.warning(f"oom time: {oom_time}, {datetime.now()}")
                    if hasattr(torch.cuda, "empty_cache"):
                        torch.cuda.empty_cache()
                else:
                    logger.warning(str(e))
                    raise e

        logger.info(f"start save model of epoch [{epoch+1}]: {datetime.now()}")
        model_path = os.path.join(args.output_path, f'epoch_{epoch+1}')
        if not os.path.exists:
            os.mkdir(model_path)
        model_save = model.module if hasattr(model, 'module') else model
        model_save.save_pretrained(model_path)
        logger.info(f'{epoch+1} epoch finished')
        finish_time = datetime.now()
        logger.info(f'time for one epoch: {finish_time - start_time}')
    logger.info('finished all epoch')


def evaluate(model, device, data, args):
    model.eval()
    dataset = DialogueDataset(data)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size, collate_fn=collate_fn)
    epoch_step = int(len(dataloader) / args.batch_size)
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            input_ids = batch.to(device)
            output = model.forward(input_ids)
            loss, accuracy = calculate_loss_and_accuracy(output, label=input_ids, device=device)
            logger.info(f'evaluate {idx+1} / {epoch_step} loss: {loss}, accuracy: {accuracy}')


def main():
    args = config_parse()
    device = 'cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu'
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    model, n_ctx = load_pretrained_model(args)
    if args.vocab_path:
        tokenizer = BertTokenizer(vocab_file=args.vocab_path)
    else:
        tokenizer = BertTokenizer.from_pretrained(args.pretrained_tokenizer_model)
    global pad_id
    pad_id = tokenizer.convert_tokens_to_ids('[PAD]')
    if args.seed:
        set_random_seed(args)
    if args.raw_file_path:
        logger.info("start processing raw data....")
        process_raw_data(args, tokenizer, n_ctx)
    check_model_parameters(model)
    raw_token = load_train_data(args)
    train_data, dev_data = train_test_split(raw_token, test_size=.2)
    logger.info(f"raw data: {len(raw_token)}, train_data: {len(train_data)}, dev_data: {len(dev_data)}")
    if args.do_train:
        train(model, device, train_data, args)
    if args.do_eval:
        evaluate(model, device, dev_data, args)


if __name__ == '__main__':
    logger = get_logger()
    main()
