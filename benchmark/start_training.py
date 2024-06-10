import os
import shutil
import subprocess
import time

import pandas as pd

import argparse



def create_input(template_path: str, output_path: str, data_path: str, src_train_path: str, tgt_train_path: str, src_val_path: str, tgt_val_path: str, log_path: str, seed: int):
    # Create nmt yaml
    with open(template_path, 'r') as f:
        template = f.read()

    src_vocab_path = os.path.join(output_path, 'data', 'vocab', 'vocab.src')
    tgt_vocab_path = os.path.join(output_path, 'data', 'vocab', 'vocab.tgt')

    save_model_path = os.path.join(output_path, 'model')

    input_file = template.format(data_path, src_vocab_path, tgt_vocab_path, src_train_path, tgt_train_path, src_val_path, tgt_val_path, log_path, save_model_path, seed)

    input_file_path = os.path.join(output_path, 'input.yaml')
    with open(input_file_path, 'w') as f:
        f.write(input_file)

    return input_file_path

def gen_vocab(log_path, input_file_path):
    # Create vocab
    with open(os.path.join(log_path, 'vocab.log'), 'w') as out:
        subprocess.call(['onmt_build_vocab', '-config', input_file_path, '-n_sample', '-1'], stdout= out, stderr=out)

def main(template_path: str, output_path: str, seed: int):
    
    log_path = os.path.join(output_path, 'logs')
    os.makedirs(log_path, exist_ok=True)

    data_path = os.path.join(output_path, 'data')

    src_test_path = os.path.join(data_path, 'src-test.txt')
    tgt_test_path = os.path.join(data_path, 'tgt-test.txt')
    src_train_path = os.path.join(data_path, 'src-train.txt')
    tgt_train_path = os.path.join(data_path, 'tgt-train.txt')
    src_val_path = os.path.join(data_path, 'src-val.txt')
    tgt_val_path = os.path.join(data_path, 'tgt-val.txt')
    
    # Create input yaml
    print('Creating input...')
    input_file_path = create_input(template_path, output_path, data_path, src_train_path, tgt_train_path, src_val_path, tgt_val_path, log_path, seed)

    # Create vocab files
    print('Creating vocab...')
    gen_vocab(log_path, input_file_path)

    input_file_path = os.path.join(output_path, 'input.yaml')
    
    # Start trainig
    print('Starting training...')
    train_logs_path = os.path.join(log_path, 'train')
    job_log = os.path.join(train_logs_path, 'sub.log')

    os.makedirs(train_logs_path, exist_ok=True)

    with open(job_log, 'w') as out:
        subprocess.call(['onmt_train', '-config', input_file_path], stdout=out, stderr=out)

parser = argparse.ArgumentParser(description='Run transformer training from data creation to inference.')
parser.add_argument('--output_path', required=True, help='Output folder')
parser.add_argument('--template_path', default='./benchmark/transformer_template.yaml', help='Output folder')
parser.add_argument('--seed', default=3435)

args = parser.parse_args()

if __name__ == '__main__':
    main(args.template_path, args.output_path, args.seed)