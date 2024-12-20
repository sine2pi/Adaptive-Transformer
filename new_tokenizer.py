from datasets import load_dataset
from transformers import WhisperTokenizerFast
from collections import Counter
import matplotlib.pyplot as plt
import re
import neologdn
from tqdm import tqdm

def custom_normalize(text):
    text = neologdn.normalize(text, repeat=1)
    text = re.sub(r'\s+', '', text)
    text = re.sub(r'[A-Za-z]+', '', text)
    text = re.sub(r'[0-9]+', '', text)
    text = re.sub(r'[、。！？：；「」（）【】『』〈〉《》〔〕［］｛｝]', '', text)
    text = ''.join([char for char in text if re.match(r'[ぁ-んァ-ン一-龥]', char)])
    return text

input_file = "D:/newproject/combined_japanese_corpus.txt"
output_file = "D:/newproject/combined_japanese_corpus2.txt"

chunk_size = 1024 * 1024 

with open(input_file, mode='r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    while True:
        chunk = infile.read(chunk_size)
        if not chunk:
            break
        
        if chunk[-1] != '\n':
            remainder = infile.readline()
            chunk += remainder
        
        processed_chunk = '\n'.join(custom_normalize(text=line) for line in chunk.splitlines())
        
        outfile.write(processed_chunk + '\n')

print("Processing complete. The cleaned file is saved.")

def data_generator(dataset, batch_size):
    batch = []
    for i, example in enumerate(iterable=dataset):
        if bool(example['sentence']):
            normalized_sentence = custom_normalize(example['sentence'])
            batch.append(normalized_sentence)
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch

data = output_file
dataset = load_dataset(path="text", data_files=data)["train"].to_iterable_dataset(num_shards=2000)

batch_size = 100
generator = data_generator(dataset=dataset, batch_size=batch_size)

tokenizer = WhisperTokenizerFast.from_pretrained(pretrained_model_name_or_path="D:/newproject/tokenizers/new/new_tokenizer")
special_tokens = [
    "<CRYING>", "<SINGING>", "<LAUGHING>", "<APPLAUSE>", "<MUSIC>", "<PAD>", "<UNK>", "<BOS>", "<EOS>", "<MASK>", "<NOISE>", "<CLS>", "<END>", "<START>"
]
tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})

oov_count = Counter()
total_count = 0

for batch in generator:
    tokens = [tokenizer.tokenize(text) for text in batch]
    oov_count.update([token for sublist in tokens for token in sublist if token == "<|endoftext|>"])
    total_count += sum(len(sublist) for sublist in tokens)

oov_rate = (sum(oov_count.values()) / total_count) * 100
print(f"OOV Rate: {oov_rate:.2f}%")

token_counts = Counter([token for sublist in tokens for token in sublist])
most_common_tokens = token_counts.most_common(n=50)

tokens, counts = zip(*most_common_tokens)
plt.bar(x=tokens, height=counts)
plt.xlabel(xlabel='Tokens')
plt.ylabel(ylabel='Frequency')
plt.title(label='Top 20 Tokens Frequency Distribution')
plt.show()
