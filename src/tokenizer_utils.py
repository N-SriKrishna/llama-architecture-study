"""
tokenizer_utils.py
Llama Architecture Implementation
"""

from tokenizers import Tokenizer, models, trainers, pre_tokenizers

rng =np.random.default_rng(42)
valid_mask = rng.random(len(train_df)) < 0.05 
train_df_,valid_df = train_df[~valid_mask],train_df[valid_mask]

valid_texts = [f"{s} {d}" for s,d in zip(valid_df["target_txt"],valid_df["input_txt"])]

def train_tokenizer(text_iter,vocab_size: int) -> Tokenizer :
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size,min_frequency=2,special_tokens=["[Unk]"])
    tok.train_from_iterator(text_iter,trainer)
    return tok

def avg_pieces_per_word(tok: Tokenizer,texts) -> float:
    pieces = words = 0
    for t in texts:
        ids   = tok.encode(t).ids
        pieces += len(ids)
        words  += len(t.split())
    return pieces / words

vocab_sizes = [1000,2000,3000,4000,6000,8000,10000,12000,16000,20000]
pieces_per_word, oov_rate = [], []

for k in vocab_sizes:
    print(k)
    train_iter = (f"{s} {d}" for s, d in
                  zip(train_df_["target_txt"], train_df_["input_txt"]))
    tok = train_tokenizer(train_iter, k)

    # 1) average sub-words per word on validation set
    pieces_per_word.append(avg_pieces_per_word(tok, valid_texts))

    # 2) OOV percentage on validation set
    unk_id = tok.token_to_id("[UNK]")
    total = unk = 0
    for t in valid_texts:
        ids   = tok.encode(t).ids
        total += len(ids)
        unk   += sum(id_ == unk_id for id_ in ids)
    oov_rate.append(100 * unk / total)

# ── plot ───────────────────────────────────────────────────────────────
plt.figure(figsize=(7,3))
plt.subplot(1,2,1)
plt.plot(vocab_sizes, pieces_per_word, marker='o')
plt.title("pieces / word"); plt.xlabel("vocab size")

plt.subplot(1,2,2)
plt.plot(vocab_sizes, oov_rate, marker='o')
plt.title("OOV %"); plt.xlabel("vocab size")

plt.tight_layout(); plt.show()

VOCAB_SIZE = 4000

CORPUS_FILE = 'all_texts.txt'
SPECIAL_TOKENS = ['[PAD]', '[UNK]', '[BOS]', '[EOS]','[SEP]']

with open(CORPUS_FILE, 'w', encoding='utf-8') as f:
    for _, row in train_df.iterrows():
        f.write(str(row['target_txt']) + ' ' + str(row['input_txt']) + '\n')

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing

# Initialize and train
tokenizer = Tokenizer(BPE(unk_token='[UNK]'))
tokenizer.pre_tokenizer = Whitespace()
trainer = BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=SPECIAL_TOKENS)
tokenizer.train([CORPUS_FILE], trainer)

# 3) Configure post-processing, padding & truncation ------------------
pad_id = tokenizer.token_to_id('[PAD]')
unk_id = tokenizer.token_to_id('[UNK]')
bos_id = tokenizer.token_to_id('[BOS]')
eos_id = tokenizer.token_to_id('[EOS]')
sep_id = tokenizer.token_to_id('[SEP]')

# Add BOS/EOS around single sequences
tokenizer.post_processor = TemplateProcessing(
    single='[BOS] $A [SEP]',
    pair='[BOS] $A [SEP] $B [EOS]',
    special_tokens=[('[BOS]', bos_id), ('[EOS]', eos_id),('[SEP]', sep_id)],
)

