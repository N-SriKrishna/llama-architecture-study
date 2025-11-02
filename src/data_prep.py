"""
data_prep.py
Llama Architecture Implementation
"""

# Data Preparation 

df1 = pd.read_excel("datasets/Inshorts Cleaned Data.xlsx").dropna(subset=['Headline','Short']).rename(columns={'Headline':'target_txt','Short':'input_txt'})[['target_txt','input_txt']]
df2 = pd.read_csv('datasets/train.csv').rename(columns={'dialogue':'input_txt','summary':'target_txt'})[['target_txt','input_txt']]
df3 = pd.read_csv('datasets/samsum-train.csv').rename(columns={'dialogue':'input_txt','summary':'target_txt'})[['target_txt','input_txt']]

train_df = pd.concat([df1,df2,df3]).reset_index(drop=True)

# Text Cleaning 
def clean_text(text: str) -> str:
    """
    Remove HTML,normalize whitespace,preserve puntuation/numbers/casing
    """
    try:
        # text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9.,!?]+', ' ', text)
        text = text.replace('\r', ' ').replace('\n', ' ')
        text = re.sub(r'\s+', ' ', text)
        return text.strip().lower()
    except:
        return text
    
train_df["target_txt"] = train_df["target_txt"].apply(lambda x: clean_text(x))
train_df["input_txt"] = train_df["input_txt"].apply(lambda x: clean_text(x))

# Combine summary + dialogue and split on whitespace
raw_lens = [
    len(f"{s} {d}".split())
    for s, d in tqdm(zip(train_df["target_txt"], train_df["input_txt"]), total=len(train_df))
]

lens = np.array(raw_lens)

# Print summary stats
def pct(x): return np.percentile(lens, x)

print(f"Total examples    : {len(lens):,}")
print(f"Min / Max words   : {lens.min()} / {lens.max()}")
print(f"Mean ± std        : {lens.mean():.1f} ± {lens.std():.1f}")
print("--- Percentiles (word count per raw text pair) ---")
for p in [50, 90, 95, 98, 99]:
    print(f"{p:>3}% : {pct(p):.0f} words")

MAX_LEN = int(pct(90))

text_pairs = []

for i,j in zip(train_df.input_txt,train_df.target_txt):
    try:
        if len(i.split(" ")+j.split(" ")) < MAX_LEN:       # BOS, SEP,EOS extra
            text_pairs.append((i,j))
    except:
        pass

