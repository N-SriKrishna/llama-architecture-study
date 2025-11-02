"""
inference.py
Llama Architecture Implementation
"""

def encode_dialogue(text: str, max_len: int = MAX_LEN) -> tf.Tensor:
    """
    Returns [BOS] dialogue [EOS] *without* any padding.
    """
    tokenizer.no_padding()                         # turn padding off
    tokenizer.enable_truncation(max_length=max_len)
    ids = tokenizer.encode(text).ids               # already BOS/EOS
    return tf.constant(ids, tf.int32)              # 1-D tensor


def generate_summary(
    dialogue_ids: tf.Tensor,        # [BOS] … [EOS]  (no pads)
    model: tf.keras.Model,
    max_new: int = 120,
    temperature: float = 0.1
) -> tf.Tensor:
    """
    Produce [BOS] dialogue [SEP] summary … [EOS]
    """
    prompt = tf.concat([dialogue_ids[:-1], [sep_id]], 0)[None, :]  # add batch dim

    for _ in range(max_new):
        if prompt.shape[1] > MAX_LEN:                # keep within PE range
            break

        logits  = model.predict(prompt,verbose=False)[:, -1, :] / temperature
        next_id = tf.random.categorical(logits, 1, dtype=tf.int32)
        prompt  = tf.concat([prompt, next_id], 1)

        if next_id[0, 0] == eos_id:
            break

    return tf.squeeze(prompt, 0)

def display_text_and_summary(dialogue: str, target: str, model):
    """
    Given a raw dialogue string and a summary model, this function:
      1. Encodes the dialogue into token IDs.
      2. Generates [BOS] dialogue [SEP] summary [EOS].
      3. Splits at [SEP], decodes each part, and prints:
         - "Text: <original dialogue…>"
         - "Summary: <generated summary…>"
    """
    # 1. Encode the dialogue (adds [BOS] … [EOS])
    dlg_ids = encode_dialogue(clean_text(dialogue))

    # 2. Generate the full sequence [BOS] dialogue [SEP] summary [EOS]
    full_ids = generate_summary(dlg_ids, model)

    # 3. Convert to a plain Python list so we can find sep_id
    full_ids_list = full_ids.numpy().tolist()

    # 4. Find the position of sep_id
    sep_index = full_ids_list.index(sep_id)

    # 5. Split into dialogue_part and summary_part
    dialogue_part = full_ids_list[:sep_index]
    summary_part  = full_ids_list[sep_index + 1:]  # skip the sep_id itself

    # 6. Decode each slice
    text_str    = decode_token_ids(dialogue_part)
    summary_str = decode_token_ids(summary_part)

    # 7. Print on two lines
    print("\n Text:", text_str)
    print("\n Genrated Summary:", summary_str)
    print("\n Original Summary:", target)

for j in range(5):
    print(j)
    dialogue = df1['input_txt'][j]
    summary = df1['target_txt'][j]
    display_text_and_summary(dialogue, summary, model)
    print('----------------')


