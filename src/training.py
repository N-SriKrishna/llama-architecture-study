"""
training.py
Llama Architecture Implementation
"""

# 2) Repack your dataset for .fit()
# Each element becomes (x, y, sample_weight)
ds_for_fit = ds.map(lambda b: (           # ds originally yields dicts
    b["input_ids"],        # x
    b["labels"],           # y_true
    b["loss_mask"]         # sample_weight: 1.0 on summary tokens, 0 elsewhere
))

model = Transformer(
    num_layers=4,
    d_model=256,
    num_heads=8,
    input_vocab_size=VOCAB_SIZE,
    max_len=MAX_LEN, 
    dropout_rate=0.1
)

# 3) Compile with a standard sparse‐CE loss and let Keras use sample weights
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none"),
    metrics=["sparse_categorical_accuracy"]
)

# Model's output (logits) is: logits = self.final_layer(x)  # (batch_size, target_len, target_vocab_size).
# This is a tensor of logits, where for each position in the sequence, you have a score for every possible token in your vocabulary.
# Because labels are integer token IDs (e.g., [10, 54, 3, 0, 0]) and not one-hot encoded vectors (e.g., [[0..1..0], [0..1..0], ...]), SparseCategoricalAccuracy is the appropriate metric.
# It correctly compares the integer label at each timestep with the class that has the highest logit output by the model for that timestep.

# Build callbacks
callbacks = [
    EarlyStopping(monitor="loss", patience=3,
                  restore_best_weights=True, verbose=1),
    ModelCheckpoint(
        filepath="best_summary.keras",        # or "best_summary.h5"
        monitor="loss",
        save_best_only=True,
        verbose=1             # full model (weights + optimizer + LR schedule)
    )
]

# 4) Fit!  Keras will print epoch/step progress by default
history = model.fit(
    ds_for_fit,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1    # 1 = progress bar, loss & acc per epoch
)

