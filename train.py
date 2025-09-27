import os
import json
import mlflow
import mlflow.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------------------
# 1. Hyperparameters
# ---------------------------
vocab_size = 10000
maxlen = 200
embedding_dim = 16
epochs = 3  # Reduced for faster training
batch_size = 64

# ---------------------------
# 2. Load and preprocess data
# ---------------------------
print("Loading IMDB dataset...")
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

print("Padding sequences...")
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

print("Preparing word index...")
# Get and adjust word index (IMDB specific)
word_index = imdb.get_word_index()
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # Out-of-vocabulary
word_index["<UNUSED>"] = 3

# Reverse word index for decoding
reverse_word_index = {v: k for k, v in word_index.items()}

# ---------------------------
# 3. Build model
# ---------------------------
print("Building model...")
model = keras.Sequential([
    keras.layers.Embedding(vocab_size, embedding_dim, input_length=maxlen),
    keras.layers.GlobalAveragePooling1D(),  # Better than Flatten for sequences
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print(model.summary())

# ---------------------------
# 4. Train + MLflow logging
# ---------------------------
print("Starting training with MLflow logging...")

# Set MLflow tracking (local by default)
mlflow.set_tracking_uri("file:./mlruns")  # Local storage
mlflow.set_experiment("imdb-sentiment")

with mlflow.start_run() as run:
    run_id = run.info.run_id
    print(f"✅ Training started. Run ID: {run_id}")

    # Log parameters
    mlflow.log_params({
        "vocab_size": vocab_size,
        "embedding_dim": embedding_dim,
        "maxlen": maxlen,
        "epochs": epochs,
        "batch_size": batch_size,
        "model_type": "Embedding+GlobalAveragePooling"
    })

    # Train model
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1
    )

    # Evaluate
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"✅ Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Log metrics
    mlflow.log_metrics({
        "test_loss": test_loss,
        "test_accuracy": test_accuracy
    })
    
    # Log final training metrics
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    mlflow.log_metrics({
        "final_train_accuracy": final_train_acc,
        "final_val_accuracy": final_val_acc
    })

    # Log model
    mlflow.keras.log_model(
        model,
        "model",
        registered_model_name="IMDBSentimentModel"
    )

    # ---------------------------
    # 5. Save artifacts for deployment
    # ---------------------------
    os.makedirs("model", exist_ok=True)
    
    # Save model in Keras format
    model.save("model/saved_model.keras")
    
    # Save word index
    with open("model/word_index.json", "w") as f:
        json.dump(word_index, f)
    
    # Save reverse word index for text generation
    with open("model/reverse_word_index.json", "w") as f:
        json.dump(reverse_word_index, f)
    
    # Log artifacts
    mlflow.log_artifacts("model", artifact_path="model")

print("✅ Training complete!")
print("✅ Model saved locally in './model/' directory")
print("✅ Model registered with MLflow as 'IMDBSentimentModel'")