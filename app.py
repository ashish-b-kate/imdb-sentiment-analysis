import gradio as gr
import tensorflow as tf
import json
import numpy as np
import os

print("🚀 Starting IMDB Sentiment Analysis App...")

# ---------------------------
# 1. Configuration
# ---------------------------
MAX_LEN = 200
VOCAB_SIZE = 10000

# ---------------------------
# 2. Load model and artifacts
# ---------------------------
def load_resources():
    """Load model and word index with error handling"""
    try:
        # Try loading the local model first
        print("Loading model from local directory...")
        model = tf.keras.models.load_model("model/saved_model.keras")
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading local model: {e}")
        print("Please run train.py first to train and save the model.")
        return None, None
    
    try:
        # Load word index
        with open("model/word_index.json", "r") as f:
            word_index = json.load(f)
        print("✅ Word index loaded successfully!")
        return model, word_index
    except Exception as e:
        print(f"❌ Error loading word index: {e}")
        return model, None

model, word_index = load_resources()

if model is None:
    print("❌ Failed to load model. Please train the model first.")
    exit(1)

# ---------------------------
# 3. Text preprocessing
# ---------------------------
def decode_review(encoded_review):
    """Decode encoded review back to text (for debugging)"""
    if not word_index:
        return "Word index not available"
    
    reverse_word_index = {v: k for k, v in word_index.items()}
    return ' '.join([reverse_word_index.get(i, '?') for i in encoded_review if i not in [0, 1, 2, 3]])

def preprocess_text(text, max_len=MAX_LEN):
    """Convert text to sequence of integers"""
    if word_index is None:
        raise ValueError("Word index not loaded")
    
    # Clean text
    text = text.lower().replace('<br />', ' ').replace('/', ' ')
    
    # Simple tokenization
    tokens = text.split()
    
    # Convert words to integers using word_index
    # Note: IMDB dataset uses specific indexing (words start from 4)
    sequence = []
    for word in tokens:
        # IMDB specific: word indices are offset by 3
        word_id = word_index.get(word, 2)  # 2 = <UNK> token
        if word_id >= 4:  # Only include actual words (not special tokens)
            sequence.append(word_id)
    
    # Truncate or pad sequence
    sequence = sequence[:max_len]
    sequence = sequence + [0] * (max_len - len(sequence))
    
    return np.array([sequence])

# ---------------------------
# 4. Prediction function
# ---------------------------
def predict_sentiment(text):
    """Predict sentiment for given text"""
    if text.strip() == "":
        return {"Please enter text": 0.0}
    
    try:
        # Preprocess text
        processed_text = preprocess_text(text)
        
        # Make prediction
        prediction = model.predict(processed_text, verbose=0)[0][0]
        
        # Format output
        if prediction > 0.5:
            return {"Positive 😊": float(prediction)}
        else:
            return {"Negative 😞": float(1 - prediction)}
            
    except Exception as e:
        print(f"Prediction error: {e}")
        return {"Error in prediction": 0.0}

# ---------------------------
# 5. Gradio Interface
# ---------------------------
def create_interface():
    """Create and return Gradio interface"""
    
    # Example reviews
    examples = [
        ["This movie was absolutely fantastic! The acting was superb and the story was engaging from start to finish."],
        ["Terrible movie. Waste of time. Poor acting and boring storyline."],
        ["The film had some good moments but overall it was quite average. Not bad, but not great either."],
        ["I loved this movie! The characters were well-developed and the cinematography was stunning."],
        ["Boring and predictable. I fell asleep halfway through."]
    ]
    
    # Create interface
    demo = gr.Interface(
        fn=predict_sentiment,
        inputs=gr.Textbox(
            lines=3,
            placeholder="Enter your movie review here...",
            label="Movie Review"
        ),
        outputs=gr.Label(
            num_top_classes=2,
            label="Sentiment Prediction"
        ),
        examples=examples,
        title="🎬 IMDB Movie Review Sentiment Analysis",
        description=(
            "This AI model analyzes movie reviews and predicts whether they have "
            "**Positive** or **Negative** sentiment. The model was trained on the "
            "IMDB movie review dataset.\n\n"
            "**How to use:** Type a movie review in the text box and click Submit, "
            "or try one of the examples below!"
        ),
        theme="soft"
    )
    
    return demo

# ---------------------------
# 6. Main execution
# ---------------------------
if __name__ == "__main__":
    print("🌐 Starting Gradio web server...")
    
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False  # Set to True if you want a public link
    )