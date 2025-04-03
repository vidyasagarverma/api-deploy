from flask import Flask, request, jsonify
from transformers import MarianMTModel, MarianTokenizer

app = Flask(__name__)

# Load the pre-trained model and tokenizer from Hugging Face (English to German model)
model_name = 'Helsinki-NLP/opus-mt-en-de'  # English to German model
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)

# Function to translate text
def translate_text(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    
    # Perform the translation
    translated = model.generate(**inputs)
    
    # Decode the translated text
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

@app.route('/translate', methods=['POST'])
def translate():
    # Get the JSON data from the request
    data = request.get_json()

    # Get the text to be translated from the request
    text_to_translate = data.get("text", "")

    if not text_to_translate:
        return jsonify({"error": "No text provided for translation"}), 400

    # Translate the text
    translated_text = translate_text(text_to_translate)

    return jsonify({"original_text": text_to_translate, "translated_text": translated_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

