import string
import inflect
import tensorflow as tf
from nltk.corpus import stopwords
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from nltk import word_tokenize, WordNetLemmatizer
import torch
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import StandardScaler
import numpy as np
import streamlit as st
import pandas as pd
from PIL import Image
import joblib
from transformers import pipeline

# Tokenization of the input text
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Loading Pre-trained BERT Model
model = BertModel.from_pretrained('bert-base-uncased')


#############################
# Preprocessing function
#############################
def preprocess_text(user_input):
    # Check if the element is a list, and join it into a string
    if isinstance(user_input, list):
        user_input = ' '.join(map(str, user_input))

    # Convert to lowercase
    text = user_input.lower()

    # Remove punctuation and special characters
    punctuation_set = set(string.punctuation)
    text = ''.join(char for char in text if char not in punctuation_set)

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.lower() not in stop_words]

    # Lemmatization
    lemma = WordNetLemmatizer()
    lemmat = [lemma.lemmatize(token) for token in tokens]

    # Converting numerical values to words
    inflect_en = inflect.engine()
    text_words = [inflect_en.number_to_words(token) if token.isdigit() else token for token in lemmat]

    # Processed text as a single string
    processed_text = ' '.join(text_words)
    return processed_text


#############################
# feature extraction
#############################
def extract_features(user_input):
    embeddings = []

    for text in user_input:
        tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(text)))

        # Adding the cls and sep Special Tokens
        tokens = ['[CLS]'] + tokens + ['[SEP]']

        # Padding and Truncation of the input text
        max_len = 40
        if len(tokens) < max_len:
            tokens = tokens + ['[PAD]' for _ in range(max_len - len(tokens))]
        else:
            tokens = tokens[:max_len - 1] + ['[SEP]']

        # generating the attention Mask
        attention_mask = [1 if token != '[PAD]' else 0 for token in tokens]

        # Converting Tokens to IDs
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # Converting to PyTorch tensors
        input_ids = torch.tensor([input_ids])
        attention_mask = torch.tensor([attention_mask])

        # extracting Features
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)

        # Extract token-level embeddings
        last_hidden_states = outputs.last_hidden_state
        # Mean Pooling
        mean_pooled_embedding = last_hidden_states.mean(axis=1).numpy()
        # Max Pooling
        max_pooled_embedding = last_hidden_states.max(axis=1).values.numpy()

        # CLS Token Output
        cls_embedding = last_hidden_states[:, 0, :].numpy()

        # Combine embeddings for downstream task
        combined_embeddings = np.concatenate([mean_pooled_embedding, max_pooled_embedding, cls_embedding], axis=1)
        aggregated_embeddings = combined_embeddings.mean(axis=0)
        embeddings.append(aggregated_embeddings)

    embeddings = np.array(embeddings)

    return embeddings


#############################
# creating the Streamlit app
#############################

def main():
    # Loading the model using TensorFlow's load_model
    loaded_model = tf.keras.models.load_model('neural_network_model.h5')
    model = loaded_model

    # loading the label encoder
    labelling = joblib.load('label_encoder.h5')

    # setting an image
    image = Image.open('viktor-forgacs-FcDqdJUM6B4-unsplash.jpg')
    # Streamlit App
    st.title("Sentiment Analysis for Infectious Disease Detection in Sub-Saharan Africa")
    st.image(image, caption='Infectious Disease', width=550)

    # Text input for user to enter data
    user_inputs = st.text_area("Enter multiple texts (one per line):", height=200).split('\n')

    if st.button('Analyse Sentiment'):

        # Preprocess and extract features for each entered text
        if user_inputs:
            processed_inputs = [preprocess_text(text) for text in user_inputs]
            # features_list = [extract_features(text) for text in processed_inputs]
            features_list = extract_features(processed_inputs)
            print(features_list.shape)
            scaler = StandardScaler()
            features_list = scaler.fit_transform(features_list)

            # Making predictions for each text
            predictions_list = [model.predict(np.array([features])) for features in features_list]

            # Inverse transform the predicted labels to get the original class labels
            predicted_labels = labelling.inverse_transform([np.argmax(predictions) for predictions in predictions_list])

            # Displaying sentiment analysis results for each text
            st.write("Sentiment Analysis Results:")
            # Initialize an empty DataFrame
            results_df = pd.DataFrame(columns=["Text", "Sentiment Class", "Probability"])

            # Inside the loop
            for i, predictions in enumerate(predictions_list):
                sentiment_class = np.argmax(predictions)
                probability = predictions[0][sentiment_class]
                results_df = pd.concat([results_df, pd.DataFrame(
                    [{"Text": user_inputs[i], "Sentiment Class": predicted_labels[i], "Probability": probability}])],
                                       ignore_index=True)

            # Aggregate sentiment overview
            st.write("Sentiment Overview:")
            st.write(results_df)


if __name__ == '__main__':
    main()
