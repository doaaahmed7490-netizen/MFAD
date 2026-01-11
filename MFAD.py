import json
import numpy as np
import pandas as pd
import re
import nltk
import emoji
import emot as e
import spacy
import textstat

from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
#from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Concatenate, Dropout,GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve, auc

import os

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nlp = spacy.load("en_core_web_sm")

def load_hc3_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    texts, labels = [], []
    for entry in data:
        for h in entry.get("human_answers", []):
        #for h in entry.get("real_abstract", []):
            texts.append(h)
            labels.append(0)
        #for a in entry.get("generated_abstract", []):
        for a in entry.get("chatgpt_answers", []):
    
            texts.append(a)
            labels.append(1)
    
    df = pd.DataFrame({'text': texts, 'label': labels})
    return df

#df = load_hc3_json("AbsDivison1to22.json")
#df = load_hc3_json("Gemini_Open_qa_Only - Copy.json")
#df=load_hc3_json("finance.json")
#df=load_hc3_json("open_qa.json")
#df=load_hc3_json("wiki_csai.json")

#df=load_hc3_json("reddit_eli5.json")
#df = load_hc3_json("all-Test.json")
#df = load_hc3_json("MedicineGPT.json")
# 
#df = load_hc3_json("Finance_GPT4.json")

#df = load_hc3_json("Dataset/medicine.json")
df = load_hc3_json("Dataset/RegeneratedDataset/Medicine_GPT4.json")

#df = load_hc3_json("Finance_Gemini.json")

#df = load_hc3_json("Finance_Claude.json")
#df = load_hc3_json("Finance_GPT4.json")
#df = load_hc3_json("MedicineGemini.json")

#df = load_hc3_json("MedicineClaude.json")


def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text.lower())
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove hashtags (keep the word or remove entirely based on need)
    text = re.sub(r'#\w+', '', text)  # remove hashtag and word
    
    # Remove currency symbols and units ($, €, £, ₹, etc.)
    text = re.sub(r'[\$\€\£\₹\¥\₽\₩\¢]+[\d,.]*|\d+[\s]?(USD|EUR|EGP|GBP|JPY|INR)', '', text, flags=re.IGNORECASE)

    # Optional: Remove extra whitespace
    #text = re.sub(r'\s+', ' ', text).strip()
    #text = emoji.demojize(text)
    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
  #  stop_words = set(stopwords.words('english'))
   # tokens = [word for word in tokens if word not in stop_words]

    # Stemming or Lemmatization

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Join back into string
    text = ' '.join(tokens)
    #return processed_text
    return text

df['text_clean'] = df['text'].apply(clean_text)
stop_words = set(stopwords.words('english'))

def extract_features(text):
    special_chars = re.findall(r'[^a-zA-Z0-9\s]', text)
    syllable_count=textstat.syllable_count(text)
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    return [
        len(tokens),
        np.mean([len(t) for t in tokens]) if tokens else 0,
        sum(1 for _, tag in pos_tags if tag.startswith('NN')),
        sum(1 for _, tag in pos_tags if tag.startswith('VB')),
        sum(1 for _, tag in pos_tags if tag.startswith('JJ')),
        sum(1 for t in tokens if t in stop_words) / len(tokens) if tokens else 0,
        #special_chars,
        #syllable_count,
    ]

X_syntactic = np.array([extract_features(t) for t in df['text_clean']])
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['text_clean'])
X_seq = pad_sequences(tokenizer.texts_to_sequences(df['text_clean']), maxlen=300)
word_index = tokenizer.word_index

embedding_index = {}
with open("G:\Code\glove.6B.100d.txt", encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vec = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = vec

embedding_dim = 100
embedding_matrix = np.zeros((len(word_index)+1, embedding_dim))
for word, i in word_index.items():
    if word in embedding_index:
        embedding_matrix[i] = embedding_index[word]
def build_model():
    text_input = Input(shape=(300,))
    meta_input = Input(shape=(X_syntactic.shape[1],))

    embedding = Embedding(input_dim=len(word_index)+1,
                          output_dim=embedding_dim,
                          weights=[embedding_matrix],
                          input_length=300,
                          trainable=False)(text_input)

    conv = Conv1D(128, 5, activation='relu')(embedding)
    pool = MaxPooling1D(pool_size=2)(conv)
    #pool = GlobalMaxPooling1D()(conv)

    bilstm = Bidirectional(LSTM(64))(pool)

    #merged = Concatenate()([bilstm, meta_input])
    merged = Concatenate(name="concat_features")([bilstm, meta_input])

   # merged = Concatenate()([pool, meta_input])
    dense = Dense(64, activation='relu')(merged)
    dropout = Dropout(0.5)(dense)
    output = Dense(1, activation='sigmoid')(dropout)

    model = Model(inputs=[text_input, meta_input], outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
X_train_seq, X_test_seq, X_train_syn, X_test_syn, y_train, y_test = train_test_split(
    X_seq, X_syntactic, df['label'], test_size=0.2, random_state=42)

model = build_model()
model.fit([X_train_seq, X_train_syn], y_train, epochs=5, batch_size=32, validation_split=0.1)
#model.fit([X_train_seq, X_train_syn], y_train, epochs=10, batch_size=32, validation_split=0.1)

# Save model
model.save("G:\Code\Casestudy1\human_ai_classifier2.h5")
y_pred = (model.predict([X_test_seq, X_test_syn]) > 0.5).astype(int)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("AUROC:", roc_auc_score(y_test, y_pred))

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# --- FPR calculation ---
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
print("False Positive Rate (FPR):", fpr)

def test_sentence(sentence):
    clean = clean_text(sentence)
    seq = pad_sequences(tokenizer.texts_to_sequences([clean]), maxlen=300)
    syn = np.array([extract_features(clean)])

    model = load_model("G:\Code\Casestudy1\human_ai_classifier2.h5")
    semantic_model = Model(inputs=model.input,
                           outputs=model.get_layer("concat_features").input[0])
    semantic_features = semantic_model.predict([seq, syn])

    # Prediction
    prediction = model.predict([seq, syn])[0][0]

    # Concatenated vector = semantic + syntactic
    concatenated = np.concatenate([semantic_features[0], syn[0]])

    print(f"\n🧠 Prediction: {'AI-generated' if prediction > 0.5 else 'Human-written'} (score = {prediction:.4f})")
    print("\n📌 Semantic Features (first 10):", semantic_features[0][:10])
    print("📌 Syntactic + Statistical Features:", syn[0])
    print("📌 Concatenated Features (first 10):", concatenated[:10])
    print("📌 Concatenated Vector Length:", len(concatenated))
    '''
    # Get semantic features (before fusion)
    semantic_layer_model = Model(inputs=model.input, outputs=model.get_layer(index=5).output)
    semantic_features = semantic_layer_model.predict([seq, syn])
    
    # Final prediction
    prediction = model.predict([seq, syn])[0][0]
    
    print(f"Prediction: {'AI-generated' if prediction > 0.5 else 'Human-written'} (score: {prediction:.4f})")
    print("Semantic feature vector shape:", semantic_features.shape)
    print("First 10 semantic features:", semantic_features[0][:10])
'''


test_sentence("It sounds like you may be struggling with a condition called intrusive thoughts. Intrusive thoughts are unwanted, involuntary thoughts, images, or urges that can be distressing and can interfere with daily life. They can take many forms and can be about a wide range of topics, including things that are disturbing or inappropriate.\n\nIf you are struggling with intrusive thoughts and they are causing you distress or interfering with your daily life, it may be helpful to seek treatment from a mental health professional. A therapist or counselor can help you learn coping strategies to manage your thoughts and improve your overall well-being.\n\nHere are a few things you can try to help manage your intrusive thoughts:\n\nPractice mindfulness: This involves paying attention to the present moment and accepting your thoughts and feelings without judgment.\n\nUse distraction techniques: Engaging in activities that take your mind off of your thoughts can be helpful in managing them.\n\nChallenge your thoughts: Try to look at your thoughts objectively and consider whether they are accurate or not.\n\nSeek support: Talking to a trusted friend or family member about your thoughts can be helpful in managing them. You can also seek support from a mental health professional.\n\nIt's important to remember that having intrusive thoughts does not mean you are abnormal or that there is something wrong with you. Many people experience intrusive thoughts from time to time, and with the right support and treatment, you can learn to manage them and improve your overall well-being.")
'''
import spacy
import numpy as np

nlp = spacy.load('en_core_web_sm')
doc = nlp(test_sentence)

# ✳️ Syntactic features
pos_counts = doc.count_by(spacy.attrs.POS)
pos_vector = [pos_counts.get(i, 0) for i in range(100)]  # First 100 POS IDs
avg_parse_depth = np.mean([token.dep_.count('_') + 1 for token in doc])
noun_phrases = len(list(doc.noun_chunks))

# ✳️ Statistical features
word_count = len(doc)
char_count = len(test_sentence)
avg_word_len = np.mean([len(token) for token in doc])
punctuation_count = sum([1 for ch in test_sentence if ch in '.,!?'])

# Combine all
syntactic_stat_features = np.array([
    word_count,
    char_count,
    avg_word_len,
    punctuation_count,
    noun_phrases,
    avg_parse_depth
] + pos_vector[:10])  # Keep only first 10 POS counts for simplicity

print("Syntactic + Statistical Features:", syntactic_stat_features)
'''

