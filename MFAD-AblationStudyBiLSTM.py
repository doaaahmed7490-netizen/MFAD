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
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Bidirectional, LSTM,SimpleRNN, GRU, Dense, Concatenate, Dropout,GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
#from tensorflow.keras.layers import SimpleRNN, GRU

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
df = load_hc3_json("all-Test.json")
#df = load_hc3_json("MedicineGPT.json")
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

    #lstm = LSTM(64)(pool)
    #rnn = SimpleRNN(64)(pool)
    gru=GRU(64)(pool)
    #merged = Concatenate()([bilstm, meta_input])
    #merged = Concatenate(name="concat_features")([lstm, meta_input])
    #merged = Concatenate(name="concat_features")([rnn, meta_input])

    merged = Concatenate(name="concat_features")([gru, meta_input])
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
# Example
#test_sentence("The implications of large language models are still being debated in academia.")
#test_sentence("There is most likely an error in the WSJ's data.  Yahoo! Finance reports the P\/E on the Russell 2000 to be 15 as of 8\/31\/11 and S&P 500 P\/E to be 13 (about the same as WSJ). Good catch, though!  E-mail WSJ, perhaps they will be grateful.")
#test_sentence("The best way for residents of smaller economies to allocate their portfolio between domestic and foreign assets will depend on their individual financial goals, risk tolerance, and investment horizon. Some factors to consider when making this decision include the potential returns, risks, and diversification benefits of each asset class, as well as the investor's ability to handle currency exchange risk and the overall economic environment in their home country.One approach to portfolio allocation for residents of smaller economies is to diversify across both domestic and foreign assets to help spread risk and potentially increase returns. This could involve investing in a mix of domestic stocks, bonds, and cash equivalents, as well as foreign stocks, bonds, and other investment instruments.It's important to remember that investing in foreign assets carries additional risks, such as currency exchange risk and political risk, that may not be present in domestic investments. Investors should carefully assess these risks and consider the impact they may have on their portfolio.It's also a good idea to consult with a financial professional or advisor to help assess the risks and potential returns of different investment options and to develop a portfolio allocation strategy that is tailored to the individual's financial situation and goals.")
#test_sentence("Residents of smaller economies should allocate their portfolios between domestic and foreign assets with a strong emphasis on international diversification. Smaller economies often have limited market size, sector concentration, and greater exposure to local economic, political, and currency risks. To mitigate these vulnerabilities, it is generally advisable for investors to allocate a significant portion—often 50% or more—of their investments to foreign assets. This includes global equities, bonds, and real assets, which offer broader diversification, access to more stable currencies, and exposure to global economic growth. However, maintaining some allocation to domestic assets remains important for familiarity, potential tax advantages, and alignment with local spending needs. The exact allocation should reflect the investor’s risk tolerance, investment goals, access to global markets, and currency outlook, with periodic rebalancing to maintain alignment with long-term objectives.")
#test_sentence("hello, the use of clindamycin can cause stomach pain or a hyperacidity of the stomach. so, i recommend using a medication to lower the acidity production such as omeprazole daily. i also suggest using maalox three time a day and avoid food that can trigger the symptom such a spicy food. hope i have answered your query. let me know if i can assist you further. take care regard, dr.dorina gurabardhi, general & family physician.")
#test_sentence("it is possible that chest pain could be related to the use of clindamycin and oxycodone. however, it is also important to consider other potential cause of chest pain, such a heart problem or other underlying health condition. it is important to speak with a healthcare provider if you are experiencing chest pain , a this could be a serious issue that requires medical attention. in the meantime, you should follow the instruction of your healthcare provider and report any adverse effect to them. if you are experiencing difficulty breathing or severe chest pain , you should seek immediate medical attention.")
#test_sentence("hello , thank you for posting on hcm. i can understand your concern regarding the skin lesion but it difficult to point specific diagnosis in absence of clinical examination. therefore , i insist you to post good clinical picture to me so that i can help you in a better way. you can reach me directly through the below mentioned link.")
#test_sentence("it's difficult to accurately diagnose a skin rash without seeing it, but it's possible that the rash could be related to your sunburn. it could be a sunburn rash, which is a common reaction to sun exposure. sunburn rash typically appear as red, swollen, and painful area on the skin and may blister. they can be itchy and may feel tender to the touch. there are a few thing you can try to help alleviate the symptom of a sunburn rash: take over-the-counter pain medication, such as ibuprofen or acetaminophen, to help reduce inflammation and pain. apply a moisturizing cream or lotion to your skin to help soothe and hydrate it. avoid further sun exposure and wear protective clothing, such as long sleeve and a hat, to protect your skin from further damage. avoid hot shower or bath, a they can further irritate your skin. instead, take lukewarm shower or bath and use a  gentle, fragrance-free soap. if your rash doe not improve after a few day or if it becomes worse, you should see a healthcare provider for further evaluation and treatment. it 's also a good idea to see a healthcare provider if you have a fever, severe pain, or if the rash is accompanied by other symptom, such as swelling or difficulty breathing.")
#test_sentence("hi, it is difficult to comment on above finding. i can not comment on your ovary & i think i can interpret it better if i know the exact dimension of the ovary. endometrial thickness (et) is normal. i suggest you to attach your exact report if you need a complete interpretation. if you want to attach your report please contact me personally at following link & upload your report: or e-mail me at: best regard.")
#test_sentence("it sound like you have undergone a follicular study, which is a test that track the development of the follicle in the ovary. the follicle contain the egg that are released during ovulation. it appears that your right ovary is smaller than your left ovary, and there is no fluid present in your uterus. polycystic ovary syndrome (pcos) is a common cause of irregular menstrual cycle and difficulty conceiving. it is characterized by the presence of multiple small cyst on the ovary and an imbalance of hormone, including high level of androgen (male hormone) and insulin. pcos can lead to problem with ovulation, a well a other reproductive and metabolic issue. it 's important to discus your fertility concern and the result of your follicular study with your healthcare provider. they can provide more information about your specific situation and recommend appropriate treatment option. this may include lifestyle change, medication, or assisted reproductive technology such as in vitro fertilization (ivf). it's also important to manage any underlying health condition, such as pcos, to increase your chance of becoming pregnant.")
#test_sentence("hi, thank you for posting your query. there is no need to worry about straightening of lumbar spine or loss of lumbar lordosis on mri. however, you need to start back extension exercise to make it better. regarding pcod, please contact a gynaecologist and you would improve with medications. for acne, consult a dermatologist. i hope my answer help. please get back if you require any additional information. wishing you good health, dr sudhir kumar md (internal medicine), dm (neurology) senior consultant neurologistapollo hospital, hyderabad, indiaclick on this link to ask me a direct query.")
#test_sentence("it's great to see you here today! i'll be happy to help with your question. regarding your question about lumbar lordosis, this refers to the normal curvature of the lower part of the spine. in a healthy spine, the lumbar region should have a natural inward curve, which is known a lordosis. straightening of the lumbar lordosis, also known as flattening or loss of lordosis, may be a cause for concern and could indicate an underlying issue such as muscle weakness, a spinal condition, or poor posture. it's important to speak with a healthcare professional about this finding in your mri report and to determine the appropriate course of action. as for your question about acne, there are many over-the-counter and prescription cream and ointment that may be effective for treating acne. some option include benzoyl peroxide, salicylic acid, and retinoids. it's important to choose a product that is appropriate for your skin type and to follow the instruction for use carefully. if you have tried several different product without success, it may be helpful to speak with a dermatologist or other healthcare professional for additional guidance and treatment option. it's also worth noting that acne can be affected by a variety of factor, including hormonal change, certain medication, and certain skin care product. if you are suffering from pcod (polycystic ovary syndrome), this could also contribute to acne breakout. it's important to work with a healthcare professional to address any underlying factor and to find a treatment plan that work for you.")
#GPT 4
#test_sentence("Yes, chest pain or tightness can potentially be related to the intake of Clindamycin and Oxycodone with Acetaminophen, and it should not be ignored. Clindamycin, though generally well tolerated, can rarely cause allergic reactions that present with symptoms such as chest tightness, difficulty breathing, or swelling. Oxycodone, an opioid pain medication, may also lead to serious side effects such as respiratory depression, which can manifest as chest discomfort or tightness, especially when combined with other medications or in individuals sensitive to opioids. Since you’ve experienced chest tightness after taking this combination three times, it is important to contact your physician immediately. This symptom could be a sign of a drug reaction or a more serious underlying issue. Until you speak with your healthcare provider, it would be wise to stop taking the medications and seek medical attention promptly if the symptoms worsen or are accompanied by shortness of breath, dizziness, or swelling.")
#Gimini
#test_sentence("It is important to discuss your symptoms with a healthcare professional immediately. While clindamycin and oxycodone are not typically directly associated with chest tightness as a common side effect, any new onset of chest pain or tightness warrants prompt medical evaluation. Chest pain can be a symptom of various conditions, some of which can be serious, and it's crucial to determine the underlying cause. Your doctor will be able to assess your specific situation, consider your medical history, and determine if your symptoms are related to the medications or another issue.")
#Claude
#test_sentence("Low-thrust electric propulsion presents a highly efficient alternative to chemical propulsion and is frequently considered for interplanetary missions. Notable mission designs that have employed low-thrust propulsion systems include the Bepi Colombo mission [1], the Hayabusa mission [2], the Dawn mission [3], and the SMART-1 mission [4]. These missions showcase the utilization of propulsion systems that generate minimal thrust, requiring its application over a significant portion of the spacecraft's trajectory. Consequently, the optimal control problem (OCP) of transferring the spacecraft between celestial bodies becomes a continuous rather than a discrete problem, significantly increasing the computational demands associated with trajectory design. This poses a particular challenge during preliminary mission design stages when numerous options must be considered and evaluated, especially when identifying attractive targets (from a dynamics perspective) for missions focused on one or more asteroids, such as the Dawn mission [3], Marco Polo [5], and Don Quijote [6].")
#test_sentence("Quantum behavior is vastly different from that of classical systems, posing unique challenges for physicists. To describe quantum effects, the use of finite quantum models has become increasingly prevalent. In this paper, we present a constructive approach towards constructing such models. We begin by defining the underlying principles of quantization, followed by a description of how a finite quantum model can be generated. We demonstrate how these models allow us to investigate the behavior of quantum systems with increased precision, especially in systems where the size of the Hilbert space is limited. Additionally, we explore the relationship between finite quantum models and other approaches to quantum behavior, such as density matrices and path integrals. Our work presents a valuable contribution to the study of quantum mechanics, illustrating the usefulness of finite quantum models in understanding complex quantum systems.")
#test_sentence("Universality of quantum mechanics -- its applicability to physical systems of quite different nature and scales -- indicates that quantum behavior can be a manifestation of general mathematical properties of systems containing indistinguishable, i.e. lying on the same orbit of some symmetry group, elements. In this paper we demonstrate, that quantum behavior arises naturally in systems with finite number of elements connected by non-trivial symmetry groups. The \"finite\" approach allows to see the peculiarities of quantum description more distinctly without need for concepts like \"wave function collapse\", \"Everett's multiverses\" etc. In particular, under the finiteness assumption any quantum dynanics is reduced to the simple permutation dynamics.\n\nThe advantage of the finite quantum models is that they can be studied constructively by means of computer algebra and computational group theory methods.")
#test_sentence("in this paper, we investigate the dynamical behavior of the dilaton field in de sitter space. by analyzing the classical equation of motion for the dilaton and it interaction with gravity, we show that the dilaton behaves as a non-minimally coupled scalar field in de sitter space. we then study the cosmological implication of this behavior, including the dilaton 's contribution to the cosmic acceleration and it potential role in inflation. finally, we discus the effect of quantum correction on our result and their implication for the long-term behavior of the dilaton in a de sitter universe.")
#test_sentence("biomolecular structure are assembly of emergent anisotropic building module such a uniaxial helix or biaxial strand. we provide an approach to understanding a marginally compact phase of matter that is occupied by protein and dna. this phase, which is in some respect analogous to the liquid crystal phase for chain molecule, stabilizes a range of shape that can be obtained by sequence-independent interaction occurring intra- and intermolecularly between polymeric molecule. we present a singularity f ree self-interaction for a tube in the continuum limit and show that this result in the tube being positioned in the marginally compact phase. our work provides a unified framework for understanding the building block of biomolecules.")
#test_sentence("We propose a new forward electricity market framework that admits heterogeneous market participants with second-order cone strategy sets, who accurately express the nonlinearities in their costs and constraints through conic bids, and a network operator facing conic operational constraints. In contrast to the prevalent linear-programming-based electricity markets, we highlight how the inclusion of second-order cone constraints enables uncertainty-, asset- and network-awareness of the market, which is key to the successful transition towards an electricity system based on weather-dependent renewable energy sources. We analyze our general market-clearing proposal using conic duality theory to derive efficient spatially-differentiated prices for the multiple commodities, comprising of energy and flexibility services. Under the assumption of perfect competition, we prove the equivalence of the centrally-solved market-clearing optimization problem to a competitive spatial price equilibrium involving a set of rational and self-interested participants and a price setter. Finally, under common assumptions, we prove that moving towards conic markets does not incur the loss of desirable economic properties of markets, namely market efficiency, cost recovery and revenue adequacy. Our numerical studies focus on the specific use case of uncertainty-aware market design and demonstrate that the proposed conic market brings advantages over existing alternatives within the linear programming market framework.")
#test_sentence("This PhD thesis is devoted to deterministic study of the turbulence in the Navier- Stokes equations. The thesis is divided in four independent chapters.The first chapter involves a rigorous discussion about the energy's dissipation law, proposed by theory of the turbulence K41, in the deterministic setting of the homogeneous and incompressible Navier-Stokes equations, with a stationary external force (the force only depends of the spatial variable) and on the whole space R3. The energy's dissipation law, also called the Kolmogorov's dissipation law, characterizes the energy's dissipation rate (in the form of heat) of a turbulent fluid and this law was developed by A.N.\n\nKolmogorov in 1941. However, its deduction (which uses mainly tools of statistics) is not fully understood until our days and then an active research area consists in studying this law in the rigorous framework of the Navier-Stokes equations which describe in a mathematical way the fluids motion and in particular the movement of turbulent fluids. In this setting, the purpose of this chapter is to highlight the fact that if we consider the Navier-Stokes equations on R3 then certain physical quantities, necessary for the study of the Kolmogorov's dissipation law, have no a rigorous definition and then to give a sense to these quantities we suggest to consider the Navier-Stokes equations with an additional damping term. In the framework of these damped equations, we obtain some estimates for the energy's dissipation rate according to the Kolmogorov's dissipation law.In the second chapter we are interested in study the stationary solutions of the damped Navier- Stokes introduced in the previous chapter. These stationary solutions are a particular type of solutions which do not depend of the temporal variable and their study is motivated by the fact that we always consider the Navier-Stokes equations with a stationary external force. In this chapter we study two properties of the stationary solutions : the first property concerns the stability of these solutions where we prove that if we have a control on the external force then all non stationary solution (with depends of both spatial and temporal variables) converges toward a stationary solution. The second property concerns the decay in spatial variable of the stationary solutions. These properties of stationary solutions are a consequence of the damping term introduced in the Navier-Stokes equations.In the third chapter we still study the stationary solutions of Navier-Stokes equations but now we consider the classical equations (without any additional damping term). The purpose of this chapter is to study an other problem related to the deterministic description of the turbulence : the frequency decay of the stationary solutions. Indeed, according to the K41 theory, if the fluid is in a laminar setting then the stationary solutions of the Navier-Stokes equations must exhibit a exponential frequency decay which starts at lows frequencies. But, if the fluid is in a turbulent setting then this exponential frequency decay must be observed only at highs frequencies. In this chapter, using some Fourier analysis tools, we give a precise description of this exponential frequency decay in the laminar and in the turbulent setting.In the fourth and last chapter we return to the stationary solutions of the classical Navier-Stokes equations and we study the uniqueness of these solutions in the particular case without any external force. Following some ideas of G. Seregin, we study the uniqueness of these solutions first in the framework of Lebesgue spaces of and then in the a general framework of Morrey spaces.")

#test_sentence("Historical price-to-earnings (P\/E) ratios for small-cap and large-cap stocks can vary significantly over time and may not be directly comparable due to the different characteristics of these two categories of stocks.Small-cap stocks, which are defined as stocks with a market capitalization of less than $2 billion, tend to be riskier and more volatile than large-cap stocks, which have a market capitalization of $10 billion or more. As a result, investors may be willing to pay a higher price for the potential growth opportunities offered by small-cap stocks, which can lead to higher P\/E ratios.On the other hand, large-cap stocks tend to be more established and stable, with a longer track record of earnings and revenue growth. As a result, these stocks may trade at lower P\/E ratios, as investors may be less willing to pay a premium for their growth potential.It is important to note that P\/E ratios are just one factor to consider when evaluating a stock and should not be used in isolation. Other factors, such as the company's financial health, industry trends, and macroeconomic conditions, can also impact a stock's P\/E ratio.")
#test_sentence("Your sister Bhagyashree’s condition, as described, involves allergic asthma along with bronchitis and bilateral hilar lymphadenopathy seen on her chest X-ray. Allergic asthma is typically triggered by allergens like dust, pollen, or pet dander, and it causes airway inflammation and breathing difficulty. Bronchitis indicates inflammation of the bronchial tubes, which can further aggravate asthma symptoms. The finding of bilateral hilar lymphadenopathy—enlarged lymph nodes near the lungs—can be caused by several conditions, including infections (like tuberculosis), inflammatory diseases (such as sarcoidosis), or even certain types of lymphoma, though not all causes are serious. Treatment should begin with controlling the asthma using inhaled corticosteroids, bronchodilators (like salbutamol or Levolin), and antihistamines if allergies are present. Avoiding known allergens, improving air quality at home, and using a peak flow meter to monitor lung function can also help. However, the lymphadenopathy requires further evaluation—your sister may need a CT scan, blood tests, or even a biopsy, depending on her symptoms and doctor’s guidance. You should consult a pulmonologist (lung specialist) for a more accurate diagnosis and treatment plan. It is important not to ignore the lymph node enlargement, as it may point to a condition that needs specific treatment beyond asthma management.")
#test_sentence("Experiencing soreness, tiredness, and pain all over the body despite having normal blood tests and a healthy lifestyle can be caused by several factors. Conditions such as fibromyalgia, chronic fatigue syndrome, or stress-related muscle tension often lead to widespread pain and fatigue without clear abnormalities in routine blood work. Other possibilities include vitamin deficiencies (like vitamin D or B12), thyroid issues that might not be detected in basic panels, or poor sleep quality, which can significantly affect how you feel physically. Mental health factors like anxiety or depression can also manifest as physical exhaustion and body aches. It’s important to follow up with your healthcare provider to explore these possibilities further, consider more specialized tests, and potentially evaluate your sleep patterns and mental health to get a clearer diagnosis and appropriate treatment.")

#test_sentence("Duphaston (dydrogesterone) is a synthetic form of progesterone often prescribed to support the luteal phase of the menstrual cycle and help prepare the uterine lining for implantation, but it does not directly cause pregnancy—it simply creates favorable conditions if fertilization occurs. Since you and your husband have normal test results, pregnancy is still possible while taking Duphaston, but success depends on whether ovulation and conception happen during this cycle. If you are not pregnant, your period will usually come within a few days to a week after stopping the medication. If your period is delayed beyond that, it’s a good idea to take a pregnancy test to confirm.")
#test_sentence("Hi, dearI have gone through your question. I can understand your concern. You may have inguinal hernia or enlarged inguinal lymphnode.  You should go for examination. If needed go for ultrasound study.  If it is lymphnode or soft tissue tumor then you should go for fine needle aspiration cytology or biopsy of that lump.  If it is hernia then go for surgery.  Consult your doctor and take treatment accordingly. Hope I have answered your question, if you have doubt then I will be happy to answer. Thanks for using health care magic. Wish you a very good health.")
#test_sentence("For the long-term management of prurigo nodularis on your lower legs, it's important to understand that this is a chronic skin condition often characterized by a vicious cycle of itching and scratching, which leads to the formation of hard, itchy nodules. The fact that you have a history of urticaria suggests you may have a heightened sensitivity in your skin, which can make conditions like prurigo nodularis more persistent. Since you have tried both conventional allopathy with cortisone injections and are now on homeopathic treatment, it's clear that your condition has been challenging to manage. While homeopathy may provide relief for some individuals, for a condition as stubborn as prurigo nodularis, the most effective treatments are often a combination of approaches. To break the itch-scratch cycle, topical treatments are crucial. You can discuss with your doctor the use of prescription-strength steroid creams or ointments to apply directly to the nodules. These can reduce inflammation and itching. Other topical options include creams with capsaicin, which can help desensitize the nerve endings in the skin, or emollients with menthol to provide a cooling sensation that distracts from the itch. For long-term relief and to address the scars and bumps, more systemic options may be needed. Your doctor may consider non-steroidal anti-itch medications or a short course of oral steroids if the flare-ups are severe. Phototherapy, which involves controlled exposure to ultraviolet light, is also a very effective treatment for prurigo nodularis. The UV light can help reduce inflammation and itching. Finally, managing the underlying psychological component is key. The intense itching can be a result of the skin condition, but anxiety and stress can exacerbate it. Finding ways to reduce stress, such as through mindfulness, meditation, or even talking to a counselor, can be a beneficial part of your treatment plan. Since your current homeopathic treatment isn't fully addressing the issue, it may be time to consult with a dermatologist again to create a comprehensive plan that combines the best of these conventional and non-invasive therapies to finally break the cycle of itching and promote healing.")
#test_sentence("Given your financial situation and multiple health conditions, there are several resources and strategies that can help you get the care and medications you need without insurance. First, you should immediately look into Federally Qualified Health Centers (FQHCs) in your area, as these federally funded nonprofit health centers serve medically underserved populations and provide primary care services regardless of your ability to pay, with services provided on a sliding scale fee based on your ability to pay, and FQHCs cannot deny services due to an inability to pay and adhere to an open door policy, offering care regardless of patients' financial ability to pay. For your diabetes medications, eligible uninsured patients can access Sanofi insulins for a fixed price of $35 per month through the Insulins Valyou Savings Program, and Novo Nordisk offers a Patient Assistance Program where you can apply to receive diabetes medicine at no cost. Additionally, you can receive a free FreeStyle blood glucose monitoring system by calling Abbott Diabetes Care at 888-522-5226, and CR3 Diabetes Association provides glucose meters, strips, and discounted supplies to uninsured people living with diabetes. For comprehensive medication assistance, nonprofits like the American Diabetes Association offer help for people with specific conditions. Beyond medication access, focus on lifestyle changes that can significantly impact all three conditions: aim for gradual weight loss through portion control and increased physical activity, reduce sodium intake for blood pressure, choose whole grains and limit refined carbohydrates for diabetes, and incorporate the fish oil you're already taking as it can help with cholesterol. Visit findahealthcenter.hrsa.gov to locate the nearest FQHC, and contact pharmaceutical companies directly about their patient assistance programs - many have income requirements that you would likely qualify for given your part-time income.")
#GPT-5
#test_sentence("try not to worry — Primolut N (norethisterone) taken accidentally for a few days before you knew you were pregnant is not generally associated with a specific pattern of major birth defects, and most women who have short-term exposure in early pregnancy go on to have healthy babies; however, no medication is absolutely risk‑free, so you should contact your obstetrician or midwife for personalized advice, start or continue routine prenatal care (including folic acid if not already taking it), and arrange early antenatal assessment and ultrasound so your provider can monitor the pregnancy and offer reassurance or further testing if needed.")
#test_sentence("Chest pain or a feeling of tightness in the chest after taking Clindamycin and Oxycodone with Acetaminophen should not be ignored, as it may indicate a serious reaction. While Clindamycin can sometimes cause gastrointestinal upset or rarely allergic reactions that may present with chest tightness, Oxycodone can depress breathing or trigger allergic responses that also manifest as chest discomfort. The combination itself is not commonly known to directly cause chest pain, but your symptoms are concerning because chest tightness can signal an allergic reaction, breathing problems, or even an unrelated cardiac issue. Since you have noticed this pattern each time after taking the medications, it is important to stop taking them until you speak with your doctor, and you should seek immediate medical attention if the chest tightness worsens, is associated with shortness of breath, dizziness, or swelling.")
#test_sentence('Please answer this Question as paragraph   "Is chest pain related to intake of clindamycin and oxycodone?Hi Dr. Bhatti, I was recently released from the hospital after a hand surgery and they provided me with Clindamycin 300mg and Oxycodone Acetaminophens. Ive taken this combination 3 times now and my chest feels really tight. Is there reason for me to worry?"')
#test_sentence("Longer paragraph: Oxycodone (an opioid) is the more likely cause among the two — opioids can produce chest discomfort by causing respiratory depression, a sense of chest heaviness, anxiety/panic, or (rarely) chest wall muscle rigidity; they can also trigger bronchospasm in people with asthma or COPD. Clindamycin can cause allergic reactions (which may include chest tightness, wheeze, hives, or swelling) but is less commonly a direct cause of isolated chest pain. There is no common dangerous pharmacologic interaction between clindamycin and oxycodone that typically causes chest pain. Because chest tightness can also signal life‑threatening conditions (anaphylaxis, severe allergic reaction, pneumonia/bronchospasm, pulmonary embolism, or cardiac ischemia), you should stop the medication only if advised by your prescriber and seek prompt medical assessment: if you have shortness of breath, difficulty breathing, swallowing, throat tightness, lip/face swelling, fainting, dizziness, or worsening pain — go to the emergency department or call emergency services right away. If symptoms are milder but persistent, contact your surgeon or prescribing clinician today, describe the timing and nature of the symptoms, and arrange review; they may advise stopping oxycodone, switching pain control, or arranging an exam and pulse/oxygen/blood-pressure check and possibly an ECG or chest evaluation.")
#GPT-5 https://askgpt.app/chat/9fd1ed28-ffb6-43f9-8f66-c7cbd8b02e7d
#test_sentence("Given your history, sharp or aching pain near the 5th–6th rib that worsens with breathing is most often musculoskeletal—costochondritis or chest-wall strain/adhesions after surgery—especially if a single spot is tender to touch and movement changes it 🩺. Implant issues can cause pain too: a saline rupture usually leads to obvious, rapid deflation, while capsular contracture or pectoral tightness causes firmness, shape change, or a “pulling” pain rather than a distinctly pleuritic pain. Metastasis is less likely with pain that started over a few days and is tied to breathing, but your cancer history means it should not be dismissed; bone spread typically causes a persistent, deep, progressive ache (often worse at night) that isn’t linked to breaths. Also consider pleurisy, a small rib fracture, shingles (pain before rash), or—rare but urgent—pulmonary embolism or pericarditis (pleuritic, sometimes positional). Seek urgent care now if you have shortness of breath, chest pressure, rapid heart rate, cough with blood, fever/chills, calf swelling, or recent long travel/immobilization ⚠️. Otherwise, contact your oncologist or breast/implant surgeon within 24–72 hours for an exam; they may check for focal costochondral tenderness, implant changes, and order targeted imaging (chest wall ultrasound, chest X‑ray; ECG if indicated; and if pain persists or exam is concerning, bone scan or PET/CT). Different viewpoints: many clinicians start with conservative treatment for likely costochondritis, while some in oncology prefer earlier imaging given your history—both are reasonable, tailored to your symptoms and exam.")
#test_sentence("The collection of symptoms you are experiencing—low grade fever, alternating hot and cold sweats, body aches, abdominal tenderness, headaches, fatigue, rashes on the arms, and poor wound healing—points to an underlying health issue that needs proper medical evaluation rather than simple home remedies. Since you also mention a history of a severe IUD infection, it is possible that your body is still dealing with lingering infection, inflammation, or even a weakened immune system, which could explain why you are not feeling like yourself and why healing is delayed. While general supportive steps such as staying well hydrated, resting, eating nutritious foods, and using mild pain relievers may help with discomfort, these are not a solution to the underlying cause. Because your symptoms are persistent, involve multiple body systems, and interfere with your daily well-being, it is very important to see a healthcare provider as soon as possible for blood tests, infection screening, and further evaluation to determine if antibiotics, immune support, or other targeted treatment is needed. If your symptoms worsen, especially if you develop high fever, spreading rash, severe abdominal pain, or dizziness, you should seek urgent medical care right away. ")
#test_sentence("Pain at the end of urination can be caused by both kidney stones and urinary tract infections (UTIs), and given your history, either could be possible. Kidney stones often cause sharp, radiating pain in the lower abdomen, back, or groin, sometimes accompanied by blood in the urine, while UTIs are more likely to cause burning with urination, frequent urges to urinate, and localized discomfort around the bladder or genital area. Since you recently took antibiotics and developed a yeast infection, your urinary tract balance may be disrupted, making you more prone to infection. Pain during intercourse can also suggest irritation or infection rather than stones alone. Because the symptoms overlap and treatment is different, it is best to see your doctor for a urine test and possibly imaging to confirm the cause. In the meantime, staying well hydrated may help ease symptoms, but prompt evaluation is important to prevent complications.")
#test_sentence("Brown discoloration of teeth caused by excess fluoride, known as dental fluorosis, is permanent in the enamel but there are effective cosmetic treatments that can greatly improve the appearance. Options include professional teeth whitening, micro-abrasion, dental bonding with tooth-colored resin, veneers, or crowns depending on how severe the discoloration is. Whitening may help in mild cases, but for deeper or darker stains, bonding or veneers usually give the best results. The cost can vary widely depending on the treatment chosen and the country or clinic, ranging from relatively affordable whitening sessions to more expensive procedures like veneers or crowns. It is best to consult a dentist who can examine the severity of the fluorosis and guide you on the most suitable and cost-effective option to restore a natural white appearance to your teeth.")
#Medicine GPT3.5 For Samples For Paper Journal Colleage
#test_sentence("It's generally safe to take Lactare capsules while breastfeeding, as they contain lactose and other ingredients that are thought to be safe for nursing mothers and their infants. However, it's always important to be cautious when taking any medication, even if it's available over the counter. It's always a good idea to talk to your healthcare provider before starting any new medication, including Lactare capsules, to make sure it's appropriate for you and your baby.\n\nThere is limited information available on the long-term safety of taking Lactare capsules, as they are not intended for long-term use. It's generally recommended to use them for a short period of time to help improve milk production, rather than continuing to take them for an extended period.\n\nIf you have any concerns about the safety of taking Lactare capsules or any other medication while breastfeeding, it's important to discuss them with your healthcare provider. They can help you weigh the potential risks and benefits and make an informed decision that's best for you and your baby.")
#Sentence5 Human
#test_sentence("HiYes you can. Add lot of garlic and fenugreek in your food. If non vegetarian then you can take meat egg and fish. Drink at least 3 to 4 liters of fluids per day.Have a healthy diet comprising lots of seasonal fruits & vegetables, with atleast 1 glass of milk every day.Enjoy your motherhood. Hope I have answered your query. Let me know if I can assist you further. RegardsDr.Saranya Ramadoss, General and Family Physician")
#Sentence6 Human
#test_sentence("I think in my opinion this is the right medicine as this can clear the infection of your abdomen and is best for that the rest depends on what comes in your ultrasound")
#Sentence6 GPT
#test_sentence("It is not appropriate for me to provide medical advice or make treatment recommendations without being able to examine you and review your medical history. The best course of action is to follow the recommendations of your healthcare provider. They are the best source of information about your health and the appropriate treatment for any medical condition you may have. If you have concerns about the treatment your healthcare provider has prescribed, it is important to discuss them with them. They will be able to provide more information and address any questions or concerns you may have.")
#Sentence7 Human
#test_sentence("Dear friend,As per your detail, you have neck swelling and chest swelling. if i am treating doctor, then i would like to examine the swelling first. Diagnosis can confirmed after examination only. A neck lump may be a lymphnode swelling and other swelling on chest me be lipoma or other cyst like sebaceous cyst. Lymphnode sweliing may subside itself, but lipoma or sebaceous cyst needs excision.Thank You.")
#Sentence7 GPT
#test_sentence("It is not possible for me to diagnose the cause of the lumps you are describing or recommend a treatment without more information and a physical examination. It is important to see a doctor for proper diagnosis and treatment.\n\nThere are many different types of lumps that can occur in the neck, chest, and back, and they can have a variety of causes. Some lumps may be benign (non-cancerous), while others may be malignant (cancerous). Only a doctor can determine the cause of the lumps and recommend the appropriate treatment.\n\nIf you are concerned about the lumps you are describing, it is important to see a doctor as soon as possible. The doctor will perform a physical examination and may recommend additional tests, such as a biopsy (a procedure in which a small sample of tissue is taken and examined under a microscope), to determine the cause of the lumps.\n\nIt is important to note that it is not uncommon for people to have small, benign lumps in their neck, chest, or back. However, it is always important to have any new or unexplained lumps checked by a doctor.")
#Sentence8 Human
#test_sentence("Thanks for asking in healthcaremagic forumI appreciate your concern, many times white hard concretions can be seen on the tonsils which give foreign body sensation during swallowing. I would like to suggest you to visit an ENT specialist for this. Do not meddle with it yourself, it may cause bleeding which cannot be controlled in home\/OPD settings. All the best.")
#Sentence8 GPT
#test_sentence("It's important to have a medical professional evaluate any growth or abnormality on the tonsils. It's possible that the sack you are describing could be a tonsil stone, which is a calcified mass that forms in the tonsils due to a build-up of bacteria, food particles, and other debris. Tonsil stones are usually benign and can often be removed by gently massaging the tonsil with a cotton swab or rinsing the mouth with water. However, it's important to have a healthcare provider evaluate the sack to rule out any other possible causes and determine the best course of treatment.\n\nI recommend that you make an appointment with a healthcare provider or a ENT (ear, nose, and throat) specialist as soon as possible to have the sack evaluated. In the meantime, you can encourage your daughter to drink plenty of fluids and avoid irritants such as tobacco smoke and alcohol, which can irritate the tonsils and make the sack more uncomfortable.")
#Sentence9 Human
#test_sentence("Hello and welcome to \u2018Ask A Doctor\u2019 service. I have reviewed your query and here is my advice. Lower back pain with migration to both lower limbs common cause is disc herniation with nerve root compression. if it's first episode take rest. Avoid forward bending and weight lifting. Start tab zerodol sp twice a day for pain relief. Tab pregalin x 75 mg one at bedtime for relief of radicular pain. Also avoid jerky movements and long distance travelling by road. You will be fine in few days. Hope I have answered your query. Let me know if I can assist you further.")
#Sentence9 GPT
#test_sentence("There are several potential causes of lower back pain that radiates to the buttocks and legs. These can include:\n\nHerniated disc: A herniated disc occurs when the inner gel-like material of a spinal disc pushes out through a tear in the outer layer. This can cause pain, numbness, and weakness in the lower back, buttocks, and legs.\n\nSciatica: This is a type of pain that occurs when the sciatic nerve, which runs from the lower back through the buttocks and down the legs, becomes compressed or irritated. Sciatica can cause pain, numbness, and tingling in the lower back, buttocks, and legs.\n\nSpinal stenosis: This is a condition in which the spaces within the spine narrow, putting pressure on the spinal cord and nerves. This can cause pain, numbness, and weakness in the lower back, buttocks, and legs.\n\nPiriformis syndrome: The piriformis muscle is located deep in the buttocks. If it becomes tight or inflamed, it can put pressure on the sciatic nerve and cause pain, numbness, and tingling in the lower back, buttocks, and legs.\n\nMuscular strains or sprains: Overuse or injury to the muscles in the lower back, buttocks, or legs can cause pain and discomfort.\n\nIt's important to see a healthcare provider for a proper diagnosis and treatment plan. In the meantime, you can try applying heat or cold to the affected area, taking over-the-counter pain medication, and doing gentle stretches to help relieve the pain.")
#Sentence10 Human
#test_sentence("Hi,Thank you for posting your query.I have noted your past history and the MRI brain findings.First of all, I would like to reassure you that these do not suggest any serious brain disease.Periventricular increased signals are a non-specific finding and can occur in a variety of situations. For example, it can be due to ischemia (decreased blood flow to brain in a high bP patient), demyelination, migraine, etc. We need to correlate the MRI findings with your symptoms. So, if you can get back with your symptoms, we would be able to give you a better opinion.I hope my answer helps. Please get back if you have any follow up queries or if you require any additional information.Wishing you good health,Dr Sudhir Kumar MD (Internal Medicine), DM (Neurology)Senior Consultant NeurologistApollo Hospitals, Hyderabad, IndiaClick on this link to ask me a DIRECT QUERY: http:\/\/bit.ly\/Dr-Sudhir-kumarMy BLOG: http:\/\/bestneurodoctor.blogspot.in")
#Sentence10 GPT
#test_sentence("The periventricular white matter is a region of the brain that is located near the ventricles, which are fluid-filled cavities in the brain. An increase in T2 and FLAIR signal in this area can indicate the presence of abnormal tissue, such as scar tissue or inflammation. This could potentially be a sign of a problem, but it's important to note that these findings need to be interpreted in the context of the rest of the MRI and the patient's overall clinical presentation.\n\nIt's also worth noting that the term \"cortical involution changes\" refers to changes in the thickness or density of the cortex, which is the outer layer of the brain. These changes can occur naturally with aging, but they can also be caused by other factors such as brain injury or disease.\n\nIt's important to speak with your doctor about the results of your MRI and any potential implications they may have for your health. Your doctor will be able to provide you with more information and guidance based on your specific situation.")
#Sentence11 Human
#test_sentence("Hello!Thank you for asking on HCM!I read your question carefully and understand your concern. Your symptoms seem to be related to anxiety. Your family history for atrial fibrilation doesn't seem to be related to your symptomatology. Has anybody besides you noticed the skin discoloration of your limbs? You are too young for such cardiac disorders. Anyway, a metabolic disorder (like thyroid dysfunction, etc.) can not be excluded. I recommend consulting with your GP for a careful physical examination and a complete check up : - a resting ECG- a chest X ray study - blood lab tests : complete blood count, thyroid hormone levels, blood electrolytes, fasting glucose, kidney and liver function tests, cortisol plasma levels, etc. A treadmill test and an ambulatory 24-48 hours ECG monitoring may be necessary if suspicions of a cardiac arrhythmia are raised. If all the above tests result normal, the main cause of your symptoms would be just pure anxiety. In such case, I would recommend consulting with a specialist of this field to help you manage your anxiety. Hope to have been helpful!Best regards!Dr. Iliri")

#Sentence11 GPT
#test_sentence("It is important to seek medical attention for your symptoms as soon as possible, as they could potentially be indicative of a serious medical condition. Chest pain, in particular, can be a symptom of a heart attack or other cardiac event, and it is important to get this evaluated by a healthcare professional as soon as possible to determine the cause and receive appropriate treatment.\n\nIn the meantime, you can try the following self-care measures to help alleviate your symptoms:\n\nTake over-the-counter pain medication, such as acetaminophen or ibuprofen, to help reduce pain and inflammation.\n\nUse heat or cold therapy to help reduce pain and swelling in your ankles and feet.\n\nElevate your feet when you are sitting or lying down to help reduce swelling.\n\nAvoid standing or walking for long periods of time.\n\nAvoid activities that put strain on your ankles and feet, such as running or jumping.\n\nWear comfortable, supportive shoes.\n\nIt is important to note that these self-care measures are not a substitute for medical treatment, and you should see a healthcare professional as soon as possible to determine the cause of your symptoms and receive appropriate treatment.")
#Sentence12 Human
#test_sentence("DearWe understand your concerns.I went through your details. I suggest you to keep calm. You are the person who know you very well. Once you know the reason for your trouble why don't you correct it. You have these problems just because of over imagination an actions. You are an IT employee. IT employees are supposed to be busy people. Are you not busy enough? Where o you get time to think and imagine?If you know answers to the above questions, you can cure your problem. Secondly keeping yourself busy in work reduces unnecessary thinking. To work hard you need to be passionate in your work. Remember ''Idle mind is devils workshop''.In case if you need more of my services in this regard, please post a direct query. I am happy to reply you.Hope this answers your query. Available for further clarifications.Good luck.")
#Sentence12 GPT
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

