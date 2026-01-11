import nltk
import spacy
import string
import re
import textstat
import language_tool_python
from spellchecker import SpellChecker
from nltk import word_tokenize, sent_tokenize, pos_tag
from collections import Counter
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# --- Input Paragraph ---
#paragraph = """Natural language processing (NLP) is a branch of artificial intelligence. It focuses on enabling computers to understand and respond to human language. Tools like ChatGPT demonstrate the power of large language models."""
#paragraph="Hello,The use of Clindamycin can cause stomach pain or a hyperacidity of the stomach. So, I recommend using a medication to lower the acidity production such as Omeprazole daily. I also suggest using Maalox three times a day and avoid food that can trigger the symptoms such as spicy food. Hope I have answered your query. Let me know if I can assist you further. Take care Regards,Dr. Dorina Gurabardhi, General & Family Physician."
#paragraph="It is possible that chest pain could be related to the use of Clindamycin and Oxycodone. However, it is also important to consider other potential causes of chest pain, such as heart problems or other underlying health conditions. It is important to speak with a healthcare provider if you are experiencing chest pain, as this could be a serious issue that requires medical attention. In the meantime, you should follow the instructions of your healthcare provider and report any adverse effects to them. If you are experiencing difficulty breathing or severe chest pain, you should seek immediate medical attention."
#paragraph="hello, the use of clindamycin can cause stomach pain or a hyperacidity of the stomach. so, i recommend using a medication to lower the acidity production such a omeprazole daily. i also suggest using maalox three time a day and avoid food that can trigger the symptom such a spicy food. hope i have answered your query. let me know if i can assist you further. take care regard, dr.dorina gurabardhi, general & family physician."
#Before preprocessing
#paragraph="it's great to see you here today ! i 'll be happy to help with your question. regarding your question about lumbar lordosis, this refers to the normal curvature of the lower part of the spine. in a healthy spine, the lumbar region should have a natural inward curve, which is known a lordosis. straightening of the lumbar lordosis, also known as flattening or loss of lordosis, may be a cause for concern and could indicate an underlying issue such a muscle weakness, a spinal condition, or poor posture. it 's important to speak with a healthcare professional about this finding in your mri report and to determine the appropriate course of action. as for your question about acne, there are many over-the-counter and prescription cream and ointment that may be effective for treating acne. some option include benzoyl peroxide, salicylic acid, and retinoids. it 's important to choose a product that is appropriate for your skin type and to follow the instruction for use carefully. if you have tried several different product without success, it may be helpful to speak with a dermatologist or other healthcare professional for additional guidance and treatment option. it 's also worth noting that acne can be affected by a variety of factor, including hormonal change, certain medication, and certain skin care product. if you are suffering from pcod (polycystic ovary syndrome), this could also contribute to acne breakout. it's important to work with a healthcare professional to address any underlying factor and to find a treatment plan that work for you."
# Tokenize into words
#Casestudy 2 
#Sentence1 human
#paragraph="Within the scope of a FRW cosmological model we have studied the role of spinor field in the evolution of the Universe when it is non-minimally coupled to the gravitational one. We have considered a few types of nonlinearity. It was found that if the spinor field nonlinearity describes an ordinary matter such as radiation, the presence of non-minimality becomes essential and leads to the rapid expansion of the Universe, whereas if the spinor field nonlinearity describes a dark energy, the evolution of the Universe is dominated by it and the difference between the minimal and non-minimally coupled cases become almost indistinguishable."
#Sentence1 GPT
#paragraph="In this paper, we investigate the behavior of a non-minimally coupled nonlinear spinor field in the context of FRW cosmology. We analyze the evolution of the homogeneous and isotropic universe with this type of field and compare it with the minimally coupled case. We show that the non-minimally coupled spinor field behaves differently from the minimally coupled one, affecting the evolution of the universe's scale factor and the spinor field itself. In particular, we find that the non-minimal coupling generates an interaction between the spinor field and the spatial curvature, resulting in unique cosmic evolution. Our results provide insights into the impact of non-minimal couplings on the behavior of physical systems in cosmology."
#Sentence2 human
#paragraph="We study the dynamical behavior of the dilaton in the background of three-dimensional Kerr-de Sitter space which is inspired from the low-energy string effective action. The perturbation analysis around the cosmological horizon of Kerr-de Sitter space reveals a mixing between the dilaton and other fields. Introducing a gauge (dilaton gauge), we can disentangle this mixing completely and obtain one decoupled dilaton equation. However it turns out that this belongs to the tachyon. The stability of de Sitter solution with J=0 is discussed. Finally we compute the dilaton absorption cross section to extract information on the cosmological horizon of de Sitter space."
#Sentence2 GPT
#paragraph="In this paper, we investigate the dynamical behavior of the dilaton field in de Sitter space. By analyzing the classical equations of motion for the dilaton and its interaction with gravity, we show that the dilaton behaves as a non-minimally coupled scalar field in de Sitter space. We then study the cosmological implications of this behavior, including the dilaton's contribution to the cosmic acceleration and its potential role in inflation. Finally, we discuss the effects of quantum corrections on our results and their implications for the long-term behavior of the dilaton in a de Sitter universe."
#Sentence3 Human
#paragraph="In this paper, we give a new characterization of the cut locus of a point on a compact Riemannian manifold as the zero set of the optimal transport density solution of the Monge-Kantorovich equations, a PDE formulation of the optimal transport problem with cost equal to the geodesic distance. Combining this result with an optimal transport numerical solver based on the so-called dynamical Monge-Kantorovich approach, we propose a novel framework for the numerical approximation of the cut locus of a point in a manifold. We show the applicability of the proposed method on a few examples settled on 2d-surfaces embedded in $R^{3}$ and discuss advantages and limitations."
#Sentence3 GPT
#paragraph="This paper presents a method to efficiently compute the cut locus, a set of points on a Riemannian manifold where geodesics cease to be minimizing. The proposed algorithm uses optimal transport theory to exploit the regularity of the cut locus. We introduce a novel reduction algorithm that allows us to find the cut locus for a large class of manifolds with comprehensive theoretical guarantees. The proposed approach is compared against the state-of-the-art techniques on a set of benchmark surfaces, demonstrating a considerable speedup without compromising accuracy. Further experiments show the practical significance of the algorithm in applications where the computation of the cut locus is required, such as collision avoidance and visibility computations."
#Sentence4 Human
#paragraph="Biomolecular structures are assemblies of emergent anisotropic building modules such as uniaxial helices or biaxial strands. We provide an approach to understanding a marginally compact phase of matter that is occupied by proteins and DNA. This phase, which is in some respects analogous to the liquid crystal phase for chain molecules, stabilizes a range of shapes that can be obtained by sequence-independent interactions occurring intra- and intermolecularly between polymeric molecules. We present a singularityf ree self-interaction for a tube in the continuum limit and show that this results in the tube being positioned in the marginally compact phase. Our work provides a unified framework for understanding the building blocks of biomolecules."
#Sentence4 GPT
#paragraph="This paper explores the structural motifs found within various biomolecules. A structural motif is defined as a recurring pattern within a molecule's overall architecture that serves a specific function. Through an analysis of the literature, we have identified several key motifs found in biomolecules, such as alpha helices, beta sheets, and beta turns. These motifs play crucial roles in determining the unique function and behavior of the respective biomolecules. This paper presents a comprehensive overview of these motifs, including their structural characteristics, mechanisms for formation, and functional significance. Understanding the structural motifs of biomolecules is crucial for the development of new therapeutic drugs and tools in the field of molecular biology."
#Senetence5 Human
paragraph="HiYes you can. Add lot of garlic and fenugreek in your food. If non vegetarian then you can take meat egg and fish. Drink at least 3 to 4 liters of fluids per day.Have a healthy diet comprising lots of seasonal fruits & vegetables, with atleast 1 glass of milk every day.Enjoy your motherhood. Hope I have answered your query. Let me know if I can assist you further. RegardsDr.Saranya Ramadoss, General and Family Physician"
#Sentence5  GPT
#paragraph="It's generally safe to take Lactare capsules while breastfeeding, as they contain lactose and other ingredients that are thought to be safe for nursing mothers and their infants. However, it's always important to be cautious when taking any medication, even if it's available over the counter. It's always a good idea to talk to your healthcare provider before starting any new medication, including Lactare capsules, to make sure it's appropriate for you and your baby.\n\nThere is limited information available on the long-term safety of taking Lactare capsules, as they are not intended for long-term use. It's generally recommended to use them for a short period of time to help improve milk production, rather than continuing to take them for an extended period.\n\nIf you have any concerns about the safety of taking Lactare capsules or any other medication while breastfeeding, it's important to discuss them with your healthcare provider. They can help you weigh the potential risks and benefits and make an informed decision that's best for you and your baby."
tokens = word_tokenize(paragraph.lower())

# Tokenize the text

# Load English stopwords
stop_words = set(stopwords.words('english'))

# Count stopwords
stop_word_count = sum(1 for word in tokens if word in stop_words)

print("Number of stop words:", stop_word_count)
# Initialize SpellChecker
spell = SpellChecker()
misspelled = spell.unknown(tokens)
num_spelling_errors = len(misspelled)

# ========== GRAMMAR ERRORS ==========
# Initialize grammar checker (English)
tool = language_tool_python.LanguageTool('en-US')
matches = tool.check(paragraph)
num_grammar_errors = len(matches)
# Set the N for N-gram
N = 2  # For bigrams, change to 3 for trigrams, etc.

# Create N-grams
ngram_list = list(ngrams(tokens, N))

# Count N-gram frequencies
ngram_freq = Counter(ngram_list)
'''
# Show all N-grams with their frequencies
print("\nAll N-grams with frequencies:")
for ng, freq in ngram_freq.items():
    print(f"{ng}: {freq}")
'''
# Most common N-grams
common_ngrams = ngram_freq.most_common(3)
print("\nMost Common N-grams:")
for ng, freq in common_ngrams:
    print(f"{ng}: {freq}")

# Rare N-grams (appear only once)
'''
rare_ngrams = [ng for ng, freq in ngram_freq.items() if freq == 1]
print("\nRare N-grams (frequency = 1):")
for ng in rare_ngrams:
    print(ng)
'''
# List of discourse markers (extend as needed)
discourse_markers = [
    "however", "therefore", "also", "but", "although", "meanwhile", "in addition",
    "moreover", "furthermore", "on the other hand", "consequently", "thus",
    "then", "next", "finally", "because", "so", "since", "nonetheless", "still"
]

# Count total discourse markers
total_DiscourseMarkerscount = 0
for marker in discourse_markers:
    pattern = r'\b' + re.escape(marker) + r'\b'
    matches = re.findall(pattern, paragraph)
    total_DiscourseMarkerscount += len(matches)

# Output the total count
print("Total discourse markers found:", total_DiscourseMarkerscount)

# Tokenization
sentences = sent_tokenize(paragraph)
words = word_tokenize(paragraph)
words_no_punct = [w for w in words if w.isalnum()]

# POS tagging
pos_tags = pos_tag(words)
pos_counts = Counter(tag for _, tag in pos_tags)

special_chars = re.findall(r'[^a-zA-Z0-9\s]', paragraph)

# --- Statistical Features ---
statistical_features = {
    "num_sentences": len(sentences),
    "num_words": len(words_no_punct),
    "num_characters": len(paragraph),
    "total_syllables": textstat.syllable_count(paragraph),
    "special_characters_count": len(special_chars),
    "discourse_markers_count":total_DiscourseMarkerscount,
    "stop_word_count":stop_word_count,
    "avg_word_length": round(sum(len(w) for w in words_no_punct) / len(words_no_punct), 2),
    "avg_sentence_length_words": round(len(words_no_punct) / len(sentences), 2),
    "punctuation_count": sum(1 for ch in paragraph if ch in string.punctuation),
    "unique_words": len(set(words_no_punct)),
    "type_token_ratio": round(len(set(words_no_punct)) / len(words_no_punct), 2),
    "digit_count": len(re.findall(r'\d', paragraph)),
    "uppercase_words": len([w for w in words if w.isupper()]),
    "num_spelling_errors":num_spelling_errors,
    "num_grammar_errors":num_grammar_errors
}

# --- Syntactic Features ---
syntactic_features = {
    "num_nouns": sum(1 for _, tag in pos_tags if tag.startswith('NN')),
    "num_verbs": sum(1 for _, tag in pos_tags if tag.startswith('VB')),
    "num_adjectives": sum(1 for _, tag in pos_tags if tag.startswith('JJ')),
    "num_adverbs": sum(1 for _, tag in pos_tags if tag.startswith('RB')),

    "pos_tag_distribution": dict(pos_counts)
}

# --- Dependency Parsing & Tree Depth with spaCy ---
doc = nlp(paragraph)

# Calculate parse tree depth (max depth per sentence)
def get_tree_depth(sent):
    def node_depth(token):
        if not list(token.children):
            return 1
        else:
            return 1 + max(node_depth(child) for child in token.children)
    return max(node_depth(sent.root), 1)

tree_depths = [get_tree_depth(sent) for sent in doc.sents]
avg_tree_depth = round(sum(tree_depths) / len(tree_depths), 2)

dependency_features = {
    "avg_dependency_tree_depth": avg_tree_depth,
    "max_dependency_tree_depth": max(tree_depths),
    "num_subordinate_clauses": sum(1 for token in doc if token.dep_ in ['advcl', 'csubj', 'ccomp', 'xcomp'])
}

# --- Syntactic Complexity Metrics ---
readability_metrics = {
    "flesch_reading_ease": textstat.flesch_reading_ease(paragraph),
    "fog_index": textstat.gunning_fog(paragraph),
    "flesch_kincaid_grade": textstat.flesch_kincaid_grade(paragraph),
    "smog_index": textstat.smog_index(paragraph)
}

# --- Print Results ---
print("\nStatistical Features:\n----------------------")
for k, v in statistical_features.items():
    print(f"{k}: {v}")

print("\nSyntactic Features:\n----------------------")
for k, v in syntactic_features.items():
    if k == "pos_tag_distribution":
        print("POS Tag Distribution:")
        for tag, count in v.items():
            print(f"  {tag}: {count}")
    else:
        print(f"{k}: {v}")

print("\nDependency Parsing Features:\n----------------------")
for k, v in dependency_features.items():
    print(f"{k}: {v}")

print("\nSyntactic Complexity Metrics:\n----------------------")
for k, v in readability_metrics.items():
    print(f"{k}: {v}")
    
    # Process text
doc = nlp(paragraph)

# POS categories to extract
target_pos = {
    "NOUN": "Nouns",
    "NUM": "Numerals",
    "PROPN": "Proper Nouns",
    "PRON": "Pronouns",
    "DET": "Determiners",
    "VERB": "Verbs",
    "AUX": "Auxiliary Verbs",
    "ADV": "Adverbs"
}

# Count POS tags
pos_counts = Counter(token.pos_ for token in doc)

# Filter only the relevant POS tags
filtered_pos_counts = {target_pos[pos]: pos_counts.get(pos, 0) for pos in target_pos}

# Display POS tag distribution
print("Part-of-Speech Tag Distribution:\n")
for tag, count in filtered_pos_counts.items():
    print(f"{tag}: {count}")
