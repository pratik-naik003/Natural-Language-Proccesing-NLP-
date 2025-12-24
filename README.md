# 📘 Introduction to Natural Language Processing (NLP)

## Simple English Notes

---

## 1️⃣ What is NLP?

**Natural Language Processing (NLP)** is a field of study that helps computers **understand, process, and generate human language**.

It is a combination of:

* **Linguistics** → understanding language rules
* **Computer Science** → programming & algorithms
* **Artificial Intelligence (AI)** → making machines intelligent

### 🔑 Main Goal of NLP

To make machines:

* Understand what humans say or write
* Respond back in **natural human language**

NLP is not only about understanding text, but also about **speaking, replying, and interacting naturally**.

---

## 2️⃣ Why is NLP Important? (Need of NLP)

Humans became powerful mainly because of:

* Communication
* Language

We shared ideas, taught future generations, and improved over time.

### 🚀 Next Big Revolution

The next big step in human progress is:

> **Talking to machines the same way we talk to humans**

### Evolution of Interaction

* **Earlier**:

  * Machines → buttons, switches
  * Computers → programming languages
  * Smartphones → touch

* **Now**:

  * Voice
  * Text
  * Conversation

### Examples

* Google Assistant
* Siri
* Alexa
* Chatbots

👉 NLP makes machines **friendly, usable, and intelligent** for everyone.

---

## 3️⃣ Real-World Applications of NLP

### 🟢 Everyday Applications

* Chatbots (customer support)
* Voice assistants
* Google Search smart answers
* Auto-reply emails
* Spam email detection

### 🟢 Business & Industry

* Targeted advertisements
* Product review analysis
* Customer feedback analysis

### 🟢 Social Media

* Hate-speech detection
* Adult content filtering
* Sentiment analysis on tweets
* Trend & opinion mining

### 🟢 Search Engines

* Direct answers (no need to open websites)
* Knowledge graphs
* Smart suggestions

---

## 4️⃣ Common NLP Tasks (Very Important)

If you want to become an **NLP Engineer**, these tasks are core skills:

### 1️⃣ Text / Document Classification

Classifying text into categories:

* Sports
* Politics
* Technology
* Spam / Not spam

---

### 2️⃣ Sentiment Analysis

Finding emotions from text:

* Positive
* Negative
* Neutral

Used in:

* Product reviews
* Movie reviews
* Social media

---

### 3️⃣ Information Extraction

Extracting useful information:

* Names
* Locations
* Dates
* Product names

---

### 4️⃣ Parts of Speech (POS) Tagging

Assigning labels to words:

* Noun
* Verb
* Adjective

Helps machines understand **sentence structure**.

---

### 5️⃣ Language Detection & Translation

* Detect language automatically
* Translate text (Google Translate)

---

### 6️⃣ Speech to Text & Text to Speech

* Voice → Text
* Text → Voice

Used in:

* Voice assistants
* Accessibility tools

---

### 7️⃣ Text Summarization

* Convert long text into short summary
* Example: News apps like **Inshorts**

---

### 8️⃣ Topic Modeling

Finding main topics in large text data

**Example**:

* Cricket article → IPL, players, matches

---

### 9️⃣ Text Generation

* Predict next word
* Auto-complete sentences
* ChatGPT-like systems

---

### 🔟 Spell Checking & Grammar Correction

* Detect spelling mistakes
* Suggest corrections

---

## 5️⃣ Approaches Used in NLP

There are **3 main approaches** used in NLP:

---

### 🔹 1. Rule-Based / Heuristic Approach (Old)

* Uses manual rules

**Example**:

* Count positive words vs negative words

**Uses**:

* Regular Expressions
* Dictionaries (WordNet)

✅ Fast & accurate
❌ Not scalable

---

### 🔹 2. Machine Learning Based Approach

* Data-driven
* Learns patterns automatically
* Text is converted into numbers (vectorization)

**Common algorithms**:

* Naive Bayes
* Logistic Regression
* SVM
* Hidden Markov Models (POS tagging)

✅ Better than rules
❌ Needs feature engineering

---

### 🔹 3. Deep Learning Based Approach (Modern)

**Big advantages**:

* Understands word order (sequence)
* Automatically learns features

**Popular models**:

* RNN
* LSTM
* GRU
* Transformers

🚀 **Transformers changed NLP completely**.

👉 All modern models like **BERT, GPT, Gemini** use Transformers.

---

## 6️⃣ Challenges in NLP (Very Important)

NLP is hard because **language is complex**.

### ⚠️ Major Challenges

1️⃣ **Ambiguity**

* One sentence → multiple meanings

2️⃣ **Context Dependency**

* Same word → different meaning

3️⃣ **Slang & Idioms**

* “Pulling someone’s leg”

4️⃣ **Sarcasm**

* “Great job” (could mean bad)

5️⃣ **Spelling Mistakes**

* Humans understand, machines struggle

6️⃣ **Creativity**

* Poems, metaphors, jokes

7️⃣ **Multiple Languages**

* Thousands of languages
* Very little data for many languages

---

## 7️⃣ Why NLP is Still Evolving

* Language changes constantly
* New slang appears
* New contexts arise
* New cultures & dialects exist

👉 We are using **less than 5%** of NLP’s full potential today.

---

## 8️⃣ Final Takeaway

* NLP is **powerful but challenging**
* It enables **human–machine communication**
* Used everywhere in modern technology
* Future growth depends heavily on NLP

---
# 📘 NLP Pipeline – Simple English Notes

---

## 1. Why this lecture is important

* NLP is a **difficult topic**
* Before jumping to algorithms, you must understand **how to think**
* This lecture teaches **how to approach any ML / NLP problem**
* In real companies, you don’t just apply models — you build **end-to-end systems**

---

## 2. What is an NLP Pipeline?

An **NLP Pipeline** is a **series of steps** followed to build a complete NLP software system.

👉 You cannot directly apply ML algorithms
👉 You must go **step by step**

---

## 3. NLP Pipeline – 5 Main Steps

1. **Data Acquisition**
2. **Text Preparation (Preprocessing)**
3. **Feature Engineering**
4. **Modeling + Evaluation**
5. **Deployment**

---

## 4. Step 1: Data Acquisition

### Meaning

Collecting text data for your NLP task.

📌 Without data → **NLP system is impossible**

---

### Data availability scenarios

#### Case 1: Data already available (Internal data)

**Examples:**

* CSV file
* Company database

**What to do:**

* If CSV → directly use it
* If database → talk to data engineering team

---

#### Case 2: Data available externally

**Sources:**

* Public datasets
* Web scraping (using BeautifulSoup)
* APIs (using `requests`)
* PDFs → extract text
* Images → OCR
* Audio → Speech-to-Text

⚠️ **Challenges:**

* Noisy data
* Website structure changes
* Unwanted text

---

#### Case 3: No data available

**What to do:**

* Collect data manually (forms, feedback)
* Label data yourself
* Start with rule-based system
* Move to ML later when data increases

---

### Data Augmentation (when data is less)

Create **synthetic data** from existing data.

**Techniques:**

* Synonym replacement
* Word order change
* Back translation
* Adding small noise

**Purpose:**

* Increase dataset size
* Improve model performance

---

## 5. Step 2: Text Preparation (Text Preprocessing)

### Goal

Make text **clean and machine-readable**

---

### Three levels of preprocessing

#### A. Basic Cleaning

* Remove HTML tags
* Handle emojis (Unicode normalization)
* Fix spelling mistakes
* Remove unwanted symbols

---

#### B. Basic Text Preprocessing (Most important)

**Common steps:**

* Tokenization (sentence & word)
* Lowercasing
* Stopword removal
* Punctuation removal
* Digit removal (optional)
* Language detection (optional)

📌 Used in **almost every NLP project**

---

#### C. Advanced Text Preprocessing

Used in complex NLP systems (chatbots, QA systems).

**Techniques:**

* POS Tagging (Part of Speech)
* Parsing (sentence structure)
* Coreference Resolution (he, she, it → who?)

---

## 6. Step 3: Feature Engineering

### Meaning

Convert **text into numbers**.

📌 ML & DL models work **only on numbers**

---

### Simple example

For each review:

* Count positive words
* Count negative words
* Count neutral words

**Text → Numbers → Model**

---

### Common Feature Engineering Techniques

* Bag of Words (BoW)
* TF-IDF
* Word Embeddings
* Deep Learning embeddings

---

### ML vs Deep Learning (Important)

#### Machine Learning

* You manually create features
* Needs domain knowledge
* Results are interpretable

❌ **Disadvantage:** More effort

---

#### Deep Learning

* Features are learned automatically
* Needs large data
* Works like a black box

❌ **Disadvantage:** Less interpretability

---

## 7. Step 4: Modeling & Evaluation

### Modeling

Apply algorithms on features.

**Approaches:**

* Rule-based (very less data)
* Machine Learning algorithms
* Deep Learning models
* Cloud APIs (ready-made solutions)

**Choice depends on:**

* Amount of data
* Nature of problem

---

### Evaluation (Very important)

#### Intrinsic Evaluation (Technical)

* Accuracy
* Precision
* Recall
* Confusion Matrix
* Perplexity (for text generation)

---

#### Extrinsic Evaluation (Business)

* User engagement
* Click rate
* Product usage

📌 Good accuracy ≠ good product
📌 Business metrics matter

---

## 8. Step 5: Deployment

Making the model usable by users.

### Deployment stages

#### Deployment

* API / Microservice
* App / Website / Chatbot

#### Monitoring

* Track model performance
* Dashboards
* Detect performance drop

#### Updating

* Retrain with new data
* Replace old models
* Online learning (optional)

---

## 9. Important Points to Remember

* NLP pipeline is **not linear**
* You may go back and forth between steps
* Pipeline differs for ML and Deep Learning
* Real-world projects are **iterative**

---

## 10. Assignment (Given in lecture)

### Problem

Detect **duplicate questions on Quora**.

### You must think about:

* Data source
* Text cleaning steps
* Feature engineering approach
* Algorithm selection
* Evaluation metrics
* Deployment strategy
* Monitoring & updates

📌 No coding required
📌 Focus on **thinking process**

---

## 11. Final Summary

* NLP is not just algorithms
* Pipeline thinking is critical
* **Data → Clean → Features → Model → Deploy**
* This approach helps in **real industry projects**

---
# 📘 Text Preprocessing in NLP – Simple English Notes (Part 1)

---

## 1. Introduction: Where Text Preprocessing Fits

In an NLP Pipeline, the main steps are:

* Data Acquisition
* Text Preprocessing ✅ (This video focuses on this)
* Feature Engineering
* Modeling
* Deployment

👉 Once you collect raw text data, you cannot directly apply ML models.
👉 You must clean and preprocess the text first.

### Types of Text Preprocessing

* **Basic Text Preprocessing** (covered here)
* **Advanced Text Processing** (POS tagging, parsing, coreference resolution – future videos)

---

## 2. Why Text Preprocessing Is Important

* Raw text is dirty and inconsistent
* Same word may appear in different formats
* Noise increases model complexity

Cleaning text improves:

* Accuracy
* Speed
* Model understanding

⚠️ **Important Rule:**
You do NOT apply all steps to every dataset.
You choose steps based on the problem and data type.

---

## 3. Lowercasing

### What is Lowercasing?

Convert all characters in text to lowercase.

### Why?

* “Basic” and “basic” should be treated as the same word
* Reduces duplicate vocabulary
* Simplifies model learning

### Example

**Without lowercasing:**

Basic ≠ basic

**With lowercasing:**

basic = basic

### Python Code (Single Text)

```python
text = "This Movie Is AMAZING"
text.lower()
```

### Python Code (Dataset – Pandas)

```python
df['review'] = df['review'].str.lower()
```

📌 This is usually the first step in text preprocessing.

---

## 4. Removing HTML Tags

### Problem

When text is scraped from websites, it often contains HTML tags like:

* `<p>`, `<br>`, `<a>`, `<div>`

These tags:

* Help browsers
* Do NOT help ML models

### Solution: Remove HTML Tags

### Python Code (Using Regex)

```python
import re

def remove_html_tags(text):
    pattern = re.compile('<.*?>')
    return re.sub(pattern, '', text)
```

### Example

```python
text = "<p>This movie was <b>great</b></p>"
remove_html_tags(text)
```

✅ **Output:**

```
This movie was great
```

### Apply to Dataset

```python
df['review'] = df['review'].apply(remove_html_tags)
```

---

## 5. Removing URLs

### Problem

Text from:

* Twitter
* WhatsApp
* Instagram
* Reviews

Often contains URLs like:

* [https://example.com](https://example.com)
* [www.site.com](http://www.site.com)

URLs:

* Add noise
* Do not contribute to sentiment or meaning

### Python Code

```python
def remove_urls(text):
    pattern = re.compile(r'https?://\S+|www\.\S+')
    return re.sub(pattern, '', text)
```

### Example

```python
text = "Check this https://google.com now!"
remove_urls(text)
```

✅ **Output:**

```
Check this  now!
```

---

## 6. Removing Punctuation

### Why Remove Punctuation?

* Punctuation becomes separate tokens
* Increases vocabulary size unnecessarily
* Confuses ML models

### Example

Hello! → "Hello" + "!"

### Method 1 (Slow – NOT recommended for big data)

```python
import string

def remove_punctuation(text):
    for char in string.punctuation:
        text = text.replace(char, '')
    return text
```

❌ Very slow for large datasets

### Method 2 (FAST & Recommended)

```python
import string

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))
```

✅ Much faster
✅ Best for large datasets

### Apply to Dataset

```python
df['tweet'] = df['tweet'].apply(remove_punctuation)
```

---

## 7. Chat Word Treatment (Short Forms)

### Problem

In chat and social media, people use shortcuts:

| Short Form | Meaning              |
| ---------- | -------------------- |
| GN         | Good Night           |
| IMO        | In My Opinion        |
| IMHO       | In My Honest Opinion |
| BRB        | Be Right Back        |

ML models cannot understand shortcuts.

### Solution

Replace shortcuts using a dictionary

### Example Dictionary

```python
chat_words = {
    "gn": "good night",
    "imo": "in my opinion",
    "imho": "in my honest opinion",
    "brb": "be right back"
}
```

### Python Code

```python
def chat_word_conversion(text):
    words = text.split()
    converted_words = []

    for word in words:
        if word.lower() in chat_words:
            converted_words.append(chat_words[word.lower()])
        else:
            converted_words.append(word)

    return " ".join(converted_words)
```

### Example

```python
text = "IMHO this movie is great GN"
chat_word_conversion(text)
```

✅ **Output:**

```
in my honest opinion this movie is great good night
```

---

## 8. Spelling Correction

### Problem

Misspellings create multiple versions of the same word

Example:

notebook ≠ noteebok

This:

* Increases vocabulary
* Reduces model accuracy

### Using TextBlob (Simple & Effective)

```python
from textblob import TextBlob

text = "I luvv this moovie"
corrected_text = str(TextBlob(text).correct())
print(corrected_text)
```

✅ **Output:**

```
I love this movie
```

⚠️ **Note:**

* Works well for common English
* For domain-specific data, custom spell checkers may be needed

---

## 9. Removing Stop Words

### What are Stop Words?

Common words that:

* Help sentence formation
* Do NOT add meaning

Examples:

is, am, are, the, a, an, in

### Why Remove?

* Reduce noise
* Improve feature quality

⚠️ **Do NOT remove stop words for:**

* POS tagging
* Grammar analysis

### Using NLTK

```python
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
```

### Python Code

```python
def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return " ".join(filtered_words)
```

### Example

```python
text = "this is a very good movie"
remove_stopwords(text)
```

✅ **Output:**

```
good movie
```

---

## 10. Handling Emojis

### Problem

ML models cannot understand emojis directly

### Two Options

* Remove emojis ❌
* Replace emojis with meaning ✅ (Better)

### Option 1: Remove Emojis

```python
import re

def remove_emojis(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)
```

### Option 2: Replace Emojis with Meaning (Recommended)

```python
import emoji

def replace_emojis(text):
    return emoji.demojize(text)
```

### Example

```python
text = "I love this movie 😍🔥"
replace_emojis(text)
```

✅ **Output:**

```
I love this movie :smiling_face_with_heart_eyes: :fire:
```

---

## 🔑 Key Takeaways

* Text preprocessing is mandatory
* Choose steps based on data
* For large datasets → always prefer efficient methods
* Clean text = Better ML models

  # 📘 Text Preprocessing – Part 2

## Tokenization, Stemming & Lemmatization (Simple English Notes)

---

## 1. Tokenization (MOST IMPORTANT STEP)

### What is Tokenization?

Tokenization is the process of breaking text into smaller units called **tokens**.

Tokens can be:

* Words
* Sentences
* Subwords (advanced)

### Example

**Sentence:**

```
I am an Indian
```

**Word Tokens:**

```
["I", "am", "an", "Indian"]
```

**Sentence Tokens:**

```
["I am an Indian.", "I love my country."]
```

---

## 2. Why Tokenization is Important?

* Machine Learning models do not understand raw text
* They work only with numbers

Before converting text into numbers:

* We must split text properly

❌ Wrong tokenization → wrong features → bad model performance

### Real Example

**Sentence:**

```
I am new in New Delhi
```

If tokenization is wrong:

```
new ≠ New
```

Correct tokenization + lowercasing:

```
["i", "am", "new", "in", "new", "delhi"]
```

✅ Unique words = 4 (not 5)

👉 Wrong tokenization confuses the model

---

## 3. Types of Tokenization

### 1️⃣ Word Tokenization

* Split sentence into words

### 2️⃣ Sentence Tokenization

* Split paragraph into sentences

📌 Choice depends on:

* Your project
* Feature engineering logic

📌 Most NLP tasks use **word tokenization**

---

## 4. Problems in Tokenization (Real-World Challenges)

Tokenization is **NOT** as simple as splitting by space.

### Common Issues:

* **Punctuation** → `Delhi!`
* **Numbers + units** → `5km → 5 + km`
* **Email IDs** → `help@gmail.com`
* **Abbreviations** → `U.S., Ph.D.`
* **Hyphenated words** → `well-known`

👉 Simple `.split()` fails in these cases

---

## 5. Tokenization Techniques

### 🔹 Technique 1: Python `split()` (VERY BASIC)

```python
text = "I am going to Delhi!"
tokens = text.split()
print(tokens)
```

❌ Output:

```
['I', 'am', 'going', 'to', 'Delhi!']
```

❌ Problem:

```
Delhi! ≠ Delhi
```

📌 Use only for very clean text

---

### 🔹 Technique 2: Regular Expressions (Better but Complex)

```python
import re

text = "I am going to Delhi!"
tokens = re.findall(r'\w+', text)
print(tokens)
```

✅ Output:

```
['I', 'am', 'going', 'to', 'Delhi']
```

⚠️ Still has problems with:

* Emails
* Units (5km)
* Abbreviations

---

### 🔹 Technique 3: NLTK Tokenizer (GOOD)

#### Word Tokenization

```python
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

text = "I am going to Delhi!"
tokens = word_tokenize(text)
print(tokens)
```

✅ Output:

```
['I', 'am', 'going', 'to', 'Delhi', '!']
```

#### Sentence Tokenization

```python
from nltk.tokenize import sent_tokenize

text = "I love India. Delhi is my city!"
sentences = sent_tokenize(text)
print(sentences)
```

✅ Output:

```
['I love India.', 'Delhi is my city!']
```

⚠️ NLTK is good, but not perfect

---

### 🔹 Technique 4: spaCy Tokenizer (BEST ⭐)

📌 Industry-recommended tokenizer

Handles:

* Emails
* Numbers + units
* Abbreviations
* Punctuation
* Complex grammar

#### Installation

```bash
pip install spacy
python -m spacy download en_core_web_sm
```

#### Word Tokenization using spaCy

```python
import spacy

nlp = spacy.load("en_core_web_sm")

text = "Email us at help@gmail.com or walk 5km today!"
doc = nlp(text)

tokens = [token.text for token in doc]
print(tokens)
```

✅ Output:

```
['Email', 'us', 'at', 'help@gmail.com', 'or', 'walk', '5', 'km', 'today', '!']
```

📌 spaCy is the best choice for real projects

---

## 6. Key Takeaway on Tokenization

* Tokenization is NOT trivial
* Wrong tokens → wrong features
* No tokenizer is 100% perfect

📌 Use:

* **spaCy** → complex / real-world text
* **NLTK** → simple tasks

---

## 🌱 Stemming

## 7. What is Stemming?

Stemming reduces words to their root form by removing prefixes/suffixes.

⚠️ The root word may NOT be a real English word.

### Example

| Word    | Stem  |
| ------- | ----- |
| working | work  |
| worked  | work  |
| studies | studi |
| movies  | movi  |

---

## 8. Why Use Stemming?

* Reduces vocabulary size
* Groups similar words
* Useful in Information Retrieval (Search Engines)

---

## 9. Stemming using NLTK (Porter Stemmer)

```python
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

stemmer = PorterStemmer()

text = "working worked works"
tokens = word_tokenize(text)

stemmed_words = [stemmer.stem(word) for word in tokens]
print(stemmed_words)
```

✅ Output:

```
['work', 'work', 'work']
```

⚠️ Problem:

```
studies → studi
```

---

## 🍃 Lemmatization

## 10. What is Lemmatization?

Lemmatization converts words to their dictionary (meaningful) base form.

✅ Output is always a real English word

---

## 11. Difference: Stemming vs Lemmatization

| Feature  | Stemming       | Lemmatization    |
| -------- | -------------- | ---------------- |
| Speed    | Fast           | Slow             |
| Output   | Not real words | Real words       |
| Logic    | Rule-based     | Dictionary-based |
| Accuracy | Lower          | Higher           |

---

## 12. Lemmatization using NLTK (WordNet)

```python
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

text = "movies are running better"
tokens = word_tokenize(text)

lemmatized_words = [lemmatizer.lemmatize(word) for word in tokens]
print(lemmatized_words)
```

✅ Output:

```
['movie', 'are', 'running', 'better']
```

📌 Better results when POS tag is provided

---

## 13. When to Use What?

### Use **Stemming** when:

* Speed matters
* Output not shown to user
* Search engines, IR systems

### Use **Lemmatization** when:

* Output shown to user
* Grammar matters
* Chatbots, summaries, NLP apps

---

## 14. Final Summary

✅ Tokenization is the foundation of NLP

✅ spaCy gives the best tokenization

✅ Stemming is fast but rough

✅ Lemmatization is slow but accurate

✅ Choose preprocessing steps based on use-case

---

## 📝 Assignment (From Video)

Create your **own dataset** (do NOT use ready datasets):

**Movie Dataset Columns:**

* Movie Name
* Description (text)
* Genre (label)

Apply the following steps:

* Lowercasing
* Remove HTML
* Remove URLs
* Remove punctuation
* Stopwords removal
* Tokenization
* Stemming / Lemmatization

📌 Learning matters more than using ready datasets

# 📘 Text Representation / Feature Extraction from Text

*(Simple English + Deep Explanation Notes – Part 1)*

---

## 1️⃣ Where This Topic Fits in NLP Pipeline

Recall the NLP Pipeline:

1. Data Collection
2. Text Preprocessing
3. **Text Representation / Feature Extraction ✅ (THIS VIDEO)**
4. Modeling
5. Evaluation
6. Deployment

👉 After preprocessing, text must be converted into numbers.
👉 Machine Learning algorithms do **not** understand text; they understand **numbers only**.

---

## 2️⃣ What is Feature Extraction from Text?

### Simple Definition

Feature Extraction from Text means:

> Converting raw text into numerical features so that machine learning algorithms can process it.

It is also called:

* Text Representation
* Text Vectorization

### Why “Vectorization”?

Because text is finally converted into **vectors (arrays of numbers)**.

### Example Problem – Sentiment Analysis

Text:

> "This movie is very good"

Output:

* Positive / Negative

⚠️ ML models do **not** understand English words.
They understand **math and numbers**.

So the flow is:

> Text → Numbers → ML Model → Output

---

## 3️⃣ Why Do We Need Feature Extraction?

### Famous ML Rule

> **Garbage In → Garbage Out**

* Bad features → Bad output
* Good features → Good output

Even:

> A simple algorithm + good features
> can outperform
> a complex algorithm + bad features

### Text Example (Sentiment Analysis)

Possible numerical features:

* Number of positive words
* Number of negative words
* Length of sentence
* Average word length

👉 These act as **numeric signals** for ML models.

---

## 4️⃣ Why is Text → Numbers Difficult?

### Compare with Other Data Types

#### 🖼️ Image Data

* Image = Pixels
* Pixels = Numbers
  ✔ Easy to convert

#### 🔊 Audio Data

* Audio = Waveform
* Waveform = Amplitude values
  ✔ Easy to convert

#### 📝 Text Data

Example:

> "Hello, how are you?"

Problems:

* ❌ No direct numeric form
* ❌ Meaning is abstract
* ❌ Order, context, emotion matter

👉 That’s why **text representation is hard**.

---

## 5️⃣ Core Idea Behind Text Representation (VERY IMPORTANT)

While converting text to numbers,
**semantic meaning must be preserved**.

If numbers do not capture meaning:
→ Model performance will be poor.

Success depends on:

* How well meaning is encoded
* Similar texts getting similar vectors
* Different texts getting distant vectors

---

## 6️⃣ Popular Text Representation Techniques

### Techniques Overview

* One Hot Encoding
* Bag of Words (BoW)
* N-grams
* TF-IDF
* Custom Hand-crafted Features
* Word Embeddings (Word2Vec, GloVe) *(Next video)*
* Deep Learning embeddings *(Later)*

👉 This video covers:

* One Hot Encoding
* Bag of Words

---

## 7️⃣ Important Terminology (Very Important)

### Corpus

All text data combined together.

### Vocabulary

Set of **unique words** in the corpus.

### Document

One individual text (review, sentence, email).

### Word / Token

Single unit inside a document.

### Example (IMDB Dataset)

* 50,000 reviews → 50,000 documents
* Each review = one document
* All words together = corpus
* Unique words = vocabulary

---

## 8️⃣ One Hot Encoding (First Technique)

### Idea

Each word is represented as a **binary vector**.

### Example Dataset

| Document | Sentence             |
| -------- | -------------------- |
| D1       | people watch campus  |
| D2       | watch campus campus  |
| D3       | people write comment |
| D4       | campus write comment |

### Step 1: Build Vocabulary

Vocabulary = [people, watch, campus, write, comment]

Size = 5

### Step 2: One Hot Representation

| Word    | Vector      |
| ------- | ----------- |
| people  | [1,0,0,0,0] |
| watch   | [0,1,0,0,0] |
| campus  | [0,0,1,0,0] |
| write   | [0,0,0,1,0] |
| comment | [0,0,0,0,1] |

### Document Representation

A document is represented by combining vectors of its words.

---

## 9️⃣ Problems with One Hot Encoding ❌

### ❌ 1. Sparse Vectors

* Vocabulary size = 50,000
* Vector length = 50,000
* Only one `1`, rest are `0`

👉 High memory and computation cost

### ❌ 2. Not Fixed Input Size

* Different sentences → different shapes
* ML models need **fixed-size input**

### ❌ 3. Out-of-Vocabulary (OOV)

* New word at prediction time → model fails

### ❌ 4. No Semantic Meaning

Example words:

* run, walk, bottle

But:

* run & walk are similar
* bottle is different

One-hot treats all equally ❌

### Final Verdict

* ❌ Not used in real applications
* ✔ Used only to understand basics

---

## 1️⃣0️⃣ Bag of Words (BoW) – MOST IMPORTANT

### Core Idea

Instead of asking:

> Is the word present?

Ask:

> How many times does the word appear?

### Vocabulary (Same as Before)

[people, watch, campus, write, comment]

### BoW Representation

| Document | people | watch | campus | write | comment |
| -------- | ------ | ----- | ------ | ----- | ------- |
| D1       | 1      | 1     | 1      | 0     | 0       |
| D2       | 0      | 1     | 2      | 0     | 0       |
| D3       | 1      | 0     | 0      | 1     | 1       |
| D4       | 0      | 0     | 1      | 1     | 1       |

✔ Fixed size vectors
✔ Works for new sentences
✔ Widely used in text classification

---

## 1️⃣1️⃣ Intuition Behind Bag of Words

Documents with:

* Similar words
* Similar frequencies

→ have similar meaning

### Vector Space View

* Each document = a point in high-dimensional space
* Similarity measured using:

  * Cosine similarity
  * Distance metrics

---

## 1️⃣2️⃣ Bag of Words Implementation (sklearn)

```python
from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "people watch campus",
    "watch campus campus",
    "people write comment",
    "campus write comment"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)

print(vectorizer.vocabulary_)
print(X.toarray())
```

### Output

**Vocabulary:**

```
{'people':0, 'watch':1, 'campus':2, 'write':3, 'comment':4}
```

**BoW Matrix:**

```
[[1 1 1 0 0]
 [0 1 2 0 0]
 [1 0 0 1 1]
 [0 0 1 1 1]]
```

---

## 1️⃣3️⃣ Important Parameters of CountVectorizer

* `lowercase=True`

  * Converts all text to lowercase

* `stop_words='english'`

  * Removes common words like: is, the, and, of

* `binary=True`

  * Uses only presence (1) or absence (0)
  * Useful in sentiment analysis

* `max_features`

  * Keeps only top-N frequent words

  ```python
  CountVectorizer(max_features=1000)
  ```

---

## 1️⃣4️⃣ Advantages of Bag of Words ✅

* ✔ Simple and intuitive
* ✔ Fixed-size input
* ✔ Handles unseen sentences
* ✔ Works well for text classification
* ✔ Better than One Hot Encoding

---

## 1️⃣5️⃣ Disadvantages of Bag of Words ❌

### ❌ 1. Sparse Vectors

Large vocabulary → many zeros

### ❌ 2. Order Ignored

"movie is good"
"good is movie"

Same vector ❌

### ❌ 3. Context Ignored

"This is good movie"
"This is not good movie"

Almost same vector ❌

### ❌ 4. Weak Semantic Understanding

Meaning is only partially captured

---

## 1️⃣6️⃣ Why We Need Next Technique (N-grams)

Problems with BoW:

* Word order ignored
* Context loss
* Negation not handled properly

👉 Next technique: **N-grams**

---

## ✅ Summary of Part-1

| Technique        | Used in Practice |
| ---------------- | ---------------- |
| One Hot Encoding | ❌ No             |
| Bag of Words     | ✅ Yes            |
| N-grams          | ✅ Yes            |
| TF-IDF           | ✅ Yes            |
| Word Embeddings  | ✅ Yes            |

---

📌 **End of Part-1: Text Representation Basics**

# 📘 Text Representation – N-grams, TF-IDF & Custom Features

(Simple English + Conceptual Notes – Part 2)

---

## 1️⃣ What are N-grams?

### Definition

N-grams are continuous sequences of **N words** taken from a sentence.

* **Unigram (1-gram)** → single word
* **Bigram (2-gram)** → two continuous words
* **Trigram (3-gram)** → three continuous words

This technique is also called **Bag of N-grams**.

### Why N-grams are Needed?

Earlier techniques (Bag of Words / Unigram) had problems:

* ❌ Ignore word order
* ❌ Miss important phrases like:

  * "not good"
  * "very bad"
  * "no improvement"

👉 Word order matters in language.
👉 N-grams solve this problem **partially**.

---

## 2️⃣ How N-grams Work (Intuition)

### Sentence

```
people watch campus
```

### Unigrams

```
people | watch | campus
```

### Bigrams

```
people watch | watch campus
```

### Trigrams

```
people watch campus
```

⚠️ **Important Rule:**

* Words must be **continuous** (no skipping words).

---

## 3️⃣ Example Dataset

| Document | Sentence             |
| -------- | -------------------- |
| D1       | people watch campus  |
| D2       | watch campus campus  |
| D3       | people write comment |
| D4       | campus write comment |

### Bigram Vocabulary (Bag of Bigrams)

* people watch
* watch campus
* campus campus
* people write
* write comment
* campus write

👉 Vocabulary size increases compared to unigrams.

---

## 4️⃣ Vector Representation with Bigrams

Each document is represented using **frequency of bigrams**.

### Example for D1: "people watch campus"

* people watch → 1
* watch campus → 1
* others → 0

✔ Fixed-size vector
✔ Captures local word order

---

## 5️⃣ N-grams using Scikit-Learn

### Important Parameter: `ngram_range`

```python
from sklearn.feature_extraction.text import CountVectorizer

# Unigram (Bag of Words)
CountVectorizer(ngram_range=(1,1))

# Bigram
CountVectorizer(ngram_range=(2,2))

# Unigram + Bigram
CountVectorizer(ngram_range=(1,2))

# Trigram
CountVectorizer(ngram_range=(3,3))
```

### Example Code (Bigram)

```python
documents = [
    "people watch campus",
    "watch campus campus",
    "people write comment",
    "campus write comment"
]

vectorizer = CountVectorizer(ngram_range=(2,2))
X = vectorizer.fit_transform(documents)

print(vectorizer.vocabulary_)
print(X.toarray())
```

---

## 6️⃣ Why N-grams are Better than Bag of Words?

### Key Example

**Sentence 1:**

```
This is a very good movie
```

**Sentence 2:**

```
This is not a good movie
```

### Problem with Unigrams / BoW

Both sentences contain:

```
this, is, good, movie
```

👉 Vectors look very similar
👉 Meanings are opposite

### Using Bigrams

* Sentence 1 bigram → **very good**
* Sentence 2 bigram → **not good**

✔ Meaning difference is captured
✔ Vectors move farther apart in vector space

---

## 7️⃣ Benefits of N-grams ✅

* Captures word order (local context)
* Handles negation ("not good")
* Better semantic representation than unigrams
* Easy to understand and implement
* Very useful in:

  * Sentiment Analysis
  * Text Classification

---

## 8️⃣ Disadvantages of N-grams ❌

### 1. Vocabulary Explosion

* Unigrams → small vocabulary
* Bigrams → larger
* Trigrams → very large

👉 High memory and computation cost

### 2. Sparse Vectors

* More features → more zeros
* Hard to handle large datasets

### 3. Out-of-Vocabulary (OOV) Problem

* New phrase at prediction time → ignored

### 4. No Deep Semantic Understanding

Example:

```
beautiful ≠ gorgeous
```

N-grams treat them as different words.

👉 This is solved using **Word Embeddings**.

---

## 9️⃣ Summary Till Now

| Technique    | Order     | Context    | Used in Practice |
| ------------ | --------- | ---------- | ---------------- |
| One Hot      | ❌         | ❌          | ❌                |
| Bag of Words | ❌         | ❌          | ✅                |
| N-grams      | ✅ (local) | ⚠️ partial | ✅                |

---

## 🔟 TF-IDF (Why We Need It)

### Problem with BoW & N-grams

* All words treated as equally important

Example words:

```
people, watch, campus
```

But in reality:

* Some words are important
* Some words are common and useless

👉 **TF-IDF fixes this problem**.

---

## 1️⃣1️⃣ What is TF-IDF?

**TF-IDF = Term Frequency × Inverse Document Frequency**

It gives higher weight to:

* Words frequent in a document
* Words rare across the corpus

---

## 1️⃣2️⃣ Term Frequency (TF)

### Formula

```
TF(word) = (Number of times word appears in document)
           -------------------------------------------
           (Total words in document)
```

👉 Measures importance **inside a document**

---

## 1️⃣3️⃣ Inverse Document Frequency (IDF)

### Formula

```
IDF(word) = log(
    Total number of documents
    -------------------------
    Number of documents containing the word
)
```

👉 Measures **rarity across corpus**

### Intuition

* Word in every document → IDF ≈ 0
* Word in few documents → IDF is high

---

## 1️⃣4️⃣ Why Log is Used in IDF?

* Without log → rare words get very large values
* TF becomes meaningless

Log function:

* Smooths extreme values
* Balances TF and IDF

---

## 1️⃣5️⃣ TF-IDF Final Formula

```
TF-IDF(word) = TF(word) × IDF(word)
```

👉 Final value shows:

* Importance in the document
* Importance in the corpus

---

## 1️⃣6️⃣ TF-IDF using Scikit-Learn

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

print(vectorizer.vocabulary_)
print(X.toarray())
```

### Note on sklearn IDF

Scikit-learn uses **smoothed IDF**:

```
idf = log((N + 1) / (df + 1)) + 1
```

* Prevents zero values
* This is only an implementation detail

---

## 1️⃣7️⃣ Where TF-IDF is Used?

* Search engines (Google)
* Information Retrieval
* Text Classification
* Document Similarity

---

## 1️⃣8️⃣ Custom Features (Hand-crafted Features)

Sometimes built-in techniques are not enough.

👉 We create **custom features** using domain knowledge.

### Examples (Sentiment Analysis)

* Number of positive words
* Number of negative words
* Ratio of positive / negative words
* Total word count
* Average word length
* Character count

### Why Custom Features Matter?

* You understand the problem better than the algorithm
* Domain knowledge improves accuracy

---

## 1️⃣9️⃣ Hybrid Features (Very Important)

In real-world ML projects, we use:

```
BoW / N-grams / TF-IDF
+
Custom Features
```

👉 This is called **Hybrid Features**
👉 Used in **99% of real ML projects**

---

## 2️⃣0️⃣ Final Big Picture

| Technique       | Solves What                   |
| --------------- | ----------------------------- |
| BoW             | Fixed size, frequency         |
| N-grams         | Word order, phrases           |
| TF-IDF          | Word importance               |
| Custom Features | Domain intelligence           |
| Word Embeddings | Semantic meaning (NEXT VIDEO) |

---

📌 **Next Topic:** Word Embeddings (Word2Vec, GloVe, FastText)

# 📘 Word Embeddings & Word2Vec (Simple English Notes)

## 1️⃣ Where this Topic Fits in NLP

In the NLP pipeline:

1. Data Collection
2. Text Preprocessing
3. **Text Representation / Feature Extraction ✅**
4. Modeling
5. Deployment

👉 **Today’s topic:** Advanced Text Representation – **Word Embeddings & Word2Vec**

---

## 2️⃣ What is Word Embedding?

### Simple Definition

Word Embedding is a technique where:

* Each word is converted into a **vector (numbers)**
* **Similar words get similar vectors**

### Example

```
king  → [0.21, -0.33, 0.58, ...]
queen → [0.20, -0.31, 0.60, ...]
```

👉 These vectors capture **meaning**, not just frequency.

---

## 3️⃣ Why Word Embeddings are Needed?

### Old Techniques

* Bag of Words (BoW)
* TF-IDF

### Problems with Old Techniques

❌ No semantic meaning
❌ Very high dimensions
❌ Sparse vectors (many zeros)

### How Word Embeddings Help

✅ Capture semantic meaning
✅ Low-dimensional vectors
✅ Dense vectors (fewer zeros)
✅ Faster computation

---

## 4️⃣ Types of Word Embedding Techniques

### 🔹 1. Frequency-based Embeddings

* Bag of Words
* TF-IDF
* GloVe

👉 Based on **word frequency**

### 🔹 2. Prediction-based Embeddings

* **Word2Vec ✅**
* FastText

👉 Based on **predicting words using neural networks**

---

## 5️⃣ What is Word2Vec?

### Definition

Word2Vec is a **prediction-based word embedding technique** that:

* Converts words into vectors
* Uses neural networks
* Learns meaning from context

📌 Developed by **Google (2013)**

---

## 6️⃣ Why Word2Vec is Better than BoW / TF-IDF?

### ✅ Major Advantages

#### 1️⃣ Semantic Meaning

* "happy" ≈ "joy"
* BoW / TF-IDF cannot capture this

#### 2️⃣ Low Dimensional

* Typically **100–300 dimensions**
* Not thousands like BoW

#### 3️⃣ Dense Vectors

* Very few zero values
* Less overfitting

👉 **Interview Answer Ready ✔️**

---

## 7️⃣ Vector Properties of Word2Vec

* 🔹 Similar words → close vectors
* 🔹 Different words → far vectors

### You Can Do:

* Similarity calculation
* Vector arithmetic

### Example

```
king - man + woman ≈ queen
```

---

## 8️⃣ Using Pre-trained Word2Vec (Google News)

### Pre-trained Model Details

* Trained on **Google News**
* ~3 million words
* 300-dimensional vectors
* File size ~1.5GB

### 🔧 Install Required Libraries

```bash
pip install gensim nltk
```

### 📌 Load Pre-trained Word2Vec Model

```python
import gensim
from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format(
    "GoogleNews-vectors-negative300.bin",
    binary=True
)
```

### 🔹 Get Vector of a Word

```python
vector = model["man"]
print(vector)
print(vector.shape)
```

**Output:**

```
(300,)
```

### 🔹 Most Similar Words

```python
model.most_similar("cricket")
```

### 🔹 Similarity Between Two Words

```python
model.similarity("man", "woman")
```

### 🔹 Odd One Out

```python
model.doesnt_match(["php", "java", "python", "banana"])
```

### 🔹 Vector Arithmetic

```python
model.most_similar(positive=["woman", "king"], negative=["man"])
```

---

## 9️⃣ Important Intuition Behind Word2Vec

### 🔑 Core Idea

> Words used in **similar contexts** have **similar meanings**

### Example

* football player
* hockey player

👉 Word2Vec learns:

```
football ≈ hockey
```

---

## 🔟 Two Architectures of Word2Vec

### 🔹 1. CBOW (Continuous Bag of Words)

* Predicts **target word** from context
* Faster
* Good for small datasets

**Context → Target**

### 🔹 2. Skip-Gram

* Predicts **context words** from target
* Slower
* Better for large datasets

**Target → Context**

---

## 1️⃣1️⃣ Training Data Creation (Sliding Window)

### Sentence

```
watch campus for data science
```

### Window Size = 3

#### CBOW Training Samples

| Input (Context) | Output (Target) |
| --------------- | --------------- |
| watch, for      | campus          |
| campus, data    | for             |
| for, science    | data            |

---

## 1️⃣2️⃣ Neural Network Structure (Conceptual)

* **Input:** One-hot vectors
* **Hidden Layer:** Embedding layer (size = embedding dimension)
* **Output:** Softmax over vocabulary
* **Loss:** Cross-entropy
* **Training:** Backpropagation

👉 **Embeddings = weights of hidden layer**

---

## 1️⃣3️⃣ Training Word2Vec on Your Own Data (Gensim)

### 📂 Dataset Example

* Game of Thrones books (text files)

### 🔹 Preprocessing

```python
from gensim.utils import simple_preprocess
from nltk.tokenize import sent_tokenize
import os

story = []

for file in os.listdir("data"):
    with open(f"data/{file}", "r", encoding="utf-8") as f:
        text = f.read()
        sentences = sent_tokenize(text)
        for sent in sentences:
            story.append(simple_preprocess(sent))
```

### 🔹 Train Word2Vec Model

```python
from gensim.models import Word2Vec

model = Word2Vec(
    sentences=story,
    vector_size=100,
    window=10,
    min_count=2,
    workers=4,
    epochs=10
)
```

### 🔹 Similar Words (Custom Model)

```python
model.wv.most_similar("daenerys")
```

### 🔹 Word Vector

```python
model.wv["jon"]
```

### 🔹 Similarity Check

```python
model.wv.similarity("arya", "sansa")
```

---

## 1️⃣4️⃣ Visualizing Word Embeddings (PCA)

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

words = list(model.wv.index_to_key)[:100]
vectors = [model.wv[word] for word in words]

pca = PCA(n_components=2)
reduced = pca.fit_transform(vectors)

plt.scatter(reduced[:,0], reduced[:,1])

for i, word in enumerate(words):
    plt.annotate(word, (reduced[i,0], reduced[i,1]))

plt.show()
```

---

## 1️⃣5️⃣ How to Improve Word2Vec Quality?

✔ Increase training data
✔ Increase embedding dimensions
✔ Increase window size
✔ Use Skip-Gram for large datasets
✔ Proper preprocessing

---

## 1️⃣6️⃣ Interview Quick Answers

**Q: Why Word2Vec over TF-IDF?**
➡ Captures semantic meaning, dense vectors, low dimension

**Q: CBOW vs Skip-Gram?**
➡ CBOW → small data
➡ Skip-Gram → large data

**Q: What does Word2Vec actually learn?**
➡ Context-based word relationships

---

## 1️⃣7️⃣ Assignment (Important)

👉 Choose any text dataset:

* Movie scripts
* News articles
* Friends / GOT scripts

### Tasks

✔ Train Word2Vec
✔ Check similarity
✔ Try vector arithmetic
✔ Visualize embeddings

---

## ✅ Final Summary

* Word2Vec is a **deep learning–based word embedding**
* Converts **words → vectors**
* Learns **meaning from context**
* Two models: **CBOW & Skip-Gram**
* **Gensim** makes implementation easy
* Very important for **NLP, ML & Interviews**







