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





