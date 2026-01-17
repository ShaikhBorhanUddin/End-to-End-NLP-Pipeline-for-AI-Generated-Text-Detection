# End to End NLP Pipeline For AI-Generated Text Detection 

<p align="left">
  <!-- Core -->
  <img src="https://img.shields.io/badge/Made%20With-Colab-blue?logo=googlecolab&logoColor=white" alt="Made with Colab">
  <img src="https://img.shields.io/badge/Language-Python-blue?logo=python" alt="Language: Python">

  <!-- Emoji Icons -->
  <img src="https://img.shields.io/badge/⚖️%20License-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/🐞%20Issues-GitHub-red" alt="Issues">

  <!-- Repo Stats -->
  <img src="https://img.shields.io/github/repo-size/ShaikhBorhanUddin/End-to-End-NLP-Pipeline-for-AI-Generated-Text-Detection?logo=github" alt="Repo Size">
  <img src="https://img.shields.io/github/last-commit/ShaikhBorhanUddin/End-to-End-NLP-Pipeline-for-AI-Generated-Text-Detection" alt="Last Commit">

  <!-- Models -->
  <img src="https://img.shields.io/badge/🤖%20Models-LR%20%7C%20SVM%20%7C%20RF%20%7C%20XGBoost%20%7C%20BiLSTM%20%7C%20DistilBERT-red" alt="Models">

  <!-- Explainability -->
  <img src="https://img.shields.io/badge/🔍%20Explainability-LIME%20%7C%20SHAP-purple" alt="Explainability">

  <!-- Visualization -->
  <img src="https://img.shields.io/badge/📊%20Data%20Visualization-Matplotlib%20%7C%20Seaborn-yellow" alt="Visualization">

  <!-- Dataset -->
  <img src="https://img.shields.io/badge/Dataset-Kaggle-blueviolet?logo=kaggle" alt="Dataset: Kaggle">
  <!-- GPU --> 
  <img src="https://img.shields.io/badge/Runtime-GPU%20(A100 | T4 | L4)-blue?logo=nvidia" alt="Runtime: GPU (A100 | Trillium)">

  <!-- Deployment -->
  <img src="https://img.shields.io/badge/Deployment-Gradio%20%7C%20HuggingFace-orange?logo=huggingface" alt="Deployment">

  <!-- DevOps -->
  <img src="https://img.shields.io/badge/Version%20Control-Git-orange?logo=git" alt="Git">
  <img src="https://img.shields.io/badge/Host-GitHub-black?logo=github" alt="GitHub">

  <!-- Social -->
  <img src="https://img.shields.io/github/forks/ShaikhBorhanUddin/End-to-End-NLP-Pipeline-for-AI-Generated-Text-Detection?style=social" alt="Forks">

  <!-- Status -->
  <img src="https://img.shields.io/badge/Project-Deployed-brightgreen" alt="Status">
</p> 

![Dashboard](https://github.com/ShaikhBorhanUddin/End-to-End-NLP-Pipeline-For-Human-vs-AI-Generated-Text-Classification-System/blob/main/Assets/ai_vs_human_title_mod.png?raw=true) 

## Overview 

The rapid progress of large language models has transformed text generation across education, media, and digital platforms, enabling AI systems (such as ChatGPT, Claude, and LLaMA) to produce content that closely mimics human writing. While these technologies provide significant benefits, they also bring forth challenges related to academic integrity, fake information, and content authenticity, making the reliable detection of AI-generated text harder than ever. This capstone project addresses the problem of AI vs human text classification by exploring several machine learning–based systems that distinguish between human-written and AI-generated content using linguistic and stylistic patterns instead of source-specific metadata. Multiple datasets of varying sizes were considered and finally integrated and preprocessed to ensure consistency and avoid data leakage. Several classical machine learning, RNN and transformer-based models were explored, with DistilBERT selected for its strong performance and computational efficiency. To enhance transparency and usability (rather than just theoretical exploration), the project includes a Hugging Face based [application](https://huggingface.co/spaces/ShaikhBorhanUddin/AI_generated_text_detection) built with Gradio. Users can input text and receive predictions with confidence scores, along with word-level explanations generated through explainability methods such as LIME. The proposed system demonstrates a practical and interpretable approach to AI-generated text detection, with real-world applications in academic integrity, e-commerce review moderation, and social media content verification. 

## Business Use Cases 

In the age of the AI revolution, recruitment and hiring platforms focus on detecting AI-generated resumes, cover letters, and job applications used by candidates during the hiring process. Hence, this use case is particularly relevant for HR departments and recruitment platforms (such as LinkedIn, Glassdoor, and Indeed), where the rise of generative AI has enabled mass-produced applications that dilute the authenticity of candidate skills and qualifications. By identifying heavily AI-generated content, the model can assist recruiters in preserving genuine candidate communication, improving screening efficiency, and maintaining fairness in the hiring process, while serving as a support tool to enhance (but definitely not replace) human decision-making. 

Journalism & News Verification also addresses a similar growing challenge of identifying AI-generated news articles, opinion pieces, and other editorial content within modern media ecosystems. Newsrooms, fact-checking organizations, and media watchdogs can use this model to flag potentially synthetic or AI-written content that may contribute to misinformation or propaganda (political, religious, or racial). By integrating the model into editorial review workflows, it can support human verification processes, help maintain journalistic integrity, and ensure that published content meets credibility and ethical standards in an era of rapidly automated content generation. 

Academic Integrity & Education Technology (EdTech) can be another use case where focus is given on detecting AI-generated academic submissions such as essays, assignments, and reflective writing. Universities, online learning platforms (such as Coursera, edX, Udemy), and assessment or marking systems can use this model to help identify potentially AI-written (or heavily assisted) coursework. This capability is critical for preventing misuse of generative AI, supporting fair and consistent evaluation, and assisting instructors by flagging suspicious (AI-generated) submissions. Importantly, the model functions as a decision-support tool rather than an automated punishment system, enabling human educators to make informed and transparent academic integrity decisions. 

To implement these use cases, it is essential to follow a series of sequential steps: identifying suitable datasets, performing feature engineering, training and testing models, selecting the most appropriate model, and building and deploying a web interface. These processes are detailed in the following sections. 

## Folder Structure 

```bash
End to End NLP Pipeline For AI-Generated Text Detection
│
├── Assets/                       # Screenshots, visualizations and images for documentation
├── Dataset/               
│      ├── Raw/                   # Original dataset from Kaggle (3rd dataset too large to upload in GitHub)
│      ├── Preprocessed/          # Dataset with added features
│      ├── Cleaned/               # Cleaned and merged datasets
│      └── Feature Engineered/    # dataset with BoW, TF-IDF, word2vec and lemmatized features
├── Models/                       # All saved models (distilbert safetensor too large to upload in GitHub)
├── Notebooks/                    # Data preprocessing, EDA, train/test, result visualization
├── app.py                        # Code for deployment
├── requirements.txt              # Python dependencies for deployment
├── README.md                     # Project documentation
└── Licence
```
## Project Workflow 

The following workflow is maintained sequencially in this project. 

- Dataset selection from multiple sources to ensure diverse human and AI-generated text samples

- Data cleaning including duplicate removal and handling missing or noisy entries

- Text preprocessing with lowercasing, punctuation removal, tokenization, lemmatization, and stopword filtering

- EDA analyzing word count distributions, frequent terms, word clouds, bigram and linguistic complexity measures

- Feature engineering using BoW, TF-IDF, and Word2Vec embeddings

- Model selection covering classical machine learning models, BiLSTM, and transformer-based DistilBERT

- Train-test splitting to ensure unbiased performance evaluation

- Model training with hyperparameter tuning and GPU acceleration

- Model evaluation using accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices

- Result visualization and explaining with LIME and SHAP

- Deployment via a Gradio web application with real-time prediction, confidence scoring, and explainability support

## Dataset, Cleaning & Pre-processing 

Three publicly available datasets were used for this project. The first dataset [AI vs Human Comparison Dataset](https://www.kaggle.com/datasets/prince7489/ai-vs-human-comparison-dataset/data) is relatively small, with 500 entries, but contains metadata for the main feature (text). The dataset is well balanced (50.2% ai generated text entry vs. 49.8% human generated text entry). 

| **Column Name**      | **Data Type**    | **Description**                                                                 |
| -------------------- | ---------------- | ------------------------------------------------------------------------------- |
| **id**               | Integer          | Unique identifier for each record                                               |
| **label**            | String           | Indicates whether the content is generated by a *human* or *AI*                 |
| **topic**            | String           | Main subject category of the text (e.g., food, travel, education, sports, etc.) |
| **text**             | Text             | Actual written content or message                                               |
| **length_chars**     | Integer          | Number of characters present in the text                                        |
| **length_words**     | Integer          | Total word count of the text                                                    |
| **quality_score**    | Float            | Automated quality assessment score of the text                                  |
| **sentiment**        | Float            | Sentiment polarity score (negative to positive range)                           |
| **source_detail**    | String           | Origin of the text (e.g., human author ID or AI model name)                     |
| **timestamp**        | DateTime         | Date and time when the text was created                                         |
| **plagiarism_score** | Float            | Similarity score indicating potential plagiarism                                |
| **notes**            | String           | Additional remarks (e.g., personal tone, blank if none)                         | 

Besides the `text` column, character and word counts are also good indicators of text classification. To find the correlation among numerical features, correlation heatmap was generated. 

![Dashboard](https://github.com/ShaikhBorhanUddin/End-to-End-NLP-Pipeline-for-AI-Generated-Text-Detection/blob/main/Assets/heatmap.png?raw=true) 

Analyzing dataset and correlation heatmap, the following columns were excluded from model training to prevent data leakage, reduce bias, and ensure that the classifier learned intrinsic linguistic patterns rather than relying on metadata or weak auxiliary signals. The quality_score column was dropped because readability-based metrics capture stylistic uniformity but show high overlap between human and AI-generated text, making them weak and non-decisive predictors. Similarly, the sentiment column was excluded as emotional tone exhibits very weak correlation with AI authorship and does not reliably distinguish between human and AI writing. Metadata attributes such as topic were removed due to the risk of shortcut learning, as topic information can indirectly reveal labels without reflecting authorship style. The source_detail column was excluded because it explicitly identifies content origin (e.g., human author IDs or AI model names), which would cause severe data leakage and artificially inflate model performance. The timestamp column was dropped as temporal information is irrelevant to linguistic structure and may introduce chronological bias. The plagiarism column was excluded because plagiarism indicators are not causally related to AI text generation and are often noisy or inconsistently defined. Finally, the notes column was removed due to its subjective and human-annotated nature, which lacks consistency and does not represent intrinsic text characteristics. These columns were retained only for exploratory data analysis, stratified sampling, and bias analysis, while the final model was trained exclusively on text-based features. 

AI generated texts tend to use grammatically correct punctuations, avoiding excessive and erratic ones. On the other hand, human texts are inconsistent, somewhat biased, containing overused or missing punctuations. This makes punctuation usage a useful stylometric signal. Mathematically it is measured by: 

```bash
punctuation ratio = number of punctuation characters / number of total characters
```

Another parameter can be repetition score which measures how repetitive a text is, i.e., how often words or phrases are reused. AI text repeats patterns more consistently than human text. Human writing may contain similar expressions, particular phrases or expressions used over and over whereas AI generated texts use more moderated, neural texts and synonyms. The formula used for calculating it is: 

```bash
repetition score = 1 - (unique words / total words)
```

High repetition score indicates fewer unique words and more repetition whereas low repetition score indicates more lexical diversity. These two features are added to the dataset. The data dictionary for modified first dataset is as follows:

| **Column Name**       | **Data Type**    | **Description**                                              |
| --------------------- | ---------------- | ------------------------------------------------------------ |
| **text**              | String           | The actual written content or message                        |
| **label**             | String           | Indicates whether the text is generated by a *human* or *AI* |
| **length_chars**      | Integer          | Total number of characters in the text                       |
| **length_words**      | Integer          | Total number of words in the text                            |
| **punctuation_ratio** | Decimal          | Ratio of punctuation marks to total characters               |
| **repetition_score**  | Decimal          | Measure of repeated words/phrases in the text                | 

The second dataset, [Human vs AI Text Classification Dataset](https://www.kaggle.com/datasets/aknjit/human-vs-ai-text-classification-dataset/data) is relatively large (5000 entries; 2500 for each label). However, a significant portion of the dataset consists of repeated rows (4540 duplicates), leaving only 460 usable entries. Fortunately, class imbalance did not occur after duplicate removal. However, since the dataset has only `text` and `label` columns, features like `character length`, `word length`, `punctuation ratio` and `repetition score` are added to it. 

The third dataset, [AI vs Human Comparison Dataset](https://www.kaggle.com/datasets/prince7489/ai-vs-human-comparison-dataset/data) is the largest among the three datasets considered. To address the challenge of handling its size efficiently, a chunk of 20,000 entries (10,000 human and 10,000 AI-generated texts) is selected for further analysis. AI vs human text classification relies mostly on writing style, repetition, sentence structure, lexical and error patterns. These signals usually appear very early in the text. In practice, the first 200 – 400 tokens already contain enough signal for classification. The rest is redundant stylistically. So, 512 tokens are almost always enough for text classification. Therefore, sentence aware truncation (code given below) is applied to `text` column to make sure maximum text length is 512 tokens and there are no broken/incomplete sentence in the text entries. 

```bash
def truncate_text(text, max_tokens=512):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""
    sentences = nltk.sent_tokenize(text)
    truncated = []
    total_tokens = 0

    for sent in sentences:
        sent_tokens = len(tokenizer.encode(sent, add_special_tokens=False))
        if total_tokens + sent_tokens > max_tokens:
            break
        truncated.append(sent)
        total_tokens += sent_tokens
    return " ".join(truncated).strip()
```

Now that 3 datasets are cleaned and preprocessed, they are combined and checked for duplicates. 

```bash
df_combined = df_1_modified + df_2_modified + df_3_truncated
```

Upon duplicate removal, there are 20877 unique entries in the combined dataset. It can be used directly for distilbert model. Lemmatization is not strictly necessary in all text classification tasks, but it can be very beneficial, especially for distinguishing between human and AI-generated text using classica ML models. It reduces words to their base or dictionary form (lemma). For example, 'running', 'runs', and 'ran' all become 'run' after lemmatization. This reduces the total number of unique tokens in vocabulary, which can simplify model and prevent overfitting. By grouping different inflections of a word, lemmatization helps model treat them as the same concept. This can improve the quality of features extracted (e.g., for TF-IDF or word embeddings), as 'good' and 'better' are recognized as related to 'well'. It also allows the model to focus more on the core meaning of words rather than their grammatical variations. This can be crucial for style analysis, where the semantic content might be similar but the stylistic choices differ. Finally, by normalizing word forms, lemmatization can lead to better generalization and potentially higher accuracy for classification model, especially if the differences between human and AI text are subtle and relate to core vocabulary usage. 

Depending on training model selection, further data engineering is necessary on this lemmatized dataset, which will be discussed in feature engineering section. 

## Exploratory Data Analysis for Text 

Exploratory Data Analysis (EDA) was conducted to gain meaningful insights into the structural and linguistic differences between human-written and AI-generated text. This step was essential to understand underlying patterns in the dataset before feature engineering and model training. The analysis focused on text length distributions, word usage patterns, n-gram structures, and linguistic complexity measures to identify distinguishing characteristics between the two classes. 

First, the text length distribution was analysed by computing the number of words per sample. 

![Dashboard](https://github.com/ShaikhBorhanUddin/End-to-End-NLP-Pipeline-for-AI-Generated-Text-Detection/blob/main/Assets/word_count.png?raw=true) 

The word count distribution analysis (shown in above images) clear structural differences between human-generated and AI-generated text across both the original and lemmatized datasets. Most samples are concentrated within the 250–450 word range, indicating that the dataset primarily consists of moderately long texts rather than short responses. Human-written content demonstrates greater variability and a stronger presence in higher word-count bins, particularly around 400–450 words, suggesting more diverse and elaborative writing styles. In contrast, AI-generated text is more evenly distributed across mid-length ranges (250–400 words), reflecting the controlled and structured nature of generative models. Very short text entries (below 100 words) are relatively rare in both classes but appear slightly more frequently among AI-generated samples. The near-identical patterns observed after lemmatization confirm that preprocessing preserves the original structural characteristics of the data, ensuring that no class-specific bias is introduced during normalization. 

The analysis of most common words per class was conducted separately for human-generated and AI-generated text across both the original and lemmatized datasets. After removing stopwords, horizontal bar charts were generated for both datasets. 

![Dashboard](https://github.com/ShaikhBorhanUddin/End-to-End-NLP-Pipeline-for-AI-Generated-Text-Detection/blob/main/Assets/most_common_words.png?raw=true) 

In the df_combined dataset, several high-frequency words appear consistently in the top 15 list for both human-generated and AI-generated text, including student, people, would, could, also, time, school, and electoral, indicating shared thematic and stylistic patterns across both classes. Notably, the words would, people, and student occur more than 20,000 times in human-written entries, highlighting their strong presence in natural discourse and opinion-based writing. Similarly, the term student appears more than 18,000 times in AI-generated text, suggesting that educational topics are frequently addressed by generative models as well. Additionally, words such as people and also exceed 10,000 occurrences in both categories, reflecting their role as common connective and contextual terms. 

After lemmatization, the overall frequency distribution of the most common words remained largely unchanged, indicating that normalization did not significantly alter the core vocabulary patterns in the dataset. The word student continued to be the most frequent term across both human-generated and AI-generated categories, reaffirming the strong presence of education-related content in the corpus. Other high-frequency words also retained similar ranking positions, demonstrating that lemmatization primarily reduced word-form variations (such as plurals or tense changes) without affecting semantic prominence. 

![Dashboard](https://github.com/ShaikhBorhanUddin/End-to-End-NLP-Pipeline-for-AI-Generated-Text-Detection/blob/main/Assets/most_common_words_lemmatized.png?raw=true) 

These findings are further reinforced through the word cloud visualizations, where high-frequency words such as student, people, car, school, vote, time, make, and electoral appear in significantly larger font sizes. The prominence of these words visually highlights their dominance across both human and AI-generated text, making the lexical patterns more intuitive and easier to interpret. 

![Dashboard](https://github.com/ShaikhBorhanUddin/End-to-End-NLP-Pipeline-for-AI-Generated-Text-Detection/blob/main/Assets/wordcloud.png?raw=true) 

When comparing to unigram, higher N-gram analysis is significantly more computationally expensive because the number of possible word combinations increases exponentially as N grows. This results in substantially higher storage requirements, as large text corpora generate a vast number of unique trigrams that must be stored along with their frequency counts, increasing memory consumption and data management complexity. Furthermore, processing costs rise since trigram generation requires additional operations during text scanning, and downstream tasks such as probability estimation, smoothing, and sequence matching must be performed over a much larger feature space, leading to slower execution times. For these reasons, N-gram analysis in this project is restricted to bigrams in order to maintain a balance between computational efficiency and analytical effectiveness. 

![Dashboard](https://github.com/ShaikhBorhanUddin/End-to-End-NLP-Pipeline-for-AI-Generated-Text-Detection/blob/main/Assets/bigram.png?raw=true) 

The bigram analysis for the original df_combined dataset reveals distinct yet overlapping thematic patterns between human-generated and AI-generated text. In both categories, **electoral college** emerges as the most frequent bigram, indicating a strong political discourse presence in the dataset. Human-generated text shows higher diversity with phrases such as driverless cars, community service, many people, and students would, suggesting more personalized and context-rich expression. In contrast, AI-generated text demonstrates stronger emphasis on structured and topic-focused phrases like car usage, traffic congestion, public transportation, and air pollution, reflecting a more formal, informational tone. While several bigrams such as popular vote, driverless cars, and united states appear in both classes, their differing frequencies highlight stylistic variation in how humans and AI describe similar topics. 

![Dashboard](https://github.com/ShaikhBorhanUddin/End-to-End-NLP-Pipeline-for-AI-Generated-Text-Detection/blob/main/Assets/bigram_lemmatized.png?raw=true) 

After lemmatization, the overall thematic patterns remain consistent, but frequency counts become more consolidated due to word normalization. “Electoral college” continues to dominate both human and AI categories, reinforcing its prominence across the corpus. Human-generated text shows increased frequency for phrases such as driverless car, cell phone, extracurricular activity, and student would, indicating enhanced clarity after normalization. Meanwhile, AI-generated text maintains strong representation of transportation and environmental themes through bigrams like car usage, limit car, traffic congestion, air pollution, and greenhouse gas. The persistence of these structured, topic-driven phrases suggests AI text generation follows consistent narrative templates. Overall, lemmatization preserves semantic meaning while reducing linguistic noise, making stylistic differences between human and AI text even more apparent. 

For complexity measure analysis, Flesch Reading Ease scores and lexical diversity metrics were visualized to compare the linguistic characteristics of human-generated and AI-generated texts. The Flesch Reading Ease analysis indicates clear differences in readability between human-generated and AI-generated texts. Human-written content consistently demonstrated higher readability, with original texts averaging a score of 62.45, which corresponds to a standard and easily understandable reading level, while lemmatized human texts showed a slight improvement, averaging 66.80. In contrast, AI-generated texts were notably more difficult to read, with original texts scoring an average of 46.05 and lemmatized versions averaging 51.17, both reflecting a more complex and less accessible writing style. Although lemmatization marginally improved readability for both categories, a substantial readability gap between human and AI-generated content persisted, highlighting fundamental stylistic differences in sentence construction and vocabulary usage. 

![Dashboard](https://github.com/ShaikhBorhanUddin/End-to-End-NLP-Pipeline-for-AI-Generated-Text-Detection/blob/main/Assets/flesch_ttr.png?raw=true) 

Lexical diversity, measured using the Type-Token Ratio (TTR), revealed that AI-generated texts exhibited slightly higher vocabulary variation compared to human-generated content. In the original dataset, AI texts achieved an average TTR of 0.65, compared to 0.63 for human texts, and this pattern remained consistent after lemmatization, with AI texts averaging 0.61 versus 0.59 for human-written samples. As expected, lemmatization reduced TTR values for both categories by consolidating inflected word forms into their base forms. Despite this reduction, AI-generated content maintained marginally higher lexical diversity, suggesting a broader distribution of unique tokens, albeit often within more structured and repetitive syntactic patterns. 

## Feature Engineering 

Due to fundamental architectural differences between classical machine learning models and transformer-based models, two separate preprocessing and feature engineering pipelines were implemented. Classical machine learning models and the BiLSTM network relied on explicitly engineered features, including Bag-of-Words (BoW) and TF-IDF representations derived from cleaned and lemmatized text. In contrast, the DistilBERT model required minimal preprocessing and employed its native tokenizer to generate contextualized subword embeddings. This allowed the model to automatically learn linguistic features through self-attention mechanisms. This separation ensured methodological correctness and enabled a fair comparison between model architectures. 

All feature engineering procedures were conducted exclusively on the df_combined_lemmatized dataset to ensure consistent and normalized text input. The dataset was first loaded, and essential NLP libraries were imported. English stopwords were downloaded and initialized using NLTK. The preprocessing pipeline ensured that all text entries were properly formatted, tokenized, and free from missing values prior to feature extraction. 

### Stopword Removal 

Each text entry was tokenized and filtered to remove stopwords, producing a new column named cleaned_text. This step eliminated high-frequency but semantically weak words such as and, the, and is, allowing the models to focus on more meaningful linguistic content. Stopword removal improved signal-to-noise ratio and enhanced feature quality for downstream modeling. 

### BoW and CountVectorizer 

Two classical text representation techniques were applied to the cleaned text. First, Bag-of-Words (BoW) features were generated using CountVectorizer, transforming the text into a sparse matrix of word frequencies. This resulted in a feature matrix of shape (20,877, 38,038), where each row represents a document and each column corresponds to a unique word in the vocabulary. 

Similarly, TF-IDF features were extracted using TfidfVectorizer, producing a matrix of identical shape. Unlike BoW, TF-IDF assigns higher weights to words that are frequent within a document but rare across the corpus. This weighting scheme improves the model’s ability to capture discriminative terms, making TF-IDF particularly effective for text classification tasks. These two representations provided strong baseline lexical features for classical machine learning models. 

### word2vec 

In addition to frequency-based methods, a Word2Vec model was trained on the cleaned and lemmatized text to capture semantic relationships between words. The model learned embeddings for 12,664 unique words, with each word represented by a 100-dimensional dense vector. These embeddings encode contextual similarity and semantic structure, allowing related words to have similar vector representations. 

The learned word embeddings can be aggregated (e.g., using mean pooling) to form document-level representations, which were used as input for deep learning models. Finally, all trained vectorizers and the Word2Vec model were saved to ensure reproducibility. The feature-engineered dataset was also exported as a CSV file to facilitate seamless integration into downstream modeling pipelines. 

This multi-representation strategy enabled the models to learn from both surface-level lexical patterns and deeper semantic structures, ultimately strengthening classification performance. 

## Models 

Classical machine learning models, BiLSTM, and DistilBERT were trained using separate pipelines to ensure methodological independence. 

Logistic Regression (LR), Support Vector Machine (SVM), Random Forest (RF), and XGBoost classifiers were trained and evaluated in parallel using identical feature sets to enable fair comparison across models. A classical machine learning text classification pipeline was implemented that integrates linguistic feature engineering with multiple text representation techniques. Text data were transformed using Bag-of-Words (BoW), TF-IDF, and Word2Vec-based document embeddings. These representations were further enriched by concatenating handcrafted numerical features, including text length, punctuation ratio, and word repetition scores. The resulting feature matrices were split into training and testing sets. Model performance was primarily evaluated using Logistic Regression classifiers, with solver configurations adapted to handle sparse (BoW, TF-IDF) and dense (Word2Vec) feature spaces. This experimental setup enabled a systematic comparison of how different feature representations influence classification performance on the binary text classification task (e.g., human vs. AI-generated text). 

| Model                    | Algorithm                   | Feature Variant      | max_iter | solver | n_jobs | eval_metric | random_state |
|--------------------------|-----------------------------|----------------------|----------|--------|--------|-------------|--------------|
| LR (BoW)                 | Logistic Regression         | Bag-of-Words         | 1000     | lbfgs  | -1     | —           | 42           |
| LR (TF-IDF)              | Logistic Regression         | TF-IDF               | 1000     | lbfgs  | -1     | —           | 42           |
| LR (Word2Vec)            | Logistic Regression         | Word2Vec (avg)       | 1000     | lbfgs  | -1     | —           | 42           |
| XGBoost (BoW)            | XGBoost Classifier          | Bag-of-Words         | —        | —      | -1     | logloss     | 42           |
| XGBoost (TF-IDF)         | XGBoost Classifier          | TF-IDF               | —        | —      | -1     | logloss     | 42           |
| XGBoost (Word2Vec)       | XGBoost Classifier          | Word2Vec (avg)       | —        | —      | -1     | logloss     | 42           |
| Random Forest (BoW)      | Random Forest Classifier    | Bag-of-Words         | —        | —      | -1     | —           | 42           |
| Random Forest (TF-IDF)   | Random Forest Classifier    | TF-IDF               | —        | —      | -1     | —           | 42           |
| Random Forest (Word2Vec) | Random Forest Classifier    | Word2Vec (avg)       | —        | —      | -1     | —           | 42           |
| SVM (BoW)                | Linear SVM                  | Bag-of-Words         | 1000     | —      | -1     | —           | 42           |
| SVM (TF-IDF)             | Linear SVM                  | TF-IDF               | 1000     | —      | -1     | —           | 42           |
| SVM (Word2Vec)           | Linear SVM                  | Word2Vec (avg)       | 1000     | —      | -1     | —           | 42           | 

The second model training pipeline implements a deep learning sequence model based on a Bidirectional LSTM (BiLSTM) architecture built with TensorFlow/Keras. This approach is designed to capture sequential and contextual dependencies in text that traditional machine learning models cannot explicitly model. Text inputs are vectorized using pretrained representations, most notably Word2Vec embeddings, optionally concatenated with engineered numerical features and reshaped into a 3-D tensor to satisfy recurrent network input requirements. In the Word2Vec + numerical-feature setting, the resulting input shape is (104, 1), where each feature dimension is treated as a timestep. The model itself is defined as a Sequential Keras model consisting of two stacked Bidirectional LSTM layers: the first with 64 units and return_sequences=True, followed by a Dropout layer (rate = 0.3) for regularization, and a second BiLSTM layer with 32 units, again followed by Dropout (0.3). The recurrent layers are followed by a Dense layer with 16 units and ReLU activation to learn higher-level abstractions, and a final Dense output layer with a sigmoid activation for binary classification. The model is compiled using the Adam optimizer, binary cross-entropy loss, and accuracy as the evaluation metric, and is trained using mini-batch gradient descent. Overall, this notebook investigates whether bidirectional temporal modeling of feature sequences via recurrent neural networks can outperform classical machine-learning approaches on the same classification task. 

![Dashboard](https://github.com/ShaikhBorhanUddin/End-to-End-NLP-Pipeline-for-AI-Generated-Text-Detection/blob/main/Assets/BiLSTM.png?raw=true) 

A transformer-based training pipeline was implemented using the Hugging Face Transformers library and a pretrained DistilBERT model for binary text classification. The model imports `DistilBertForSequenceClassification` initialized from the `distilbert-base-uncased` checkpoint and configured with `num_labels = 2`. Its architecture consists of a DistilBERT encoder that generates contextualized token embeddings, followed by a linear pre-classifier layer and a final classification head that outputs logits for the two target classes. Text inputs are tokenized using the native DistilBERT tokenizer and passed through the model for end-to-end supervised fine-tuning. During training and inference, both the model and input tensors are automatically transferred to the CUDA device when available to accelerate computation. This approach uses self-attention mechanisms to learn deep semantic and contextual representations that are not accessible to classical or recurrent models. 

## Performance Matrix Evaluation 

Performance matrices of 4 classical ML models are included here. 

| Model               | Feature Type          | Accuracy | Precision | Recall | F1-score |
|--------------------|----------------------|----------|-----------|--------|----------|
| Logistic Regression | BoW + Numerical      | 0.96     | 0.96      | 0.96   | 0.96     |
| Logistic Regression | TF-IDF + Numerical   | 0.98     | 0.98      | 0.98   | 0.98     |
| Logistic Regression | Word2Vec             | 0.95     | 0.95      | 0.95   | 0.95     |
| SVM                 | BoW + Numerical      | 0.96     | 0.96      | 0.96   | 0.96     |
| SVM                 | TF-IDF + Numerical   | 0.97     | 0.97      | 0.97   | 0.97     |
| SVM                 | Word2Vec             | 0.95     | 0.95      | 0.95   | 0.95     |
| Random Forest       | BoW + Numerical      | 0.97     | 0.97      | 0.97   | 0.97     |
| Random Forest       | TF-IDF + Numerical   | 0.96     | 0.96      | 0.96   | 0.96     |
| Random Forest       | Word2Vec             | 0.94     | 0.94      | 0.94   | 0.94     |
| XGBoost             | BoW + Numerical      | 0.96     | 0.96      | 0.96   | 0.96     |
| XGBoost             | TF-IDF + Numerical   | 0.97     | 0.97      | 0.97   | 0.97     |
| XGBoost             | Word2Vec             | 0.95     | 0.95      | 0.95   | 0.95     | 

The analysis shows that TF-IDF feature representation consistently outperforms both Bag-of-Words (BoW) and Word2Vec across most models, with Logistic Regression using TF-IDF achieving the highest accuracy of 98%. Random Forest with BoW also performed strongly, reaching 97% accuracy, while Word2Vec features generally yielded lower performance across all models. Simpler models, such as Logistic Regression and Linear SVM, performed on par with more complex models, and evaluation metrics including Accuracy, Precision, Recall, and F1-score were highly aligned, indicating balanced classification. Tree-based models tend to perform better with BoW features, and XGBoost demonstrated stable performance across all feature types. Overall, all models achieved above 94% accuracy, reflecting strong dataset separability, and TF-IDF emerges as the most effective feature extraction method for this task. 

### BiLSTM With word2vec Features 

The very large feature set (`X_train_bow` shape: (16701, 38042)) is the primary reason for the BiLSTM model taking a very long time to train. Each input to the LSTM layers contains 38,042 features, which leads to a huge number of trainable parameters in the Bidirectional LSTM layers and significantly increases computational complexity and training time. For high-dimensional and sparse representations such as Bag-of-Words and TF-IDF, traditional machine learning models like Logistic Regression or SVMs are generally more appropriate, or dimensionality reduction techniques such as PCA are applied before using deep learning models. However, performing PCA on more than 38,000 features is itself computationally expensive and time-consuming. Training an LSTM model with BoW and numerical features was estimated to take approximately 5 hours on an A100 GPU and around 12 hours on an L4 GPU. Due to these limitations, only the Word2Vec-based model was trained. 

BiLSTM model training was conducted for 100 epochs, during which training accuracy steadily increased from roughly 0.70 to about 0.99, while validation accuracy rose rapidly early on and stabilized around 0.95–0.97 with minor fluctuations. Training loss consistently decreased throughout all 100 epochs, indicating continued optimization on the training set. In contrast, validation loss dropped sharply in the initial epochs but began to fluctuate and slightly increase in later epochs, suggesting the onset of mild overfitting toward the end of the 100-epoch training period. 

![Dashboard](https://github.com/ShaikhBorhanUddin/End-to-End-NLP-Pipeline-for-AI-Generated-Text-Detection/blob/main/Assets/BiLSTM_acc_loss.png?raw=true) 

### Distilbert base 

The performance matrices of distilbert model are given below. 

![Dashboard](https://github.com/ShaikhBorhanUddin/End-to-End-NLP-Pipeline-for-AI-Generated-Text-Detection/blob/main/Assets/distilbert_acc_pre.png?raw=true)   

Here, validation accuracy remains consistently high across epochs, starting around 0.97, quickly reaching approximately 0.99, and stabilizing with only minor fluctuations. Validation precision shows a brief dip in early epochs but rapidly improves to around 0.99 and remains stable thereafter, indicating very few false positives as training progresses. 

![Dashboard](https://github.com/ShaikhBorhanUddin/End-to-End-NLP-Pipeline-for-AI-Generated-Text-Detection/blob/main/Assets/distilbert_rec_f1.png?raw=true) 

Validation recall stays extremely high throughout training, close to 0.99–1.00, with only a small temporary drop around the middle epochs before recovering. The validation F1-score closely follows this trend, remaining around 0.98–0.99 and stabilizing toward the later epochs, reflecting a strong and well-balanced performance between precision and recall. 

### ROC - AUC Analysis 

All models showed nearly perfect and identical ROC Curves, and therefore not discussed with specific details. 

![Dashboard](https://github.com/ShaikhBorhanUddin/End-to-End-NLP-Pipeline-for-AI-Generated-Text-Detection/blob/main/Assets/roc_ML.png?raw=true) 

### Confusion Matrix 

Confusion Matrices of all tested models are included next. 

![Dashboard](https://github.com/ShaikhBorhanUddin/End-to-End-NLP-Pipeline-for-AI-Generated-Text-Detection/blob/main/Assets/cm_ML.png?raw=true) 

Across all ML models confusion matrices, the models demonstrate consistently strong classification performance with high true positive and true negative counts and relatively low misclassification rates. Logistic Regression performs best with Bag-of-Words and TF-IDF features, showing balanced errors and only a slight increase in false positives and false negatives when using Word2Vec. Linear SVM exhibits very strong results overall, particularly with TF-IDF, where both false positives and false negatives are among the lowest across all models, while Word2Vec again introduces more misclassifications. Random Forest achieves high true negative counts but tends to show comparatively higher false negatives, indicating a slightly weaker ability to capture all positive instances, especially with Word2Vec. XGBoost delivers consistently robust performance across all feature sets, with Bag-of-Words and Word2Vec yielding very low error counts and TF-IDF showing only a minor increase in misclassifications. Overall, Bag-of-Words and TF-IDF representations outperform Word2Vec across most models, and Linear SVM with TF-IDF and XGBoost variants emerge as the most reliable configurations based on confusion matrix outcomes. 

![Dashboard](https://github.com/ShaikhBorhanUddin/End-to-End-NLP-Pipeline-for-AI-Generated-Text-Detection/blob/main/Assets/cm_non_ML.png?raw=true) 

The BiLSTM with Word2Vec model correctly classified most samples, with 1,956 true negatives and 1,950 true positives. However, it produced 129 false positives and 141 false negatives, leading to an overall accuracy of about 93.5%. This indicates solid performance, but with a noticeable number of misclassifications in both classes. 

In contrast, the DistilBERT model performed significantly better. It achieved 2,114 true negatives and 2,038 true positives, while producing only 17 false positives and 7 false negatives. This resulted in a much higher accuracy of approximately 99.4%, showing that DistilBERT is far more effective at minimizing classification errors and providing more reliable predictions overall. 

## Explainability 

Based on its superior overall performance and suitability for trouble-free deployment, the DistilBERT model was selected for decision visualization. Both LIME and SHAP were evaluated to visualize and interpret the model’s predictions. Some samples generated with LIME are included below. Due to high use of computational resources, number of features were restricted to 10 – 15. 

![Dashboard](https://github.com/ShaikhBorhanUddin/End-to-End-NLP-Pipeline-for-AI-Generated-Text-Detection/blob/main/Assets/LIME_exp.png?raw=true)  

Samples of SHAP visualizations are included next. 

![Dashboard](https://github.com/ShaikhBorhanUddin/End-to-End-NLP-Pipeline-for-AI-Generated-Text-Detection/blob/main/Assets/SHAP_exp_generic.png?raw=true) 

The generic interactive visualization of SHAP is sometimes difficult for end users to comprehend, especially when the text length is high. So, an alternative visualization (looking similar to LIME) is also developed. This alternative visualization along with generic LIME output is used in deployment.

![Dashboard](https://github.com/ShaikhBorhanUddin/End-to-End-NLP-Pipeline-for-AI-Generated-Text-Detection/blob/main/Assets/SHAP_exp_mod.png?raw=true) 

When comparing the two frameworks, SHAP demonstrates superior explainability by precisely highlighting the text segments contributing to the classification outcome. 

## Deployment 

To access the app click the [link](https://huggingface.co/spaces/ShaikhBorhanUddin/AI_generated_text_detection). 

Due to the large model size (over 300 MB), the model files were uploaded to the Hugging Face platform for deployment. The user interface was developed using Gradio. For each user input, the application generates both LIME and SHAP visualizations side by side along with the classification output, with important text segments highlighted. In addition, the most influential words driving the classification are displayed in the form of  bar charts. 

## Practical Applications 

To demonstrate the real-world effectiveness of the proposed AI-generated text detection system, practical experiments were conducted using authentic human-written content and their AI-modified counterparts. These examples were tested directly through the deployed Gradio application, allowing for an interactive and transparent evaluation of model predictions. 

In the first scenario, an original news article published by BBC was used as a sample of human-authored journalistic content. This article was then modified using a generative AI model to create an AI-assisted version while preserving the original topic and structure. Both versions were separately tested in the application. The system successfully differentiated between the human-written news article and its AI-generated counterpart, assigning higher confidence scores to the AI-modified version. This experiment highlights the model’s capability to support journalism and news verification workflows by flagging synthetically altered content for further editorial review. 

<p align="center">
  <img src="https://github.com/ShaikhBorhanUddin/End-to-End-NLP-Pipeline-for-AI-Generated-Text-Detection/blob/main/Assets/news_human.png" width="49.5%" />
  <img src="https://github.com/ShaikhBorhanUddin/End-to-End-NLP-Pipeline-for-AI-Generated-Text-Detection/blob/main/Assets/news_ai.png" width="49.5%" />
</p> 

In the second scenario, a formal application letter written by a student to a school principal was used as a human-authored sample. An AI-enhanced version of the same letter was generated using a large language model to simulate assisted writing. When both texts were tested in the application, the model demonstrated clear discrimination between human-written and AI-assisted content, reinforcing its relevance for academic integrity and education technology use cases. This showcases how the system can help educators identify potentially AI-assisted submissions without enforcing automated penalties. 

<p align="center">
  <img src="https://github.com/ShaikhBorhanUddin/End-to-End-NLP-Pipeline-for-AI-Generated-Text-Detection/blob/main/Assets/edu_human.png" width="49%" />
  <img src="https://github.com/ShaikhBorhanUddin/End-to-End-NLP-Pipeline-for-AI-Generated-Text-Detection/blob/main/Assets/edu_ai.png" width="49.95%" />
</p> 

These real-world demonstrations validate the practical applicability of the system across different domains. By testing both authentic and AI-modified texts in controlled scenarios, the application proves its ability to function as a reliable decision-support tool, providing confidence scores and explainability outputs to assist human reviewers in making informed judgments.

## Limitations 

While the proposed system demonstrates promising results, certain limitations were observed that highlight areas for further improvement and optimization. 

### Dataset domain bias 

Although the dataset contains a large number of samples, there is a noticeable concentration of politically themed content, particularly related to U.S. politics. Frequent occurrences of terms such as electoral college, popular vote, electoral vote, and vice president indicate topic imbalance. Since real-world users may input text from diverse domains (education, business, healthcare, social media, etc.), this topical skew may limit the model’s generalization ability to unseen contexts. 

### High memory consumption 

The data cleaning, feature engineering, and model training stages require substantial computational resources. During experimentation, RAM usage reached up to 48 GB in the Colab Pro environment, especially when working with TF-IDF matrices and deep learning models. This resource requirement may restrict reproducibility for users without access to high-memory cloud instances or high-end local machines. 

### Inference latency for long texts 

When processing lengthy text inputs, the deployed Gradio application takes approximately 30–40 seconds to generate predictions and highlight influential words. This delay can negatively impact user experience, particularly in real-time applications where quick feedback is expected. Optimization techniques such as text chunking, model distillation, or backend acceleration could help mitigate this issue in future improvements. 

## Tools & Technologies Used 

This project relies on a combination of classical machine learning, deep learning, and transformer-based techniques, along with essential data processing and visualization libraries, to build an effective AI-generated text detection system. 

- Programming Language: Python 3.10+ 
- Machine Learning Frameworks: Logistic Regression, Support Vector Machine (SVM), Random Forest, XGBoost 
- Deep Learning Frameworks: TensorFlow, Keras (BiLSTM) 
- Transformer Model: DistilBERT (Hugging Face Transformers) 
- Data Processing: Pandas, NumPy 
- NLP Tools: NLTK, spaCy (tokenization, lemmatization, stopword removal) 
- Feature Extraction: TF-IDF Vectorizer, CountVectorizer, Word2Vec 
- Model Evaluation: Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix 
- Visualization: Matplotlib, Seaborn 
- Explainability: LIME, SHAP 
- Deployment: Gradio Web Application 
- Development Environment: Jupyter Notebook, VS Code, Google Colab (GPU-enabled) 

## Licence 

This project is licensed under the MIT License, a permissive open-source license that allows reuse, modification, and distribution with attribution. Users are free to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the project, provided that the original copyright and license notice are included in all copies or substantial portions of the software. 

For more details, refer to the Licence file in this repository. 

## Contact 

If you have any questions or would like to connect, feel free to reach out!

**Shaikh Borhan Uddin**  
📧 [`Email`](mailto:shaikhborhanuddin@gmail.com)  
🔗 [`LinkedIn`](https://www.linkedin.com/in/shaikh-borhan-uddin-905566253/)  
🌐 [`Portfolio`](https://github.com/ShaikhBorhanUddin)

Feel free to fork the repository, improve the queries, or add visualizations! 
