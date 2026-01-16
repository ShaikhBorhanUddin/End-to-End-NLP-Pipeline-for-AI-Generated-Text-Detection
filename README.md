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
│      └── Feature Engineered/    # dataset with BoW, TF-IDF and word2vec features
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

![Dashboard](https://github.com/ShaikhBorhanUddin/End-to-End-NLP-Pipeline-for-AI-Generated-Text-Detection/blob/main/Assets/heatmap_df_1.png?raw=true) 

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

The third dataset, [AI vs Human Comparison Dataset](https://www.kaggle.com/datasets/prince7489/ai-vs-human-comparison-dataset/data) is the largest among the three datasets considered. To address the challenge of handling its size efficiently, a chunk of 20,000 entries (10,000 human and 10,000 AI-generated texts) is selected for further analysis. AI vs human text classification relies mostly on writing style, repetition, sentence structure, lexical and error patterns. These signals usually appear very early in the text. In practice, the first 200 – 400 tokens already contain enough signal for classification. The rest is redundant stylistically. So, 512 tokens are almost always enough for text classification. Therefore, sentence aware truncation is applied to ‘text’ column to make sure maximum text length is 512 tokens and there are no broken/incomplete sentence in the text entries. 


## Exploratory Data Analysis (EDA) for Text 

## Feature Engineering 

Due to fundamental architectural differences between classical machine learning models and transformer-based models, two separate preprocessing and feature engineering pipelines were implemented. Classical machine learning models and the BiLSTM network relied on explicitly engineered features, including Bag-of-Words and TF-IDF representations derived from cleaned and lemmatized text. In contrast, the DistilBERT model utilized minimal preprocessing and employed its native tokenizer to generate contextualized subword embeddings, allowing the model to learn linguistic features automatically through self-attention mechanisms. This separation ensures methodological correctness and fair model comparison. 

## Models 

## Tests 

## Evaluation 

## Explainability 

## Applications 

## Deployment 

To access the app click the [link](https://huggingface.co/spaces/ShaikhBorhanUddin/AI_generated_text_detection). 

## Limitations 

## Technologies Used 

## Future Developement 

## Licence 

## Contact  
