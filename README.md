# End to End NLP Pipeline for AI-Generated Text Detection 

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

## Project Workflow 

## Dataset, Cleaning & Pre-processing 

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
