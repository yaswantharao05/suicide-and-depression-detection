# 🧠 Depression and Suicide Risk Detection using Machine Learning

This project leverages advanced machine learning and natural language processing techniques to enable **early detection of depression and suicidal ideation**, primarily through analysis of textual data from digital sources.

## 🚀 Motivation

Mental health disorders, particularly depression and suicidal tendencies, are escalating global issues. Traditional diagnosis methods—relying on self-reporting and clinical evaluation—often fail to identify early warning signs. This project was motivated by the need for **automated, real-time systems** that can flag mental health concerns at an early stage, offering a chance for timely intervention and support.

## 🛠️ Technologies and Approach

- **NLP & Deep Learning Models**: Bi-LSTM, CNN, SVM, and Transformer-based models like BERT were used to classify mental health indicators in text.
- **Dataset**: Social media posts and forums provided linguistic data to train classifiers on patterns associated with depression and suicidal ideation.
- **Hybrid Model**: Combines CNN and RNN architectures to enhance classification accuracy by capturing both local and sequential features in text.

## ✅ Accuracy & Results

- **BERT (Binary Classification)**: Achieved **96.4% accuracy** in detecting suicidal messages.
- **BERT (Multi-Class Classification)**: Achieved **91.2% accuracy** for classifying between healthy, depressive, and suicidal messages.
- **Support Message Generation**: The system also suggests empathetic, supportive responses based on detected emotional tone.

## 🖥️ Frontend / User Interface

- A **Chrome Extension** monitors web search queries for potentially harmful terms, sending data for backend analysis.
- A **Parent-friendly Dashboard** allows real-time visualization of flagged content, model predictions, and AI-generated suggestions for intervention.
- The UI promotes accessibility and usability, especially for non-technical caregivers or mental health professionals.

## 🌟 Advantages

- **Early Intervention**: Proactively detects signs of mental distress from everyday digital interactions.
- **Non-intrusive Monitoring**: Uses text analysis to assess risk without violating privacy through invasive means.
- **Extensible Architecture**: Easily adaptable for various mental health conditions beyond depression and suicide.
- **Human-in-the-loop Design**: Allows professionals or parents to review predictions before taking action.

## 📁 Repository Structure

