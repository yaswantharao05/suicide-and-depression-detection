import random, re, torch
import pandas as pd
import torch.nn.functional as F
from collections import Counter

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification

from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, HTTPException

from huggingface_hub import InferenceClient
from langchain_ollama import ChatOllama
import time



Mistral_llm = ChatOllama(model="mistral:latest")

def llm_predict(prompt: str) -> str:
    return Mistral_llm.invoke(prompt).content


app = FastAPI()

# Improved CORS configuration (more secure than ["*"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # Replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # Specify only needed methods
    allow_headers=["*"],  # Specify only needed headers
)

# Load the fine-tuned BERT model
MODEL_SAVE_PATH = r"\suicide-detection-backend\saved_model-bert_binary"
tokenizer = tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_SAVE_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Loaded binary classification model")

MULTI_CLASS_MODEL_SAVE_PATH = "\suicide-detection-backend\bert_mental_disorder_model_final_multiClass"
MULTI_CLASS_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
MULTI_CLASS_model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=5,  # Number of classes
    # hidden_dropout_prob=0.2,   
    # attention_probs_dropout_prob=0.2,
)
MULTI_CLASS_model.load_state_dict(torch.load(r"\suicide-detection-backend\bert_mental_disorder_model_final_multiClass\best_model.pth", map_location=torch.device('cpu')))
MULTI_CLASS_model.to(device)
print("Loaded multi classification model")

client = InferenceClient(
    provider="hf-inference",
    api_key=HUGGINGFACE_API_KEY,  # Replace with your Hugging Face API key
)

def predict_text(text):
    """Predict if the given text is suicide-related or not."""
    inputs = tokenizer(text, padding="max_length", truncation=True, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = F.softmax(logits, dim=-1)
    predicted_class = torch.argmax(logits, dim=-1).item()

    print("1->",predicted_class)

    return "suicide" if predicted_class == 1 else "non-suicide", probs.tolist()


def predict_text_multi_class(text):
    """Predict multi class text classification"""
    inputs = MULTI_CLASS_tokenizer(text, padding="max_length", truncation=True, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = MULTI_CLASS_model(**inputs)

    logits = outputs.logits
    probs = F.softmax(logits, dim=-1)
    predicted_class = torch.argmax(logits, dim=-1).item()

    print("2->",predicted_class)

    if predicted_class==1 or predicted_class == 4:
        return random.choice([0, 2, 3])
    return predicted_class


def extract_frequent_words(texts):
    """Extract most frequent words from flagged texts."""
    bag_of_words = set([
        "suicide", "depression", "end my life", "kill myself", "hopeless", "worthless", "no way out",
        "overdose", "self-harm", "cutting", "life is meaningless", "I hate my life", "suicidal thoughts",
        "can't go on", "dark thoughts", "nobody cares", "drowning in pain", "I want to die",
        "mental anguish", "give up", "why live", "cry for help", "lost all hope", "end it all",
        "jump off a bridge", "hang myself", "sleep forever", "death is the answer", "I feel empty",
        "can't take it anymore", "alone in the world", "I'm a burden", "no one understands",
        "severe depression", "deep sadness", "I can't handle this", "it hurts so much", "draining",
        "isolated", "unloved", "better off dead", "I don't want to wake up", "goodbye world",
        "stop the pain", "it's too much", "disappear forever", "emotional pain", "mental breakdown",
        "feeling numb", "severe anxiety", "panic attacks", "stress overload", "trauma", "despair",
        "grief", "mourning", "unbearable", "I can't keep pretending", "nothing matters",
        "life is too hard", "giving up", "please help", "ending it all", "life isn't worth it",
        "no reason to live", "feeling trapped", "everything is falling apart", "it's pointless",
        "can't escape this", "inner torment", "I don't belong here", "escape from this pain",
        "I feel invisible", "no one listens", "can't breathe", "constant crying", "shattered inside",
        "breaking down", "lost in my mind", "suffocating", "feeling hopeless", "all alone",
        "life is overwhelming", "I feel broken", "tired of pretending", "hiding my pain", "dying inside",
        "in too deep", "giving in", "self-loathing", "can't find peace", "I don't fit in",
        "reaching out for help", "my mind won't stop", "negative thoughts", "consumed by sadness",
        "wishing for an end", "I can't hold on", "falling apart", "emotional scars", "please end this",
        "silent suffering", "I feel dead inside", "no joy", "constant sadness", "feeling defeated",
        "life is unbearable", "I just want to sleep", "I can't move on", "please notice me",
        "feeling unworthy", "I feel nothing", "I can't fight anymore", "my heart hurts", "aching inside",
        "overwhelmed with sadness", "numb to the world", "struggling to breathe", "nobody understands",
        "lost in the darkness", "I feel so empty", "nobody notices me", "wishing for peace",
        "I just want to be free", "crying myself to sleep", "broken beyond repair", "my life is a mess",
        "I hate myself", "I'm so tired", "nobody loves me", "what's the point", "it's all too much",
        "I'm drowning", "I'm so scared", "I'm all alone", "I need help", "it's not worth it",
        "feeling hollow", "my mind is racing", "I can't go on like this", "I'm losing it",
        "everything is crumbling", "I feel so lost", "please save me", "I don't see a future",
        "the pain won't stop", "my heart feels heavy", "I'm so ashamed", "I can't let go",
        "my world is falling apart", "I'm spiraling", "I'm so broken", "I can't feel anything",
        "everything is meaningless", "I'm a failure", "I just want to escape", "I can't make it",
        "I'm falling", "I can't keep up", "I'm so worthless", "I'm just so sad", "I can't find hope"
    ])

    word_counts = Counter()
    for text in texts:
        words = re.findall(r'\b\w+\b', text.lower())  # Tokenize words
        for word in words:
            if word in bag_of_words:
                word_counts[word] += 1

    return word_counts.most_common(15)  # Return top 15 frequent words


def get_systemMessage(pred):
    """Get system message based on multi class predictions"""
    predicted_class_index = max(pred, key=lambda k: len(pred[k]))
    print("predicted_class_index",predicted_class_index)
    label = {0: "Anxiety",
             2: "Suicide-Watch",
             3: "depression",
             4: "Off my chest"}
    
    predicted_class = label[predicted_class_index]
    print("predicted_class",predicted_class)

    chat_history = sorted(pred[predicted_class_index], key=lambda x: len(x))[:4]
    
    support_message = generate_support_message(chat_history, predicted_class)

    return predicted_class, support_message #"You've been expressing feelings of deep sadness, hopelessness, or fatigue in your messages." 


def generate_support_message(concerning_phrases, predicted_class):        
    # Condition-specific guidance
    condition_responses = {
        "depression": {
            "description": "You've been expressing feelings of deep sadness, hopelessness, or fatigue in your messages.",
            "examples": concerning_phrases,
            "support": "It sounds like you're carrying a heavy weight. You're not alone—many people feel this way, and help exists.",
            "resources": "Consider reaching out to a trusted friend or a counselor. Resources like the National Suicide Prevention Lifeline (988) are available 24/7."
        },
        "Suicide-Watch": {
            "description": "Your messages include thoughts about ending your life, which is extremely concerning.",
            "examples": concerning_phrases,
            "support": "Your life matters deeply. These thoughts are a sign of pain, not a solution.",
            "resources": "Please contact a crisis line immediately (e.g., 988 in the US or a local service). You don't have to face this alone."
        },
        "Anxiety": {
            "description": "Your chats suggest overwhelming worry or fear, which can feel paralyzing.",
            "examples": concerning_phrases,
            "support": "Anxiety can make the world feel unsafe. Small steps to ground yourself (like deep breathing) can help.",
            "resources": "Apps like Calm or Headspace offer guided exercises. A therapist can also provide coping strategies."
        },
        "Off my chest": {
            "description": "You've been sharing heavy emotions, which takes courage.",
            "examples": concerning_phrases,
            "support": "Getting things off your chest is a brave first step. Holding space for your feelings is important.",
            "resources": "Talking to someone who listens without judgment (like a support group) can be healing."
        }
    }
    # Generate the LLM prompt
    prompt = f"""
            You are a compassionate and knowledgeable mental health support assistant. Your task is to help a guardian understand and support a user who may be struggling with **{predicted_class}**, based on their recent chat history.

            The user's chat history contains potentially concerning phrases such as:  
            {chr(10).join(f'- "{phrase}"' for phrase in concerning_phrases)}

            Please provide a **thorough, thoughtful, and supportive analysis**, following these guidelines:

            1. **Address the Guardian**: Speak directly to the guardian, offering observations and actionable guidance.
            - Use phrases like: *"Based on the user's recent messages..."* or *"It's important to recognize that..."*

            2. **Validate the User's Emotional State**: 
            - Gently acknowledge the seriousness of the user's emotions without causing alarm.
            - Provide an emotional interpretation of 2-3 of the most critical phrases from the chat.
            - Example: *"The phrase '[concerning_phrase]' suggests the user may be feeling [emotion/struggle], which is a sign they might be overwhelmed or emotionally exhausted."*

            3. **Contextual Summary**:
            - Summarize the overall emotional tone of the chat history (e.g., recurring themes like hopelessness, fear, isolation).
            - Highlight how different phrases might reflect patterns or intensifying emotions.

            4. **Use Extended Message Fragments**:
            - Create or reconstruct longer, plausible messages or excerpts based on the provided concerning phrases (combine related phrases if needed), to give the guardian a clearer picture of how the user may be expressing themselves emotionally.

            5. **Recommend Immediate and Supportive Actions**:
            - Suggest practical, empathetic approaches the guardian can take right now.
            - Example: *"Approach the user in a private and safe space. Use open-ended questions like, 'I noticed you've been feeling low lately—do you want to talk about it?'”*

            6. **Provide Tailored Resources and Follow-up Suggestions**:
            - Share the following confidential support resource(s):  
                {condition_responses[predicted_class]['resources']}
            - Emphasize confidentiality and encourage proactive support, such as seeking help from a trusted professional or counselor.

            7. **Maintain an Appropriate Tone**:
            - If the situation is high-risk (e.g., suicidewatch), respond with urgency and deep compassion.
            - For moderate-risk categories (e.g., anxiety, depression), be vigilant and reassuring, promoting open communication.

            8. **Include a Crisis Helpline for India**:
            - India Suicide Prevention Lifeline:  
                Call: 9152987821  
                Website: https://icallhelpline.org/  
                FAQs: https://icallhelpline.org/faqs/

            **Important:**  
            - Do NOT dismiss or minimize the user's experiences.  
            - Avoid vague or overly generic responses like "Just be there for them."  
            - Focus on empathy, context, and guidance tailored to the emotional weight of the messages.

            Your response should be **clear, emotionally aware, and at least 3–4 paragraphs long**, combining analytical insights with practical advice.
            """
    
    messages_for_llm = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": "Draft a response."}
    ]

    start_time = time.time()

    response = llm_predict(prompt)

    end_time = time.time()
    # Print time taken
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    message = response.split('\n')
    return message


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    # try:
    df = pd.read_csv(BytesIO(await file.read()))
    # print("df",df)
    results = []
    flagged_texts = []
    multi_class_predictions = {0:[], 2:[], 3:[], 4:[]}
    count = 0
    for text in df["text"][:250]:  # Ensure CSV has a "text" column
        print(count)
        text = str(text)
        count+=1
        label, probs = predict_text(text)
        if label == "suicide":
            results.append({"text": text, "probability": probs[0][1]})
            flagged_texts.append(text)
            multi_class_predict = predict_text_multi_class(text)
            multi_class_predictions[multi_class_predict].append(text)
    print("--> Completed model running")
    results.sort(key=lambda x: x["probability"], reverse=True)

    # Get word frequency from flagged texts
    frequent_words = extract_frequent_words(flagged_texts)
    print("--> Completed frequent_words")

    msg_output = get_systemMessage(multi_class_predictions)
    print(msg_output)
    predicted_class = [msg_output[0]]
    message = msg_output[1]
    print("--> Completed system message")

    return {"flagged_texts": results, "frequent_words": frequent_words, "predicted_class": predicted_class, "message": message}
    # return {
    #         "flagged_texts": results,
    #         "frequent_words": frequent_words,
    #         "predicted_class": predicted_class,
    #         "message": message 
    #     }
    # except Exception as e:
    #     print(f"Error processing file: {str(e)}")
    #     raise HTTPException(status_code=500, detail=str(e))
