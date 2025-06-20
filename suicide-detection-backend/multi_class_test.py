from collections import Counter
# from fastapi import FastAPI, UploadFile, File
# import pandas as pd
# import torch
# from transformers import BertTokenizer, BertForSequenceClassification, AdamW
# import torch.nn.functional as F
# from io import BytesIO
# from fastapi.middleware.cors import CORSMiddleware


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MULTI_CLASS_MODEL_SAVE_PATH = "E:\Capstone Project\dashboard\suicide-detection-backend\bert_mental_disorder_model_final_multiClass"
# print(1)
# MULTI_CLASS_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# print(2)
# MULTI_CLASS_model = BertForSequenceClassification.from_pretrained(
#     "bert-base-uncased",
#     num_labels=5,  # Number of classes
#     # hidden_dropout_prob=0.2,   
#     # attention_probs_dropout_prob=0.2,
# )
# print(3)
# MULTI_CLASS_model.load_state_dict(torch.load(r"E:\Capstone Project\dashboard\suicide-detection-backend\bert_mental_disorder_model_final_multiClass\best_model.pth", map_location=torch.device('cpu')))
# print(4)
# MULTI_CLASS_model.to(device)
# print(5)
# print(device)

# def predict_text_multi_class(text):
#     """Predict if the given text is suicide-related or not."""
#     inputs = MULTI_CLASS_tokenizer(text, padding="max_length", truncation=True, return_tensors="pt")
#     inputs = {key: value.to(device) for key, value in inputs.items()}

#     with torch.no_grad():
#         outputs = MULTI_CLASS_model(inputs)

#     logits = outputs.logits
#     probs = F.softmax(logits, dim=-1)
#     predicted_class = torch.argmax(logits, dim=-1).item()

#     print(predicted_class)

#     return predicted_class

# df = pd.read_csv("E:\Capstone Project\main_chat.csv")

# results = []
# flagged_texts = []

# for text in df["text"]:  # Ensure CSV has a "text" column
#     label = predict_text_multi_class(text)
#     flagged_texts.append(label)

# print(flagged_texts)

# def get_systemMessage(pred):
#     """Get system message based on multi class predictions"""
#     prediction_counts = Counter(pred)
#     return prediction_counts
# l=[3, 3, 3, 4, 2, 2, 3, 3, 1, 3, 3, 3, 2, 2, 3, 3, 2, 2, 4, 2, 2, 3, 3, 2, 2, 3, 1, 3, 0, 3]
# print(get_systemMessage(l)[3])

# hf api key: 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





# import torch

# from transformers import AutoTokenizer, AutoModel
# tokenizer = AutoTokenizer.from_pretrained("AIMH/mental-roberta-large")
# model = AutoModel.from_pretrained("AIMH/mental-roberta-large")




# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # set_seed(0)
# prompt = """I am feeling sad and depressed, What I need to do?."""
# model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
# input_length = model_inputs.input_ids.shape[1]
# generated_ids = model.generate(model_inputs, max_new_tokens=20)
# print(tokenizer.batch_decode(generated_ids[:, input_length:], skip_special_tokens=True)[0])






from huggingface_hub import InferenceClient

client = InferenceClient(
    provider="hf-inference",
    api_key="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",
)

def generate_support_message(chat_history, predicted_class):
    # Extract concerning phrases from chat history
    concerning_phrases = chat_history
    # concerning_phrases = [
    #     f'"{msg}"' for msg in chat_history 
    #     if any(keyword in msg.lower() for keyword in ["die", "hopeless", "worthless", "can't go on", "end it"])
    # ][:3]  # Limit to 3 examples
    
    # Condition-specific guidance
    condition_responses = {
        "depression": {
            "description": "You've been expressing feelings of deep sadness, hopelessness, or fatigue in your messages.",
            "examples": concerning_phrases,
            "support": "It sounds like you're carrying a heavy weight. You're not alone—many people feel this way, and help exists.",
            "resources": "Consider reaching out to a trusted friend or a counselor. Resources like the National Suicide Prevention Lifeline (988) are available 24/7."
        },
        "suicidewatch": {
            "description": "Your messages include thoughts about ending your life, which is extremely concerning.",
            "examples": concerning_phrases,
            "support": "Your life matters deeply. These thoughts are a sign of pain, not a solution.",
            "resources": "Please contact a crisis line immediately (e.g., 988 in the US or a local service). You don't have to face this alone."
        },
        "anxiety": {
            "description": "Your chats suggest overwhelming worry or fear, which can feel paralyzing.",
            "examples": concerning_phrases,
            "support": "Anxiety can make the world feel unsafe. Small steps to ground yourself (like deep breathing) can help.",
            "resources": "Apps like Calm or Headspace offer guided exercises. A therapist can also provide coping strategies."
        },
        "offmychest": {
            "description": "You've been sharing heavy emotions, which takes courage.",
            "examples": concerning_phrases,
            "support": "Getting things off your chest is a brave first step. Holding space for your feelings is important.",
            "resources": "Talking to someone who listens without judgment (like a support group) can be healing."
        }
    }
    print(chr(10).join(f'- "{phrase}"' for phrase in concerning_phrases))
    # Generate the LLM prompt
    prompt = f"""
    You are a mental health support assistant tasked with helping a guardian understand and support a user who may be struggling with {predicted_class}.  
    The user's chat history includes concerning phrases like:  
    {chr(10).join(f'- "{phrase}"' for phrase in concerning_phrases)}  

    Guidelines for your response:  
    1. Audience: Address the guardian directly (e.g., "Based on [User]'s messages...").  
    2. Validation: Acknowledge the severity of the user's emotions *without* alarming the guardian unnecessarily.  
    - Example: *"[User] mentioned '[concerning_phrase]', which suggests they're feeling [interpretation]."*  
    3. Specifics: Highlight 1-2 of the most concerning phrases and explain their implications.  
    4. Actionable Steps: Provide the guardian with:  
    - Immediate actions (e.g., "Check in with [User] privately and ask open-ended questions like, 'How can I support you right now?'").  
    - Resources: Share {condition_responses[predicted_class]['resources']} and emphasize confidentiality.  
    5. Tone:  
    - Compassionate but urgent for high-risk cases (e.g., suicidewatch).  
    - Reassuring but vigilant for others (e.g., anxiety/depression).  

    Avoid:  
    - Judgmental language (e.g., "They're overreacting").  
    - Generic advice (e.g., "Just be there for them").  
    """
    
    messages_for_llm = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": "Draft a response."}
    ]
    
    completion = client.chat.completions.create(
        model="HuggingFaceH4/zephyr-7b-alpha",
        messages=messages_for_llm,
        max_tokens=500,
    )
    return completion.choices[0].message.content

# Example usage
chat_history = [
    "I can't take this anymore.",
    "No one would care if I was gone.",
    "I just want to disappear."
]
predicted_class = "suicidewatch"
support_message = generate_support_message(chat_history, predicted_class)
print(support_message)
