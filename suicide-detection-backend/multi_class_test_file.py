from collections import Counter
from huggingface_hub import InferenceClient

client = InferenceClient(
    provider="hf-inference",
    api_key=HUGGINGFACE_API_KEY,
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
