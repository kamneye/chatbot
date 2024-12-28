# chatbot
import pandas as pd
import numpy as np

# Sample data for the chatbot (questions and answers in a structured format)
data = {
    'question': ['What is your name?', 'How are you?', 'What is your age?', 'Tell me a joke', 'What is Python?'],
    'answer': ['I am a chatbot.', 'I am doing great, thank you!', 'I am ageless!', 'Why don’t skeletons fight each other? They don’t have the guts!', 'Python is a programming language.']
}

# Create a DataFrame using pandas
df = pd.DataFrame(data)

# Function to clean the user input (convert to lowercase and remove extra spaces)
def clean_input(user_input):
    return user_input.strip().lower()

# Function to find the best matching response
def get_answer(user_input):
    cleaned_input = clean_input(user_input)
    
    # Basic similarity check using Numpy's vectorization (or you could use more advanced methods)
    # We will compare the cleaned input with the predefined questions using simple exact matching
    for i, row in df.iterrows():
        if cleaned_input in clean_input(row['question']):
            return row['answer']
    
    return "Sorry, I don't understand that question."

# Interactive loop to chat with the bot
print("Hello! I am your chatbot. Ask me anything.")
while True:
    user_input = input("You: ")
    
    # Exit condition
    if user_input.lower() == 'exit':
        print("Goodbye!")
        break
    
    # Get response from the chatbot
    response = get_answer(user_input)
    print("Bot:", response)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to calculate the best match based on cosine similarity 
def get_answer_using_similarity(user_input):
    # Combine user input with all questions to calculate similarity
    questions = df['question'].tolist()
    questions.append(user_input)  # Add user input to the list
    
    # Convert text data into vectors (bag of words model)
    vectorizer = CountVectorizer().fit_transform(questions)
    
    # Calculate cosine similarity between the user input and all predefined questions
    cosine_sim = cosine_similarity(vectorizer[-1], vectorizer[:-1])
    
    # Get the index of the best match
    best_match_index = np.argmax(cosine_sim)
    
    return df.iloc[best_match_index]['answer']

# Updated interaction loop using similarity
while True:
    user_input = input("You: ")
    
    if user_input.lower() == 'exit':
        print("Goodbye!")
        break
    
    response = get_answer_using_similarity(user_input)
    print("Bot:", response)
