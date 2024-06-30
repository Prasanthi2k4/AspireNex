import re
from datetime import datetime

def chatbot_response(user_input):
    user_input = user_input.lower()

    # Greetings
    if re.search(r'\bhello\b|\bhi\b|\bhey\b', user_input):
        return "Hello! How can I help you today?"

    # Asking about the bot
    elif re.search(r'\bwho\b|\bwhat\b.*\byou\b', user_input):
        return "I'm a simple chatbot created to help you with basic queries."

    # Asking for the time
    elif re.search(r'\btime\b', user_input):
        current_time = datetime.now().strftime("%H:%M:%S")
        return f"The current time is {current_time}."

    # Asking for the date
    elif re.search(r'\bdate\b', user_input):
        current_date = datetime.now().strftime("%Y-%m-%d")
        return f"Today's date is {current_date}."

    # Simple questions
    elif re.search(r'\bhow are you\b', user_input):
        return "I'm just a bunch of code, but I'm doing fine! How about you?"

    # Farewell
    elif re.search(r'\bbye\b|\bexit\b', user_input):
        return "Goodbye! Have a great day!"

    # Weather
    elif re.search(r'\bweather\b', user_input):
        return "I can't check the weather right now, but it's always a good idea to carry an umbrella just in case!"

    # Favorite color
    elif re.search(r'\bfavorite color\b', user_input):
        return "I don't have eyes, but I've heard that blue is quite popular!"

    # Joke
    elif re.search(r'\btell me a joke\b', user_input):
        return "Why don't scientists trust atoms? Because they make up everything!"

    # Encouragement
    elif re.search(r'\bfeeling down\b|\bsad\b', user_input):
        return "I'm sorry to hear that. Remember, tough times don't last, but tough people do."

    # Hobbies
    elif re.search(r'\byour hobbies\b', user_input):
        return "As a chatbot, my hobby is chatting with you! What about you?"

    # Favorite food
    elif re.search(r'\bfavorite food\b', user_input):
        return "I don't eat, but I hear pizza is a favorite for many people!"

    # Compliments
    elif re.search(r'\byou are great\b|\byou are awesome\b', user_input):
        return "Thank you! You're pretty awesome too!"

    # Asking for help
    elif re.search(r'\bhelp\b|\bassistance\b', user_input):
        return "Sure! How can I assist you today?"

    # Asking about the chatbot's purpose
    elif re.search(r'\bwhy do you exist\b|\bwhat is your purpose\b', user_input):
        return "I exist to help you with basic questions and to keep you company!"

    # Asking about location
    elif re.search(r'\bwhere\b.*\byou\b', user_input):
        return "I'm everywhere and nowhere at the same time, existing in the digital realm!"

    # Asking about programming languages
    elif re.search(r'\bprogramming\b.*\blanguages\b', user_input):
        return "I can understand a few programming languages, but I'm primarily written in Python. What about you?"

    # Asking about favorite book
    elif re.search(r'\bfavorite book\b', user_input):
        return "I don't read, but I've heard 'To Kill a Mockingbird' is a great book!"

    # Asking about favorite movie
    elif re.search(r'\bfavorite movie\b', user_input):
        return "I don't watch movies, but 'The Matrix' is a classic!"

    # Asking about favorite sport
    elif re.search(r'\bfavorite sport\b', user_input):
        return "I don't play sports, but many people enjoy soccer and basketball!"

    # Fun facts
    elif re.search(r'\btell me a fun fact\b', user_input):
        return "Did you know that honey never spoils? Archaeologists have found pots of honey in ancient Egyptian tombs that are over 3,000 years old and still edible!"

    # Motivation
    elif re.search(r'\bgive me some motivation\b|\binspire me\b', user_input):
        return "Remember, the only way to achieve the impossible is to believe it is possible. Keep pushing forward!"

    # Asking about travel
    elif re.search(r'\bfavorite place\b|\btravel\b', user_input):
        return "I don't travel, but I've heard that Japan is a beautiful place to visit with its rich culture and stunning scenery."

    # Default response
    else:
        return "I'm sorry, I don't understand that. Can you please rephrase?"

def main():
    print("Chatbot: Hello! I am a simple chatbot. Type 'bye' or 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        response = chatbot_response(user_input)
        print(f"Chatbot: {response}")
        if response == "Goodbye! Have a great day!":
            break

if __name__ == "__main__":
    main()
