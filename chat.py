import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

# Load trained model
FILE = "data.pth"
try:
    data = torch.load(FILE)
except FileNotFoundError:
    print("Error: 'data.pth' not found. Run 'python train.py' to train the model.")
    exit()

input_size = data.get("input_size")
hidden_size = data.get("hidden_size")
output_size = data.get("output_size")
all_words = data.get("all_words", [])
tags = data.get("tags", [])
model_state = data.get("model_state")

if not all([input_size, hidden_size, output_size, all_words, tags, model_state]):
    print("Error: 'data.pth' is corrupted or incomplete. Please re-train the model.")
    exit()

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"

def get_response(msg):
    sentence = tokenize(msg)
    print("User Input Tokenized:", sentence)

    X = bag_of_words(sentence, all_words)
    print("Bag of Words Vector:", X)

    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    print("Raw Model Output:", output)

    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    print(f"Predicted tag: {tag} (Confidence: {prob.item()})")

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])

    return "I'm sorry, I didn't understand that."

if __name__ == "__main__":
    print(f"{bot_name}: Hello! Let's chat. Type 'quit' to exit.")
    while True:
        sentence = input("You: ")
        if sentence.lower() == "quit":
            print(f"{bot_name}: Goodbye!")
            break

        response = get_response(sentence)
        print(f"{bot_name}: {response}")
