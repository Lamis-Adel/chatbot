import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import json
import pickle
from time import sleep

stemmer = LancasterStemmer()

# load data
with open("intents.json") as file:
    data = json.load(file)

# preprocessing
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []
    
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))
    labels = sorted(labels)

    training = []
    output = []
    out_empty = [0 for _ in range(len(labels))]
 
    #creating training files
    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            bag.append(1 if w in wrds else 0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)
    #storing the preprocessed data to use recently
    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

# نموذج PyTorch
class ChatbotModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ChatbotModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1) #transform outputs to softmax function  probapilities

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)

input_size = len(training[0])
hidden_size = 8
output_size = len(output[0])
#create chat model
model = ChatbotModel(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# التدريب
try:
    model.load_state_dict(torch.load("model.pth"))
except:
    #training on the model
    for epoch in range(1000):
        inputs = torch.tensor(training, dtype=torch.float32)#transform training data to tensor
        labels_tensor = torch.tensor(np.argmax(output, axis=1), dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels_tensor)
        loss.backward()
        optimizer.step()
        #print loss every 100 repeatation
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/1000], Loss: {loss.item():.4f}')

    torch.save(model.state_dict(), "model.pth")#storing the trained model

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    #create graghing the bag
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return np.array(bag, dtype=np.float32)

def chat():
    print("Hi, how can I help you?")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        with torch.no_grad():
            input_bag = torch.tensor(bag_of_words(inp, words), dtype=torch.float32).unsqueeze(0)
            results = model(input_bag).numpy()
        
        print("Results:", results)  # طباعة النتائج للتحقق
        results_index = np.argmax(results, axis=1)  # تأكد من استخدام axis=1
        tag = labels[results_index[0]]  # استخدم الفهرس من النتائج
        
        if results[0][results_index[0]] > 0.8:  # تحقق من القيمة
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
            sleep(3)
            Bot = random.choice(responses)
            print(Bot)
        else:
            print("I don't understand!")

chat()