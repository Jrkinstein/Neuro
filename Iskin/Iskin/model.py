import torch
import torch.nn as nn
import torch.optim as optim
from data_utils import text_to_indices, build_vocab, tokenize
from internet_utils import get_information_from_internet, is_internet_available
import json
import os
import shutil
from typing import List, Tuple

class HybridNet(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, personality_dim: int, texts, labels, emotion_labels, word_to_index, personalities, index_to_word):
        super(HybridNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim + personality_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 10)  # Классификация текста
        self.emotion_fc = nn.Linear(hidden_size, 9)  # Классификация эмоций, изменили на 8 для сложных эмоций
        self.personality_dim = personality_dim
        self.collected_data = self.collected_data = {
            "texts": texts,
            "labels": labels,
            "emotion_labels": emotion_labels,
            "personalities": personalities,
            "word_to_index": word_to_index,
            "index_to_word": index_to_word
        }  # Список для хранения данных

    def forward(self, x: torch.Tensor, personality: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.embedding(x)
        personality = personality.unsqueeze(1).repeat(1, x.size(1), 1)
        x = torch.cat((x, personality), dim=2)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        text_class = self.fc(x)
        emotion_class = self.emotion_fc(x)
        return text_class, emotion_class

    def train_model(self, texts: List[str], labels: List[int], emotion_labels: List[int], word_to_index: dict, personalities: List[List[float]], epochs: int = 10, learning_rate: float = 0.001, log_path: str = "training_logs.txt"):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        with open(log_path, 'a') as log_file:
            for epoch in range(epochs):
                running_loss = 0.0
                for i, text in enumerate(texts):
                    inputs = torch.tensor(text_to_indices(text, word_to_index)).unsqueeze(0)
                    labels_tensor = torch.tensor([labels[i]])
                    emotion_labels_tensor = torch.tensor([emotion_labels[i]])
                    personality_tensor = torch.tensor(personalities[i]).unsqueeze(0)

                    optimizer.zero_grad()
                    text_output, emotion_output = self.forward(inputs, personality_tensor)
                    text_loss = criterion(text_output, labels_tensor)
                    emotion_loss = criterion(emotion_output, emotion_labels_tensor)
                    loss = text_loss + emotion_loss
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

                log_message = f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(texts)}\n"
                log_file.write(log_message)
                print(log_message)

    def adapt(self, text: str, label: int, emotion_label: int, word_to_index: dict, personality: List[float], log_path: str = "training_logs.txt"):
        self.train_model([text], [label], [emotion_label], word_to_index, [personality], 1, 0.001, log_path)

    def save_model(self, path: str):
        backup_path = path.replace('.pt', '_backup.pt')
        if os.path.exists(path):
            shutil.copyfile(path, backup_path)
            print(f"Резервная копия модели сохранена по пути: {backup_path}")
        torch.save(self.state_dict(), path)
        print(f"Модель сохранена по пути: {path}")

    def load_model(self, path: str):
        self.load_state_dict(torch.load(path))
        print(f"Модель загружена с пути: {path}")

    def save_data(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.collected_data, f)
        print(f"Данные сохранены по пути: {path}")

    def load_data(self, path: str):
        if os.path.exists(path):
            with open(path, 'r') as f:
                self.collected_data = json.load(f)
            print(f"Данные загружены с пути: {path}")

    def communicate(self, input_text: str, word_to_index: dict, index_to_word: dict, personality: List[float]) -> str:
        self.eval()
        inputs = torch.tensor(text_to_indices(input_text, word_to_index)).unsqueeze(0)
        personality_tensor = torch.tensor(personality).unsqueeze(0)
        text_output, emotion_output = self.forward(inputs, personality_tensor)
        text_class = text_output.argmax(1).item()
        emotion_class = emotion_output.argmax(1).item()
        response = self.generate_dynamic_response(input_text, word_to_index, index_to_word, personality)
        return response

    def monitor_performance(self, log_path: str):
        with open(log_path, 'r') as log_file:
            lines = log_file.readlines()
            total_loss = sum(float(line.split("Loss: ")[1]) for line in lines if "Loss: " in line)
            avg_loss = total_loss / len(lines)
            print(f"Average Loss: {avg_loss}")
            if avg_loss > 1.0:
                print("Loss is high, adapting the model...")
                self.search_for_improvements()

    def generate_creative_idea(self, prompt: str, word_to_index: dict, index_to_word: dict, personality: List[float], max_length: int = 50) -> str:
        indices = text_to_indices(prompt, word_to_index)
        inputs = torch.tensor(indices).unsqueeze(0)
        personality_tensor = torch.tensor(personality).unsqueeze(0)
        generated_indices = indices[:]

        self.eval()
        for _ in range(max_length):
            with torch.no_grad():
                text_output, _ = self.forward(inputs, personality_tensor)
                next_word_index = text_output.argmax(1).item()
                generated_indices.append(next_word_index)
                inputs = torch.tensor(generated_indices).unsqueeze(0)

        generated_text = ' '.join([index_to_word[idx] for idx in generated_indices])
        return generated_text

    def search_for_improvements(self):
        if not is_internet_available():
            print("No internet connection. Cannot search for improvements.")
            return

        new_data = get_information_from_internet("latest AI research")
        if not new_data:
            print("No new data found for improvements.")
            return

        self.collected_data.extend(new_data)
        new_texts = [item['title'] for item in new_data]
        word_count = {}
        for text in new_texts:
            for word in tokenize(text):
                if word not in word_count:
                    word_count[word] = 1
                else:
                    word_count[word] += 1

        common_words = [word for word, count in word_count.items() if count > 1]
        for word in common_words:
            self.adapt(word, 0, 0, word_to_index, personalities[0])
        print("Model adapted based on new data from the internet.")

    def analyze_internet_data(self):
        if not is_internet_available():
            print("No internet connection. Cannot analyze data.")
            return

        new_data = get_information_from_internet("latest news")
        if not new_data:
            print("No new data found for analysis.")
            return

        self.collected_data.extend(new_data)  # Сохраняем данные
        new_texts = [item['title'] for item in new_data]
        word_count = {}
        for text in new_texts:
            for word in tokenize(text):
                if word not in word_count:
                    word_count[word] = 1
                else:
                    word_count[word] += 1

        verified_info = " ".join(word for word, count in word_count.items() if count > 1)
        print(f"Verified Information: {verified_info}")

    def generate_dynamic_response(self, input_text: str, word_to_index: dict, index_to_word: dict, personality: List[float], max_length: int = 50) -> str:
        if is_internet_available():
            internet_data = get_information_from_internet(input_text)
            if internet_data:
                response_texts = [item['snippet'] for item in internet_data]
                response_text = " ".join(response_texts)
                if response_text:
                    return response_text

        indices = text_to_indices(input_text, word_to_index)
        inputs = torch.tensor(indices).unsqueeze(0)
        personality_tensor = torch.tensor(personality).unsqueeze(0)
        generated_indices = indices[:]

        self.eval()
        for _ in range(max_length):
            with torch.no_grad():
                text_output, _ = self.forward(inputs, personality_tensor)
                next_word_index = text_output.argmax(1).item()
                generated_indices.append(next_word_index)
                inputs = torch.tensor(generated_indices).unsqueeze(0)

        generated_text = ' '.join([index_to_word[idx] for idx in generated_indices])
        return generated_text
