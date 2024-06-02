import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import json
from pymystem3 import Mystem
import nltk

# Загрузка данных NLTK
nltk.download('stopwords')
nltk.download('punkt')

# Инициализация Mystem
mystem = Mystem()

# Загрузка данных
def load_data():
    texts = [
        "Я очень рад сегодня!",
        "Это худший день в моей жизни.",
        "Я в восторге от нового проекта.",
        "Это так раздражает и бесит.",
        "Я боюсь предстоящего экзамена.",
        "Какой чудесный мир!",
        "Я в депрессии и чувствую себя подавленным.",
        "Это захватывающая возможность!",
        "Я беспокоюсь о будущем.",
        "Это отвратительно и ужасно.",
        "У меня много работы, и я очень устал.",
        "Сегодня был отличный день на работе.",
        "Я не могу поверить, что это произошло.",
        "Это прекрасное место для отдыха.",
        "Я так рад, что мы встретились.",
        "Этот фильм был очень интересным.",
        "Я не хочу идти на работу завтра.",
        "Сегодня я узнал много нового.",
        "Этот ресторан готовит вкусную еду.",
        "Я хочу провести выходные с семьей.",
        "Этот день был полон неожиданностей.",
        "Мне нужно больше времени для отдыха.",
        "Я люблю гулять в парке.",
        "Этот город такой красивый.",
        "Я устал от постоянного стресса.",
        "Сегодня я чувствую себя лучше.",
        "Я очень благодарен за вашу помощь.",
        "Эти выходные были очень скучными.",
        "Я волнуюсь перед важной встречей.",
        "Этот подарок был очень приятным сюрпризом.",
        "Мне нравится слушать музыку.",
        "Я расстроен из-за плохих новостей.",
        "Я хочу попробовать что-то новое.",
        "Сегодня я встретил старого друга.",
        "Я не могу найти свои ключи.",
        "Этот проект требует много усилий.",
        "Я рад, что мы наконец-то закончили.",
        "Я чувствую себя одиноким.",
        "Этот урок был очень полезным.",
        "Мне не нравится эта погода.",
        "Я не могу уснуть из-за шума.",
        "Сегодня я научился готовить новое блюдо.",
        "Я рад, что у меня есть такая поддержка.",
        "Этот тест был очень сложным.",
        "Я хочу улучшить свои навыки.",
        "Мне нравится проводить время на природе.",
        "Этот концерт был потрясающим.",
        "Я не могу справиться с этой задачей.",
        "Сегодня я чувствую себя вдохновленным.",
        "Мне нужно больше уверенности в себе.",
        "Этот парк такой спокойный и тихий.",
        "Я боюсь сделать ошибку.",
        "Мне нужно больше практики.",
        "Этот матч был очень напряженным.",
        "Я рад, что у меня есть хобби.",
        "Мне нужно больше мотивации.",
        "Этот фильм заставил меня задуматься.",
        "Я хочу больше путешествовать.",
        "Сегодня я помог другу с переездом.",
        "Мне не нравится моя работа.",
        "Этот магазин предлагает хороший выбор.",
        "Я хочу научиться новому языку.",
        "Мне нужно больше отдыха.",
        "Этот ресторан всегда переполнен.",
        "Я хочу улучшить свое здоровье.",
        "Сегодня я посетил интересную выставку.",
        "Мне нужно больше времени на подготовку.",
        "Этот тренинг был очень информативным.",
        "Я рад, что у меня есть верные друзья.",
        "Мне нужно больше терпения.",
        "Этот фильм был очень скучным.",
        "Я хочу больше заниматься спортом.",
        "Сегодня я получил хорошую новость.",
        "Мне не нравится работать по выходным.",
        "Этот парк всегда полон людей.",
        "Я хочу улучшить свои отношения.",
        "Сегодня я встретил интересного человека.",
        "Мне нужно больше уверенности.",
        "Этот урок был очень скучным.",
        "Я хочу больше времени проводить с семьей.",
        "Сегодня я помог соседу с покупками.",
        "Мне не нравится эта еда.",
        "Этот фильм был очень трогательным.",
        "Я хочу больше читать книги.",
        "Сегодня я узнал о новом проекте.",
        "Мне нужно больше спокойствия.",
        "Этот магазин предлагает отличные скидки.",
        "Я хочу улучшить свои знания.",
        "Сегодня я почувствовал себя лучше.",
        "Мне не нравится эта погода.",
        "Этот урок был очень интересным.",
        "Я хочу больше времени проводить на свежем воздухе.",
        "Сегодня я помог коллеге с работой.",
        "Мне нужно больше мотивации.",
        "Этот фильм был очень смешным.",
        "Я хочу больше времени для себя.",
    ]
    
    labels = [0, 1, 0, 1, 1, 0, 1, 0, 1, 1] * 10  # Повторим для увеличения объема данных
    emotion_labels = [0, 2, 3, 4, 5, 0, 6, 3, 7, 8] * 10
    personalities = [
        [0.8, 0.1, 0.1],
        [0.2, 0.7, 0.1],
        [0.9, 0.0, 0.1],
        [0.1, 0.8, 0.1],
        [0.3, 0.4, 0.3],
        [0.7, 0.2, 0.1],
        [0.2, 0.6, 0.2],
        [0.8, 0.1, 0.1],
        [0.3, 0.5, 0.2],
        [0.1, 0.9, 0.0],
        # Повторим для увеличения объема данных
    ] * 10
    
    return texts, labels, emotion_labels, personalities

# Предобработка текстов с использованием Mystem
def preprocess_texts(texts):
    stop_words = set(stopwords.words('russian'))
    processed_texts = []
    
    for text in texts:
        # Удаление специальных символов и цифр
        text = re.sub(r'[^а-яА-Я\s]', '', text, re.I|re.A)
        text = text.lower()  # Приведение текста к нижнему регистру
        text = text.strip()  # Удаление лишних пробелов
        tokens = word_tokenize(text)
        # Удаление стоп-слов
        tokens = [token for token in tokens if token not in stop_words]
        # Лемматизация с помощью Mystem
        lemmas = mystem.lemmatize(' '.join(tokens))
        lemmas = [lemma for lemma in lemmas if lemma.strip()]
        processed_texts.append(' '.join(lemmas))
    
    return processed_texts

# Построение словаря и преобразование текстов в индексы
def build_vocab_and_tokenize(texts):
    vectorizer = CountVectorizer(tokenizer=word_tokenize)
    vectorizer.fit(texts)
    word_to_index = vectorizer.vocabulary_
    index_to_word = {v: k for k, v in word_to_index.items()}
    tokenized_texts = vectorizer.transform(texts).toarray()
    return word_to_index, index_to_word, tokenized_texts

# Класс Dataset для DataLoader
class TextDataset(Dataset):
    def __init__(self, texts, labels, emotion_labels, personalities):
        self.texts = texts
        self.labels = labels
        self.emotion_labels = emotion_labels
        self.personalities = personalities

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            'text': torch.tensor(self.texts[idx], dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
            'emotion_label': torch.tensor(self.emotion_labels[idx], dtype=torch.long),
            'personality': torch.tensor(self.personalities[idx], dtype=torch.float)
        }

# Гибридная нейронная сеть
class HybridNet(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, personality_size):
        super(HybridNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.conv1 = nn.Conv1d(in_channels=embed_size, out_channels=hidden_size, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size + personality_size, 10)  # Для предсказания классов
        self.fc_emotion = nn.Linear(hidden_size + personality_size, 9)  # Для предсказания эмоций

    def forward(self, text, personality):
        x = self.embedding(text)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)
        x, _ = self.rnn(x)
        x = x[:, -1, :]
        x = torch.cat((x, personality), dim=1)
        class_output = self.fc(x)
        emotion_output = self.fc_emotion(x)
        return class_output, emotion_output

# Обучение модели
def train_model(model, dataloader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            texts = batch['text']
            labels = batch['label']
            emotion_labels = batch['emotion_label']
            personalities = batch['personality']

            optimizer.zero_grad()
            class_outputs, emotion_outputs = model(texts, personalities)
            loss_class = criterion(class_outputs, labels)
            loss_emotion = criterion(emotion_outputs, emotion_labels)
            loss = loss_class + loss_emotion
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}')

# Основная функция
def main():
    texts, labels, emotion_labels, personalities = load_data()
    texts = preprocess_texts(texts)
    word_to_index, index_to_word, tokenized_texts = build_vocab_and_tokenize(texts)

    dataset = TextDataset(tokenized_texts, labels, emotion_labels, personalities)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    vocab_size = len(word_to_index)
    embed_size = 50
    hidden_size = 100
    personality_size = 3

    model = HybridNet(vocab_size, embed_size, hidden_size, personality_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, dataloader, criterion, optimizer, num_epochs=10)

    # Сохранение модели и данных
    torch.save(model.state_dict(), 'Iskin\\hybrid_net_model.pt')
    with open('Iskin\\collected_data.json', 'w', encoding='utf-8') as f:
        json.dump({
            "texts": texts,
            "labels": labels,
            "emotion_labels": emotion_labels,
            "personalities": personalities,
            "word_to_index": word_to_index,
            "index_to_word": index_to_word
        }, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    main()
