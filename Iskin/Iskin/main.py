import threading
import signal
import sys
import os
import time
from typing import List
from model import HybridNet
from data_utils import build_vocab

def improvement_thread(model: HybridNet, word_to_index: dict, index_to_word: dict, personalities: List[List[float]]):
    while True:
        try:
            model.monitor_performance("training_logs.txt")
            model.search_for_improvements()
            model.save_model("hybrid_net_model.pt")
            model.save_data("collected_data.json")
        except Exception as e:
            print(f"Exception in improvement_thread: {e}")
        time.sleep(3600)  # Каждые 1 час

def communication_thread(model: HybridNet, word_to_index: dict, index_to_word: dict, personalities: List[List[float]]):
    while True:
        try:
            user_input = input("Enter your message: ")
            personality = personalities[0]  # Используем первую личность для общения
            response = model.communicate(user_input, word_to_index, index_to_word, personality)
            print(f"Response: {response}")
        except Exception as e:
            print(f"Exception in communication_thread: {e}")

def internet_data_analysis_thread(model: HybridNet):
    while True:
        try:
            model.analyze_internet_data()
            model.save_model("hybrid_net_model.pt")
            model.save_data("collected_data.json")
        except Exception as e:
            print(f"Exception in internet_data_analysis_thread: {e}")
        time.sleep(3600)  # Каждые 1 час

def signal_handler(sig, frame):
    print('Saving model and data before exiting...')
    model.save_model("hybrid_net_model.pt")
    model.save_data("collected_data.json")
    sys.exit(0)

if __name__ == "__main__":
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

    labels = [0, 1, 0, 1, 1, 0, 1, 0, 1, 1] * 10
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
    ] * 10
    
    word_to_index, index_to_word = build_vocab(texts)
    model = HybridNet(len(word_to_index), 50, 100, len(personalities[0]), texts, labels, emotion_labels, word_to_index, personalities, index_to_word)

    # Загрузка модели и данных, если они существуют
    if os.path.exists("hybrid_net_model.pt"):
        model.load_model("hybrid_net_model.pt")
    if os.path.exists("collected_data.json"):
        model.load_data("collected_data.json")
   

    model.train_model(texts, labels, emotion_labels, word_to_index, personalities, epochs=10)

    signal.signal(signal.SIGINT, signal_handler)

    threading.Thread(target=improvement_thread, args=(model, word_to_index, index_to_word, personalities)).start()
    threading.Thread(target=communication_thread, args=(model, word_to_index, index_to_word, personalities)).start()
    threading.Thread(target=internet_data_analysis_thread, args=(model,)).start()
