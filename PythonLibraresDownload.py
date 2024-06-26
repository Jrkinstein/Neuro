import os
import subprocess

# Указываем путь к Python (замените на свой путь)
python_path = "C:\\Users\\Artem\\AppData\\Local\\Programs\\Python\\Python312\\python.exe"

# Устанавливаем библиотеку requests
subprocess.run([python_path, "-m", "pip", "install", "requests"])
# Устанавливаем библиотеку torch
subprocess.run([python_path, "-m", "pip", "install", "torch"])
# Устанавливаем библиотеку scikit-learn
subprocess.run([python_path, "-m", "pip", "install", "scikit-learn"])
# Устанавливаем библиотеку numpy
subprocess.run([python_path, "-m", "pip", "install", "numpy"])
# Устанавливаем библиотеку pandas
subprocess.run([python_path, "-m", "pip", "install", "pandas"])
# Устанавливаем библиотеку pymystem3
subprocess.run([python_path, "-m", "pip", "install", "pymystem3"])
