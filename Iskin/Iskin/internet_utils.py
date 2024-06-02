import requests
from bs4 import BeautifulSoup

# Функция для получения данных из интернета
def get_information_from_internet(query: str) -> list:
    url = f"https://www.google.com/search?q={query}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    results = []

    for g in soup.find_all('div', class_='BNeawe vvjwJb AP7Wnd'):
        results.append({'title': g.get_text()})
    
    return results

# Функция для проверки доступности интернета
def is_internet_available() -> bool:
    try:
        requests.get("http://www.google.com", timeout=5)
        return True
    except requests.ConnectionError:
        return False
