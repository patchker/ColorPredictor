import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import threading
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from collections import Counter


# Zoptymalizowana funkcja find_duplicate_start

def update_numbers(driver, numbers, refresh_interval=10):
    numbers.reverse()
    while True:
        driver.refresh()
        time.sleep(10)

        roll_boxes = driver.find_elements(By.CSS_SELECTOR, "div.box.box-center div.box.box-small")

        divs = driver.find_elements(By.CSS_SELECTOR, "div[data-v-3df2c054]")

        # Zaktualizowane pobieranie liczb
        new_numbers = []
        for div in divs[:10]:
            text = div.find_element(By.CSS_SELECTOR, "span[data-v-3df2c054]").text
            if '-' in text:
                index = text.find('-') + 1
                number = int(text[index:].strip())
                new_numbers.append(number)
        new_numbers = new_numbers[2:]
        print("Numbers: ", numbers)
        print("New numbers: ", new_numbers)


        new_numbers_unique = []

        for i in range(5):
            # print("1: ", new_numbers[i], " : ", numbers[0])
            # print("2: ", new_numbers[i + 1], " : ", numbers[1])
            # print("3: ", new_numbers[i + 2], " : ", numbers[2])

            if new_numbers[i] == numbers[0] and new_numbers[i + 1] == numbers[1] and new_numbers[i + 2] == numbers[2] and i!=0:
                print("I(wazne): ", i)
                print("New unique numbers:", new_numbers_unique)
                numbers.insert(0, new_numbers_unique)
                append_numbers_to_file(new_numbers_unique)
                break
            else:
                numbers.insert(0, new_numbers_unique[0])
                print("NEW NUMBER:", numbers[0])
                print("Numbers: ", numbers)






def find_duplicate_start(new_numbers, existing_numbers):
    counter = Counter(existing_numbers)
    for i, new_number in enumerate(new_numbers):
        if counter[new_number] >= 1:
            return i
    return -1




def scrape_csgoempire_numbers():
    input("Enter")

    divs = driver.find_elements(By.CSS_SELECTOR, "div[data-v-3df2c054]")

    # Zaktualizowane pobieranie liczb
    numbers = []
    for div in divs[:10]:
        text = div.find_element(By.CSS_SELECTOR, "span[data-v-3df2c054]").text
        if '-' in text:
            index = text.find('-') + 1
            number = int(text[index:].strip())
            numbers.append(number)

    print(f"Pobrano liczby: {numbers}")

    # Usuń pierwsze dwie wartości z listy
    return numbers[2:]


def append_numbers_to_file(numbers, file_name="csgoroll_numbers.txt"):
    with open(file_name, "a") as file:
        for number in numbers:
            file.write(f"{number}\n")
            print(f"Zapisuję liczbę {number} do pliku")


def save_numbers_to_file(numbers, file_name="csgoroll_numbers.txt"):
    with open(file_name, "w") as file:
        for number in numbers:
            file.write(f"{number}\n")
            print(f"Zapisuję liczbę {number} do pliku")



def load_numbers_from_file(file_name="csgoroll_numbers.txt"):
    with open(file_name, "r") as file:
        numbers = [int(line.strip()) for line in file][::-1]
    return numbers


def create_features_and_labels(numbers):
    features = []
    labels = []

    for i in range(len(numbers) - 1):
        features.append(numbers[i])
        labels.append(number_to_color(numbers[i + 1]))

    features = np.array(features).reshape(-1, 1)
    labels = np.array(labels)

    return features, labels


def number_to_color(number):
    if number == 0:
        return "green"
    elif 1 <= number <= 7:
        return "red"
    else:
        return "black"


if __name__ == "__main__":
    driver_path = "path/to/chromedriver"  # Podaj prawidłową ścieżkę do chromedriver
    chrome_options = Options()

    #chrome_options.add_argument("--headless")
    chrome_options.add_argument("--blink-settings=imagesEnabled=false")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-javascript")

    service = Service(executable_path=driver_path)

    driver = webdriver.Chrome(service=service, options=chrome_options)

    url = "https://csgoempire.com/history?seed=3023"


    driver.get(url)

    numbers = scrape_csgoempire_numbers()


    print(numbers)
    save_numbers_to_file(numbers)
    print("saved")

    numbers = load_numbers_from_file()
    features, labels = create_features_and_labels(numbers)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # KNN model
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    knn_predictions = knn.predict(X_test)
    knn_accuracy = accuracy_score(y_test, knn_predictions)
    print(f"KNN accuracy: {knn_accuracy}")

    # Random Forest model
    random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
    random_forest.fit(X_train, y_train)
    rf_predictions = random_forest.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_predictions)
    print(f"Random Forest accuracy: {rf_accuracy}")

    # Wypisz przewidywany kolor dla obu modeli
    last_number = numbers[-1]
    knn_prediction = knn.predict([[last_number]])
    rf_prediction = random_forest.predict([[last_number]])

    print(f"KNN przewiduje kolor: {knn_prediction[0]}")
    print(f"Random Forest przewiduje kolor: {rf_prediction[0]}")
    update_numbers(driver, numbers)

