import time
import numpy as np
from selenium import webdriver
from selenium.common import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from bs4 import BeautifulSoup
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def knn_predict(numbers):
    print("KNN INPUT NUMBERS: ",numbers)
    features, labels = create_features_and_labels(numbers)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    best_params = {'n_neighbors': 8}

    model = KNeighborsClassifier(**best_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy for KNN: {test_accuracy}")
    print("KNN numbers: ", numbers)
    last_number = numbers[-1]
    next_color_prediction = model.predict([[last_number]])
    print(f"KNN przewiduje kolor: {next_color_prediction[0]}")

    return next_color_prediction[0]


def random_forest_predict(numbers):
    features, labels = create_features_and_labels(numbers)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    best_params = {'max_depth': None, 'n_estimators': 50, 'random_state': 42}

    model = RandomForestClassifier(**best_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy for Random Forest: {test_accuracy}")

    last_number = numbers[-1]
    next_color_prediction = model.predict([[last_number]])
    print(f"Random Forest przewiduje kolor: {next_color_prediction[0]}")

    return next_color_prediction[0]


def svc_predict(numbers):
    features, labels = create_features_and_labels(numbers)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    best_params = {'random_state': 42, 'kernel': 'poly', 'gamma': 'auto', 'degree': 2, 'C': 100}

    model = SVC(**best_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy for SVC: {test_accuracy}")

    last_number = numbers[-1]
    next_color_prediction = model.predict([[last_number]])
    print(f"SVC przewiduje kolor: {next_color_prediction[0]}")

    return next_color_prediction[0]


def gradient_boosting_predict(numbers):
    features, labels = create_features_and_labels(numbers)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    best_params = {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 150, 'random_state': 42}

    model = GradientBoostingClassifier(**best_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy for Gradient Boosting: {test_accuracy}")

    last_number = numbers[-1]
    next_color_prediction = model.predict([[last_number]])
    print(f"Gradient Boosting przewiduje kolor: {next_color_prediction[0]}")

    return next_color_prediction[0]


def update_numbers(driver, numbers, refresh_interval=10):
    numbers.reverse()
    while True:
        driver.refresh()
        time.sleep(2)

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

        new_numbers_unique = []

        for i in range(10):
            # print("1: ", new_numbers[i], " : ", numbers[0])
            # print("2: ", new_numbers[i + 1], " : ", numbers[1])
            # print("3: ", new_numbers[i + 2], " : ", numbers[2])

            if new_numbers[i] == numbers[0] and new_numbers[i + 1] == numbers[1] and new_numbers[i + 2] == numbers[
                2] and i != 0:
                # print("I(wazne): ", i)
                print("[NEW UNIQUE NUMERS]:", new_numbers_unique)
                print("[NUMERS BEFORE]: ", numbers)
                new_numbers_unique.reverse()
                for j, value in enumerate(new_numbers_unique):
                    numbers.insert(0, value)
                print("[NUMERS AFTER]: ", numbers)
                # Przenieśmy część odpowiedzialną za obliczanie i wyświetlanie predykcji tutaj
                features, labels = create_features_and_labels(numbers)
                X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

                last_number = numbers[0]

                numbers.reverse()
                gradient_boosting_predict(numbers)
                knn_predict(numbers)
                random_forest_predict(numbers)
                numbers.reverse()

                break
            elif new_numbers[i] == numbers[0] and new_numbers[i + 1] == numbers[1] and new_numbers[i + 2] == numbers[
                2] and i == 0:
                print("[NO NEW NUMBER]");
                break
            else:
                new_numbers_unique.append(new_numbers[i])

        time.sleep(5)


def scrape_csgoempire_numbers():
    input("Enter")

    divs = driver.find_elements(By.CSS_SELECTOR, "div[data-v-3df2c054]")

    # Zaktualizowane pobieranie liczb
    numbers = []
    for div in divs:
        text = div.find_element(By.CSS_SELECTOR, "span[data-v-3df2c054]").text
        if '-' in text:
            index = text.find('-') + 1
            number = int(text[index:].strip())
            numbers.append(number)

    print(f"Pobrano liczby: {numbers}")

    # Usuń pierwsze dwie wartości z listy
    return numbers[2:]

def scrape_csgoempire_numbers_v3(driver):
    url = "https://csgoempire.com/history?seed=3025"
    driver.get(url)

    input("Enter")
    try:
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "div[data-v-3df2c054]")))
    except TimeoutException:
        print("Error: Timed out waiting for page to load.")
        return []

    page_source = driver.page_source
    soup = BeautifulSoup(page_source, 'html.parser')
    divs = soup.select("div[data-v-3df2c054]")


    # Zaktualizowane pobieranie liczb
    numbers = []
    for div in divs:
        span = div.find("span")
        if not span:
            span = div.find_element_by_xpath('.//span')
        if span:
            text = span.text
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

    for i in range(1, len(numbers)):
        features.append(numbers[i])
        labels.append(number_to_color(numbers[i - 1]))

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

    # chrome_options.add_argument("--headless")
    chrome_options.add_argument("--blink-settings=imagesEnabled=false")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-javascript")

    service = Service(executable_path=driver_path)

    driver = webdriver.Chrome(service=service, options=chrome_options)

    url = "https://csgoempire.com/history?seed=3025"

    driver.get(url)

    numbers = scrape_csgoempire_numbers_v3(driver)

    print("WCZYTANO")
    print(numbers)
    save_numbers_to_file(numbers)
    print("saved")

    numbers = load_numbers_from_file()
    features, labels = create_features_and_labels(numbers)
    update_numbers(driver, numbers)
