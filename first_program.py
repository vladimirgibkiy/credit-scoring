import math

def calculate_median(data_list: list) -> float:
    if data_list:
        result = []
        for item in data_list:
            if isinstance(item, (int, float)) and not isinstance(item, bool):
                result.append(item)

        result.sort()
        n = len(result)

        if n % 2 == 1:
            median = result[n // 2]
        else:
            median = (result[n // 2 - 1] + result[n // 2]) / 2

        return median

    else:
        return 0.0

def preprocess_data(raw: list) -> tuple:
    ages = [row[0] for row in raw]
    incomes = [row[1] for row in raw]

    median_age = calculate_median(ages)
    median_income = calculate_median(incomes)

    cleaned_data = []

    for item in raw:
        container = []

        if item[0] is None:
            container.append(median_age)
        else:
            container.append(item[0])

        if item[1] is None:
            container.append(median_income)
        else:
            container.append(item[1])

        container.append(item[2])
        container.append(item[3])

        cleaned_data.append(container)

    cleaned_ages = [item[0] for item in cleaned_data]
    cleaned_incomes = [item[1] for item in cleaned_data]

    min_age = min(cleaned_ages)
    max_age = max(cleaned_ages)

    min_income = min(cleaned_incomes)
    max_income = max(cleaned_incomes)

    x = []
    y = []

    for row in cleaned_data:
        container = []
        scaled_age = (row[0] - min_age) / (max_age - min_age + 1e-15)
        scaled_income = (row[1] - min_income) / (max_income - min_income + 1e-15)

        container.append(scaled_age)
        container.append(scaled_income)

        if row[2] is None:
            container.append(0.0)
        else:
            container.append(row[2])

        x.append(container)
        y.append(row[3])

    return x, y

#print(preprocess_data(raw_data))

def train_test_split(x: list, y: list, test_size = 0.2) -> tuple:

    partition_index = int(len(x) * (1 - test_size))

    x_train = x[:partition_index]
    y_train = y[:partition_index]

    x_test = x[partition_index:]
    y_test = y[partition_index:]

    return x_train, y_train, x_test, y_test

class LogisticRegression:

    def __init__(self) -> None:
        self.weights = None
        self.bias = 0.0

    def _sigmoid(self, z: float) -> float:
        """
        Вычисляет сигмоиду с защитой от переполнения (OverflowError).
        Границы 20 и -20 выбраны, так как при |z| > 20 значение 
        сигмоиды неотличимо от 1.0 или 0.0 в рамках точности float.
        """
        if z > 20:
            return 1.0        
        elif z < -20:
            return 0.0
        else:
            return 1 / (1 + math.exp(-z))

    def predict_proba(self, X: list) -> list:
        """
        Принимает список списков X (признаки клиентов).
        Возвращает плоский список вероятностей дефолта для каждого клиента.
        """
        probabilities = []
        
        for row in X:

            z = 0.0
            for i in range(len(row)):
                z += row[i] * self.weights[i]
            z += self.bias

            prob = self._sigmoid(z)

            probabilities.append(prob)

        return probabilities
    
    def fit(self, X: list, y: list, epochs: int = 500, lr: float = 0.1):
        """
        Обучение модели методом градиентного спуска на чистом Python.
        """
        num_samples = len(X)
        # БЕРЕМ ДЛИНУ ПЕРВОЙ СТРОКИ, ЧТОБЫ ПОЛУЧИТЬ КОЛИЧЕСТВО ПРИЗНАКОВ (3)
        num_features = len(X[0]) 
        
        # Инициализируем веса нулями: [0.0, 0.0, 0.0]
        self.weights = [0.0] * num_features
        self.bias = 0.0
        
        # Главный цикл обучения
        for epoch in range(epochs):
            y_pred = self.predict_proba(X)
            
            dw = [0.0] * num_features
            db = 0.0
            
            # Считаем градиенты
            for i in range(num_samples):
                error = y_pred[i] - y[i]
                db += error
                for j in range(num_features):
                    dw[j] += error * X[i][j]
            
            # Находим среднее
            db = db / num_samples
            for j in range(num_features):
                dw[j] = dw[j] / num_samples
            
            # Обновляем параметры
            self.bias = self.bias - lr * db
            for j in range(num_features):
                self.weights[j] = self.weights[j] - lr * dw[j]


raw_data = [(25.0, 50000.0, 1.0, 0), (None, 32000.0, 0.0, 1), (45.0, None, 1.0, 0)]

x_full, y_full = preprocess_data(raw_data)
x_train, y_train, x_test, y_test = train_test_split(x_full, y_full, test_size=0.2)


model = LogisticRegression()
model.fit(x_train, y_train, epochs=100, lr=0.1)

print("Обученные веса:", model.weights)
print("Обученное смещение:", model.bias)

preds = model.predict_proba(x_test)
print("Предсказание на тесте:", preds)






