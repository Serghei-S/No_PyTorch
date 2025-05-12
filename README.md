# Простой NumPy-based Фреймворк для Нейронных Сетей

Этот репозиторий содержит простую реализацию основных строительных блоков нейронных сетей с использованием исключительно библиотеки NumPy (с минимальным использованием SciPy для сверток). Цель проекта — показать, как работают прямые (forward) и обратные (backward) проходы различных слоев и функций потерь.

## Особенности

-   **Базовый класс Module:** Абстрактный класс для всех слоев с методами `forward` и `backward`.
-   **Sequential Container:** Позволяет объединять модули (слои) в последовательную модель.
-   **Реализованные слои:**
    -   `Linear`: Полносвязный (линейный) слой.
    -   Активации: `ReLU`, `LeakyReLU`, `ELU`, `SoftPlus`, `SoftMax`, `LogSoftMax`.
    -   `BatchNormalization`: Нормализация по батчу.
    -   `Dropout`: Слой для регуляризации с выбыванием.
    -   `Conv2d`: Сверточный слой (с 'same' padding).
    -   `MaxPool2d`: Слой субдискретизации (pooling) по максимуму.
    -   `ChannelwiseScaling`: Слой масштабирования по каналам (встроен в BN, но оставлен).
-   **Базовый класс Criterion:** Абстрактный класс для функций потерь.
-   **Реализованные функции потерь:**
    -   `MSECriterion`: Среднеквадратичная ошибка.
    -   `ClassNLLCriterion`: Отрицательная логарифмическая правдоподобность (для использования с LogSoftMax).
    -   `ClassNLLCriterionUnstable`: (Не рекомендуется, для SoftMax, нестабильна).
-   **Простые Оптимизаторы:**
    -   `sgd_momentum`: Стохастический градиентный спуск с моментумом.
    -   `adam_optimizer`: Оптимизатор Adam.
-   **Проверка Градиентов:** Утилиты для численной проверки аналитических градиентов слоев и функций потерь.

## Структура кода

-   `Module`: Базовый класс с абстрактными методами `updateOutput` (forward) и `updateGradInput` (backward), а также методами для работы с параметрами (`accGradParameters`, `zeroGradParameters`, `getParameters`, `getGradParameters`).
-   `Sequential`: Модуль-контейнер, который выполняет последовательный `forward` и обратный `backward` через добавленные в него слои.
-   `Criterion`: Базовый класс для функций потерь с методами `updateOutput` (вычисление потери) и `updateGradInput` (вычисление градиента потери по входу).
-   Конкретные классы слоев наследуются от `Module` и реализуют `updateOutput` и `updateGradInput`. Сюда же добавляются параметры (если есть) и методы для работы с их градиентами.
-   Конкретные классы критериев наследуются от `Criterion`.
-   Функции оптимизаторов принимают параметры модели, градиенты и конфигурацию.
-   Функции `check_gradient_numerical`, `check_layer_gradients`, `check_criterion_gradients` содержат логику для проверки правильности реализации обратного прохода.

## Использование (Концептуальное)

Хотя полный цикл обучения (загрузка данных, итерации по батчам, цикл оптимизации) не включен в предоставленный код, базовое использование модулей выглядит так:

```python
# Пример создания простой модели
model = Sequential()
model.add(Linear(10, 20)) # Входной размер 10, выходной 20
model.add(ReLU())
model.add(Linear(20, 2)) # Выходной размер 2
model.add(LogSoftMax()) # Для классификации с ClassNLLCriterion

# Пример создания функции потерь
criterion = ClassNLLCriterion()

# Пример forward прохода (во время обучения)
input_data = np.random.randn(32, 10).astype(np.float32) # Батч из 32 примеров
model.train() # Переводим модель в режим обучения (важно для Dropout/BN)
output = model.forward(input_data)

# Пример вычисления потери и backward прохода
target_data = np.zeros((32, 2), dtype=np.float32) # Пример целевых данных (one-hot)
target_data[np.arange(32), np.random.randint(0, 2, 32)] = 1 # Случайные метки
loss = criterion.forward(output, target_data)
grad_from_criterion = criterion.backward(output, target_data)

# Обратный проход через модель
grad_input_to_model = model.backward(input_data, grad_from_criterion)

# Получение параметров и их градиентов для оптимизации
model_params = model.getParameters()
model_grads = model.getGradParameters()

# Применение оптимизатора (концептуально)
# optimizer_config = {'learning_rate': 0.001, 'momentum': 0.9}
# optimizer_state = {} # Состояние оптимизатора (например, для моментума или Адама)
# sgd_momentum(model_params, model_grads, optimizer_config, optimizer_state)

# После шага оптимизации нужно обнулить градиенты
# model.zeroGradParameters()

Проверка Градиентов
В коде реализованы функции для численной проверки корректности вычисления аналитических градиентов (обратного прохода). Это критически важно при реализации новых слоев.

Чтобы запустить проверки:

1. Установите RUN_GRADIENT_CHECKS = True в начале файла.
2. Запустите скрипт Python.
