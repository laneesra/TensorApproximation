# TensorApproximation
## Install
Для работы  необходимо  иметь  предустановленный  Python версии 3.6 и больше, а также модуль виртуальной среды venv, 
который можно установить с помощью команды:
<pre><code>sudo apt install -y python3-venv;</code></pre>
Исходный текст можно загрузить из репозитория:
<pre><code>git clone https://github.com/laneesra/TensorApproximation;</code></pre>
Настройку виртуальной  среды  и  установку всех  зависимостей можно совершить, выполнив команду:
<pre><code>cd TensorApproximation && chmod+xinstall.sh && ./install.sh;</code></pre>
Для активации окружения:
<pre><code>source TN_venv/bin/activate;</code></pre>
## Usage
Скачать веса моделей alexnet, alexnet_cp, alexnet_tt и тестовые данные по ссылке:
<pre>https://drive.google.com/drive/folders/1EMSnDLo4m_LeS_7M-JpURuptsmZvJyAY?usp=sharing</pre>
Выполнить в главной директории:
<pre><code>unzip test.zip -d CNNs/data/ &&  unzip models.zip -d CNNs/
</code></pre>
Тестовый файл для класса тензора и результатов: <code>Tensor/test.py</code>
В файле последовательно выполняется визуализация результатов ускорения и сжатия AlexNet, тестирование TT-SVD, CP-ALS и CPRAND, сравнение времени исходного полносвязного слоя и TT-слоя.

### Примеры работы с программным комплексом для ускорения сетей в <code>CNNs/main.py</code> 
Тестирование модели:
<pre><code>python3 main.py --eval --model alexnet --test_path data/test -v</code></pre>
<pre><code>python3 main.py --eval --model alexnet_cp --test_path data/test -v</code></pre>
<pre><code>python3 main.py --eval --model alexnet_tt --test_path data/test -v</code></pre>

Факторизация сверточного слоя с ключом 3:
<pre><code>python3 main.py --decompose --model alexnet --type conv --key 3 --factorization cp -v</code></pre>
Обучение с нуля:
<pre><code>python3 main.py --train --model alexnet --train_path data/train --epochs 100 -v --device gpu</code></pre>
Тонкая настройка модели:
<pre><code>python3 main.py --train --model “alexnet_cp_[‘3’]_conv” --train_path data/train --epochs 100 -v --device gpu</code></pre>
