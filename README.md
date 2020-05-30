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
<pre><code>unzip data.zip CNNs/test/ &&  unzip models.zip CNNs/
</code></pre>

