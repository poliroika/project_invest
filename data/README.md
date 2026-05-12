# Data directory

Сюда попадают JSON, которые при **auto_download** создаёт сам проект, либо экспорт/скачивание через Freqtrade в `user_data/data/`.

Имена файлов и формат будут описаны в коде загрузчика (`src/pair_trading/data_loader.py`).

Крупные файлы данных **не** коммитятся (см. `.gitignore`).
