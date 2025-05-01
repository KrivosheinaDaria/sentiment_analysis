import pandas as pd
from model.SentimentAnalysis import SentimentAnalysis
import re
import nltk
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
import pymorphy2
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import f1_score

def preprocessing(text):
    text = text.lower()                                             # нижний регистр
    text = re.sub(r"http\S+|www\S+", "", text)         # ссылки
    text = re.sub(r"@\w+|#\w+", "", text)              # упоминания и хэштеги
    text = re.sub(r"[^\w\s]", " ", text)               # пунктуация
    text = re.sub(r"\d+", "", text)                    # цифры
    text = re.sub(r"\s+", " ", text).strip()           # лишние пробелы
    text = wordpunct_tokenize(text)
    text = [word for word in text if word not in stop_words]
    text = [morph.parse(word)[0].normal_form for word in text]
    return text

def accuracy(predictions, targets):
    classes = torch.argmax(predictions, dim=1)
    return torch.mean((classes == targets).float())

def train(model, loader, criterion, optimizer, num_epoch):
    model.train()
    all_accuracy = []
    all_loss = []
    for t in tqdm(range(num_epoch)):
        epoch_loss = []
        current_accuracy = 0
        current_loss = 0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(x_batch)
            y_batch = y_batch.long()
            loss = criterion(y_pred, y_batch)
            epoch_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            current_accuracy += accuracy(y_pred, y_batch)
            current_loss += loss.item()

        epoch_accuracy = current_accuracy / len(train_loader)
        epoch_loss = current_loss / len(train_loader)

        all_accuracy.append(epoch_accuracy)
        all_loss.append(epoch_loss)
        print('Epoch {} \t'.format(t), 'Accuracy: {}\t'.format(np.round(epoch_accuracy.item(), 6)),
              'Loss: {}'.format(np.round(epoch_loss, 6)))
    return model


if __name__ == '__main__':
    data = pd.read_csv('sentiment_dataset.csv')

    nltk.download('stopwords')
    stop_words = set(stopwords.words('russian'))
    morph = pymorphy2.MorphAnalyzer()

    data['text'] = data['text'].apply(preprocessing)

    tokenizer = Tokenizer(oov_token='<UNK>')
    tokenizer.fit_on_texts(data['text'])
    sequences = tokenizer.texts_to_sequences(data['text'])
    padded_sequences = pad_sequences(sequences, maxlen=90)

    x_train, x_test, y_train, y_test = train_test_split(padded_sequences, data['label'], test_size=0.4)

    x_train_tensor = torch.tensor(x_train, dtype=torch.long)
    x_test_tensor = torch.tensor(x_test, dtype=torch.long)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentimentAnalysis(len(tokenizer.word_index) + 1, 100, 32).to(device)

    train_data = TensorDataset(x_train_tensor, y_train_tensor)
    test_data = TensorDataset(x_test_tensor, y_test_tensor)

    batch_size = 128
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    train(model, train_loader, criterion, optimizer, 15)
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.to(device)
            outputs = model(data)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print('f1_score =', f1_score(all_labels, all_preds, average='macro'))
    
    # тестирование на собственных предложениях
    strings = [
        'Ужасное качество и плохой пошив. Возврат.',
        'Очень приятно было посетить данное место. Кормят вкусно и недорого.',
        'Неплохо, интересно, но затянуто. В целом, пойдет.',
        'Веселый, добрый и трогательный фильм, я даже заплакала.',
        'Песня растрогала, аплодисменты стоя.',
        'Сначала соседи громко шумели, затем обнаружилось отсутствие полотенец, напоследок на кухне нашли таракана. Не рекомендую.',
        'Положили лук хотя не просили, перепутали мясо. Больше не вернемся!',
        'Скучновато, зато зал украшен.',
        'После сеанса еще долго отходил и размышлял, тяжелый труд.'
    ]

    semantics = {
        0: 'нейтральный',
        1: 'положительный',
        2: 'отрицательный'
    }

    j = 0
    for i in strings:
        i = preprocessing(i)
        i = tokenizer.texts_to_sequences([i])
        i = pad_sequences(i, maxlen=90)
        i = torch.tensor(i)
        i = i.to(device)

        with torch.no_grad():
            logits = model(i)
        predicted_class = torch.argmax(logits, dim=1).item()
        print(strings[j])
        print('Предсказание: ', semantics[predicted_class])
        print()
        j += 1
    