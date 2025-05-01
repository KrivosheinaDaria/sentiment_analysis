import torch
import torch.nn as nn

class SentimentAnalysis(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(SentimentAnalysis, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim,
                          hidden_dim,
                          batch_first=True,
                          bidirectional=True)
        self.dropout = nn.Dropout(0.7)
        self.fc = nn.Linear(hidden_dim*2, 3)

    def forward(self, x):
        x = self.embedding(x)
        _, (hn, _) = self.lstm(x)
        hn_combined = torch.cat((hn[-2], hn[-1]), dim=1)
        x = self.fc(self.dropout(hn_combined))
        return x