import torch
import torch.nn as nn

class EMG_RCNN(nn.Module):
    def __init__(self, input_channels, hidden_size, output_dim):
        super(EMG_RCNN, self).__init__()

        # --- CNN Feature Extractor ---
        self.conv1 = nn.Conv1d(
            in_channels=input_channels,
            out_channels=32,
            kernel_size=5,
            padding=2
        )
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=5,
            padding=2
        )
        self.bn2 = nn.BatchNorm1d(64)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        # --- LSTM Temporal Modeling ---
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        # --- Regression Output ---
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        # x shape: (batch, seq_len, channels)

        # CNN expects (batch, channels, seq_len)
        x = x.permute(0, 2, 1)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        # Back to (batch, seq_len, features)
        x = x.permute(0, 2, 1)

        lstm_out, _ = self.lstm(x)

        # Take last time step
        last_out = lstm_out[:, -1, :]

        output = self.fc(last_out)

        return output

