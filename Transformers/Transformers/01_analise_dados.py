from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn
import matplotlib.pyplot as plt


class SlidingWindowDataset(Dataset):
    def __init__(self, values, input_length, forecast_length):
        self.values = values
        self.input_length = input_length
        self.forecast_length = forecast_length
        self.window_count = max(
            0,
            len(values) - input_length - forecast_length + 1
        )

    def __len__(self):
        return self.window_count

    def __getitem__(self, index):
        if index < 0 or index >= self.window_count:
            raise IndexError("Window index out of range")

        input_end = index + self.input_length
        target_end = input_end + self.forecast_length

        x = torch.from_numpy(self.values[index:input_end])
        y = torch.from_numpy(self.values[input_end:target_end])

        return x, y


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len):
        super().__init__()

        positions = torch.arange(
            max_len, dtype=torch.float32
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32)
            * (-np.log(10000.0) / embed_dim)
        )

        encoding = torch.zeros(max_len, embed_dim)
        encoding[:, 0::2] = torch.sin(positions * div_term)
        encoding[:, 1::2] = torch.cos(positions * div_term)

        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, x):
        return x + self.encoding[:, :x.size(1), :]


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None

    def forward(self, x):
        attention_output, attention_weights = self.attention(
            x,
            x,
            x,
            need_weights=True,
            average_attn_weights=False
        )
        self.attention_weights = attention_weights

        x = self.norm1(x + self.dropout(attention_output))
        x = self.norm2(x + self.dropout(self.feedforward(x)))
        return x


class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        input_length,
        forecast_length,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        token_size=1
    ):
        super().__init__()

        self.input_dim = input_dim
        self.forecast_length = forecast_length
        self.token_size = token_size

        self.embedding = nn.Linear(input_dim * token_size, embed_dim)
        self.positional_encoding = PositionalEncoding(
            embed_dim, input_length // token_size
        )
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(
            embed_dim, forecast_length * input_dim
        )

    def forward(self, x):
        # (B, L, F) -> (B, L / token_size, token_size * F)
        x = x.reshape(x.size(0), -1, self.token_size * self.input_dim)
        x = self.embedding(x)
        x = self.positional_encoding(x)

        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)

        last_token = x[:, -1, :]

        predictions = self.output_layer(last_token)

        return predictions.reshape(
            x.size(0),
            self.forecast_length,
            self.input_dim
        )


def train_model(model, loader, epochs, device, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        samples_seen = 0

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            samples_seen += batch_x.size(0)

        history.append(total_loss / samples_seen)
        print(f"Epoch {epoch + 1}/{epochs} - MSE: {history[-1]:.6f}")

    return history


def evaluate_model(model, dataset, device, step=12):
    loader = DataLoader(
        Subset(dataset, range(0, len(dataset), step)),
        batch_size=128
    )
    model.eval()
    model_error = 0.0
    baseline_error = 0.0
    samples_seen = 0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            predictions = model(batch_x)
            # Baseline: repete o último token observado
            baseline = batch_x[:, -1:, :].expand_as(batch_y)

            model_error += ((predictions - batch_y) ** 2).mean().item() * batch_x.size(0)
            baseline_error += ((baseline - batch_y) ** 2).mean().item() * batch_x.size(0)
            samples_seen += batch_x.size(0)

    return model_error / samples_seen, baseline_error / samples_seen


def evaluate_per_variable(model, dataset, device, step=12):
    loader = DataLoader(
        Subset(dataset, range(0, len(dataset), step)),
        batch_size=128
    )
    model.eval()
    model_sq = 0.0
    base_sq = 0.0
    count = 0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            predictions = model(batch_x)
            baseline = batch_x[:, -1:, :].expand_as(batch_y)

            model_sq += ((predictions - batch_y) ** 2).sum(dim=(0, 1))
            base_sq += ((baseline - batch_y) ** 2).sum(dim=(0, 1))
            count += batch_y.size(0) * batch_y.size(1)

    return (model_sq / count).cpu().numpy(), (base_sq / count).cpu().numpy()


def print_per_variable(feature_names, model_mse, base_mse):
    print(f"{'Variable':<18}{'Model':>10}{'Baseline':>10}")
    for name, m, b in zip(feature_names, model_mse, base_mse):
        print(f"{name:<18}{m:>10.4f}{b:>10.4f}")


def plot_loss(history, title, path):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(history) + 1), history, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training MSE")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_attention(model, dataset, device, path, head=0, n_samples=8):
    model.eval()
    indices = np.linspace(0, len(dataset) - 1, n_samples, dtype=int)
    inputs = torch.stack([dataset[int(i)][0] for i in indices]).to(device)

    with torch.no_grad():
        model(inputs)

    # Formato: (amostras, cabeças, m, m) -> média sobre as amostras
    weights = model.encoder_blocks[-1].attention_weights[:, head]
    mean_weights = weights.mean(dim=0).cpu().numpy()

    plt.figure(figsize=(7, 6))
    plt.imshow(mean_weights, cmap="viridis", aspect="auto")
    plt.colorbar(label="Attention weight")
    plt.xlabel("Key (time step)")
    plt.ylabel("Query (time step)")
    plt.title(f"Mean attention - last block, head {head}")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_forecast(model, dataset, device, scaler, feature_names,
                  feature_name, path, sample_index=0):
    idx = feature_names.index(feature_name)
    x, y = dataset[sample_index]

    model.eval()
    with torch.no_grad():
        pred = model(x.unsqueeze(0).to(device))[0].cpu().numpy()

    mean, scale = scaler.mean_[idx], scaler.scale_[idx]
    history = x[:, idx].numpy() * scale + mean
    truth = y[:, idx].numpy() * scale + mean
    forecast = pred[:, idx] * scale + mean

    past = np.arange(-len(history), 0)
    future = np.arange(len(truth))

    plt.figure(figsize=(9, 5))
    plt.plot(past, history, label="Input (past 24 h)")
    plt.plot(future, truth, label="Actual (next 24 h)")
    plt.plot(future, forecast, "--", label="Forecast")
    plt.axvline(0, color="gray", linewidth=0.8)
    plt.xlabel("Time step (10 min)")
    plt.ylabel(feature_name)
    plt.title(f"24-hour forecast - {feature_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def run_sweep(train_dataset, test_dataset, input_dim, input_length,
              forecast_length, device, configs, epochs=3, window_step=48):
    loader = DataLoader(
        Subset(train_dataset, range(0, len(train_dataset), window_step)),
        batch_size=64,
        shuffle=True
    )
    results = []

    for config in configs:
        print(f"\nConfig: {config}")
        model = TimeSeriesTransformer(
            input_dim=input_dim,
            input_length=input_length,
            forecast_length=forecast_length,
            **config
        ).to(device)

        train_model(model, loader, epochs, device)
        test_mse, base_mse = evaluate_model(model, test_dataset, device)
        results.append((config, test_mse, base_mse))

    print("\nSweep summary (24 h):")
    for config, test_mse, base_mse in results:
        print(f"{config} -> Test MSE {test_mse:.4f} (baseline {base_mse:.4f})")

    return results


def main():
    dataset_path = Path(
        r"C:\Projetos\Projeto_Transformers\Dataset\jena_climate_2009_2016.csv"
    )

    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    df = pd.read_csv(dataset_path)

    df["Date Time"] = pd.to_datetime(
        df["Date Time"],
        format="%d.%m.%Y %H:%M:%S"
    )

    print("Original dimensions:", df.shape)
    print("First date:", df["Date Time"].min())
    print("Last date:", df["Date Time"].max())

    wind_columns = ["wv (m/s)", "max. wv (m/s)"]

    for column in wind_columns:
        invalid_count = (df[column] == -9999.0).sum()
        print(f"Invalid values in {column}: {invalid_count}")
        df[column] = df[column].replace(-9999.0, np.nan)

    df[wind_columns] = df[wind_columns].interpolate(method="linear")

    rows_before_cleaning = len(df)
    df = df.dropna().reset_index(drop=True)

    print("\nRows removed:", rows_before_cleaning - len(df))
    print("Dimensions after cleaning:", df.shape)
    print("\nStatistical summary:")
    print(df.describe().T.round(2))

    train_fraction = 0.8
    train_size = int(len(df) * train_fraction)

    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()

    print("\nChronological split:")
    print("Training rows:", len(train_df))
    print("Testing rows:", len(test_df))
    print(
        "Training period:",
        train_df["Date Time"].min(),
        "to",
        train_df["Date Time"].max()
    )
    print(
        "Testing period:",
        test_df["Date Time"].min(),
        "to",
        test_df["Date Time"].max()
    )

    feature_names = [
        column for column in df.columns
        if column != "Date Time"
    ]

    scaler = StandardScaler()
    train_values = scaler.fit_transform(
        train_df[feature_names]
    ).astype(np.float32)
    test_values = scaler.transform(
        test_df[feature_names]
    ).astype(np.float32)

    input_length = 144
    next_token_length = 1
    forecast_24h_length = 144

    train_next = SlidingWindowDataset(
        train_values, input_length, next_token_length
    )
    test_next = SlidingWindowDataset(
        test_values, input_length, next_token_length
    )

    train_24h = SlidingWindowDataset(
        train_values, input_length, forecast_24h_length
    )
    test_24h = SlidingWindowDataset(
        test_values, input_length, forecast_24h_length
    )

    x_next, y_next = train_next[0]
    x_24h, y_24h = train_24h[0]

    print("\nSliding-window shapes:")
    print("Input:", tuple(x_next.shape))
    print("Next-token target:", tuple(y_next.shape))
    print("24-hour target:", tuple(y_24h.shape))
    print("Training windows for next token:", len(train_next))
    print("Training windows for 24 hours:", len(train_24h))
    print("Testing windows for next token:", len(test_next))
    print("Testing windows for 24 horas:", len(test_24h))

    model_next = TimeSeriesTransformer(
        input_dim=len(feature_names),
        input_length=input_length,
        forecast_length=next_token_length
    )

    model_24h = TimeSeriesTransformer(
        input_dim=len(feature_names),
        input_length=input_length,
        forecast_length=forecast_24h_length
    )

    device = torch.device("cpu")
    output_dir = Path(__file__).resolve().parent
    window_step = 24
    epochs = 5

    # Objetivo 1: próximo token
    loader_next = DataLoader(
        Subset(train_next, range(0, len(train_next), window_step)),
        batch_size=64,
        shuffle=True
    )
    model_next = model_next.to(device)
    history_next = train_model(model_next, loader_next, epochs, device)
    plot_loss(history_next, "Next-token prediction", output_dir / "loss_next_token.png")

    test_mse, base_mse = evaluate_model(model_next, test_next, device)
    print(f"[Next token] Test MSE: {test_mse:.4f} | Baseline: {base_mse:.4f}")
    plot_attention(model_next, test_next, device, output_dir / "attention_next_token.png")

    # Objetivo 2: próximas 24 h
    loader_24h = DataLoader(
        Subset(train_24h, range(0, len(train_24h), window_step)),
        batch_size=64,
        shuffle=True
    )
    model_24h = model_24h.to(device)
    history_24h = train_model(model_24h, loader_24h, epochs, device)
    plot_loss(history_24h, "24-hour forecast", output_dir / "loss_24h.png")

    test_mse, base_mse = evaluate_model(model_24h, test_24h, device)
    print(f"[24 h] Test MSE: {test_mse:.4f} | Baseline: {base_mse:.4f}")
    plot_attention(model_24h, test_24h, device, output_dir / "attention_24h.png")

    model_mse, base_var_mse = evaluate_per_variable(model_24h, test_24h, device)
    print("\nPer-variable test MSE (24 h):")
    print_per_variable(feature_names, model_mse, base_var_mse)

    plot_forecast(
        model_24h, test_24h, device, scaler, feature_names,
        "T (degC)", output_dir / "forecast_24h_temperature.png",
        sample_index=1000
    )

    # Varredura de hiperparâmetros (desative com RUN_SWEEP = False)
    RUN_SWEEP = True
    if RUN_SWEEP:
        configs = [
            {"token_size": 1,  "embed_dim": 64,  "num_heads": 4, "num_layers": 2},
            {"token_size": 6,  "embed_dim": 64,  "num_heads": 4, "num_layers": 2},
            {"token_size": 12, "embed_dim": 64,  "num_heads": 4, "num_layers": 2},
            {"token_size": 6,  "embed_dim": 32,  "num_heads": 4, "num_layers": 2},
            {"token_size": 6,  "embed_dim": 128, "num_heads": 4, "num_layers": 2},
            {"token_size": 6,  "embed_dim": 64,  "num_heads": 2, "num_layers": 2},
            {"token_size": 6,  "embed_dim": 64,  "num_heads": 8, "num_layers": 2},
            {"token_size": 6,  "embed_dim": 64,  "num_heads": 4, "num_layers": 1},
            {"token_size": 6,  "embed_dim": 64,  "num_heads": 4, "num_layers": 4},
        ]
        run_sweep(
            train_24h, test_24h, len(feature_names), input_length,
            forecast_24h_length, device, configs
        )

    print("Figures saved to:", output_dir)


if __name__ == "__main__":
    main()