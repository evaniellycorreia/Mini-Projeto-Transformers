import argparse
import random
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn
import matplotlib.pyplot as plt


DATASET_URL = (
    "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
    "jena_climate_2009_2016.csv.zip"
)

INPUT_LENGTH = 144
NEXT_TOKEN_LENGTH = 1
FORECAST_24H_LENGTH = 144


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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(dataset, window_step, seed, batch_size=64):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        Subset(dataset, range(0, len(dataset), window_step)),
        batch_size=batch_size,
        shuffle=True,
        generator=generator
    )


def train_model(model, loader, epochs, device, lr=0.001, max_grad_norm=1.0):
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
            # Limita a norma do gradiente para evitar divergência (NaN)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            samples_seen += batch_x.size(0)

        history.append(total_loss / samples_seen)
        print(f"Epoch {epoch + 1}/{epochs} - MSE: {history[-1]:.6f}")

        if not np.isfinite(history[-1]):
            raise RuntimeError(
                "Training diverged (NaN). Reduce lr in train_model."
            )

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
    plt.xticks(range(1, len(history) + 1))
    plt.xlabel("Epoch")
    plt.ylabel("Training MSE")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def get_attention(model, dataset, device, n_samples):
    model.eval()
    indices = np.linspace(0, len(dataset) - 1, n_samples, dtype=int)
    inputs = torch.stack([dataset[int(i)][0] for i in indices]).to(device)

    with torch.no_grad():
        model(inputs)

    # Formato: (amostras, cabeças, m, m)
    return model.encoder_blocks[-1].attention_weights


def plot_attention(model, dataset, device, path, head=0, n_samples=8):
    weights = get_attention(model, dataset, device, n_samples)[:, head]
    mean_weights = weights.mean(dim=0).cpu().numpy()

    plt.figure(figsize=(7, 6))
    plt.imshow(mean_weights, cmap="viridis", aspect="auto")
    plt.colorbar(label="Attention weight")
    plt.xlabel("Key (token)")
    plt.ylabel("Query (token)")
    plt.title(f"Mean attention - last block, head {head}")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_last_query_attention(model, dataset, device, path, n_samples=32):
    weights = get_attention(model, dataset, device, n_samples)[:, :, -1, :]
    mean_weights = weights.mean(dim=0).cpu().numpy()

    plt.figure(figsize=(9, 5))
    for head, values in enumerate(mean_weights):
        plt.plot(values, label=f"Head {head}")
    plt.xlabel("Key (token)")
    plt.ylabel("Attention weight")
    plt.title("Last-query attention per head (last block)")
    plt.legend()
    plt.grid(True)
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
    baseline = np.full_like(truth, history[-1])

    past = np.arange(-len(history), 0)
    future = np.arange(len(truth))

    plt.figure(figsize=(9, 5))
    plt.plot(past, history, label="Input (past 24 h)")
    plt.plot(future, truth, label="Actual (next 24 h)")
    plt.plot(future, forecast, "--", label="Forecast")
    plt.plot(future, baseline, ":", label="Baseline (last value)")
    plt.axvline(0, color="gray", linewidth=0.8)
    plt.xlabel("Time step (10 min)")
    plt.ylabel(feature_name)
    plt.title(f"24-hour forecast - {feature_name} (sample {sample_index})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def run_sweep(train_dataset, test_dataset, input_dim, device, configs,
              seed, epochs=3, window_step=48):
    results = []

    for config in configs:
        print(f"\nConfig: {config}")
        set_seed(seed)
        loader = make_loader(train_dataset, window_step, seed)
        model = TimeSeriesTransformer(
            input_dim=input_dim,
            input_length=INPUT_LENGTH,
            forecast_length=FORECAST_24H_LENGTH,
            **config
        ).to(device)

        train_model(model, loader, epochs, device)
        test_mse, base_mse = evaluate_model(model, test_dataset, device)
        results.append((config, test_mse, base_mse))

    print("\nSweep summary (24 h):")
    for config, test_mse, base_mse in results:
        print(f"{config} -> Test MSE {test_mse:.4f} (baseline {base_mse:.4f})")

    return results


def ensure_dataset(data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "jena_climate_2009_2016.csv"

    if not csv_path.is_file():
        zip_path = data_dir / "jena_climate_2009_2016.csv.zip"
        print("Downloading dataset...")
        urllib.request.urlretrieve(DATASET_URL, zip_path)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(data_dir)
        zip_path.unlink()

    return csv_path


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Transformer para o Jena Climate"
    )
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42],
        help="Uma ou mais sementes. Ex.: --seeds 1 2 3"
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--window-step", type=int, default=24)
    parser.add_argument(
        "--skip-sweep", action="store_true",
        help="Pula a varredura de hiperparâmetros"
    )
    return parser.parse_args()


def load_data(csv_path):
    df = pd.read_csv(csv_path)

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
    # A velocidade média não pode exceder a rajada máxima do mesmo intervalo
    df["wv (m/s)"] = df["wv (m/s)"].clip(upper=df["max. wv (m/s)"])

    rows_before_cleaning = len(df)
    df = df.dropna().reset_index(drop=True)

    print("\nRows removed:", rows_before_cleaning - len(df))
    print("Dimensions after cleaning:", df.shape)
    print("\nStatistical summary:")
    print(df.describe().T.round(2))

    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()

    print("\nChronological split:")
    print("Training rows:", len(train_df))
    print("Testing rows:", len(test_df))
    print("Training period:", train_df["Date Time"].min(),
          "to", train_df["Date Time"].max())
    print("Testing period:", test_df["Date Time"].min(),
          "to", test_df["Date Time"].max())

    feature_names = [c for c in df.columns if c != "Date Time"]

    scaler = StandardScaler()
    train_values = scaler.fit_transform(
        train_df[feature_names]
    ).astype(np.float32)
    test_values = scaler.transform(
        test_df[feature_names]
    ).astype(np.float32)

    return feature_names, scaler, train_values, test_values


def run_experiment(seed, args, device, feature_names, scaler,
                   datasets, save_dir, run_sweep_too):
    train_next, test_next, train_24h, test_24h = datasets
    save_dir.mkdir(parents=True, exist_ok=True)
    input_dim = len(feature_names)

    # Objetivo 1: próximo token
    set_seed(seed)
    model_next = TimeSeriesTransformer(
        input_dim, INPUT_LENGTH, NEXT_TOKEN_LENGTH
    ).to(device)
    loader = make_loader(train_next, args.window_step, seed)
    history = train_model(model_next, loader, args.epochs, device)
    plot_loss(history, "Next-token prediction", save_dir / "loss_next_token.png")

    next_mse, next_base = evaluate_model(model_next, test_next, device)
    print(f"[Next token] Test MSE: {next_mse:.4f} | Baseline: {next_base:.4f}")
    plot_attention(model_next, test_next, device,
                   save_dir / "attention_next_token.png")

    # Objetivo 2: próximas 24 h
    set_seed(seed)
    model_24h = TimeSeriesTransformer(
        input_dim, INPUT_LENGTH, FORECAST_24H_LENGTH
    ).to(device)
    loader = make_loader(train_24h, args.window_step, seed)
    history = train_model(model_24h, loader, args.epochs, device)
    plot_loss(history, "24-hour forecast", save_dir / "loss_24h.png")

    h24_mse, h24_base = evaluate_model(model_24h, test_24h, device)
    print(f"[24 h] Test MSE: {h24_mse:.4f} | Baseline: {h24_base:.4f}")
    plot_attention(model_24h, test_24h, device,
                   save_dir / "attention_24h.png")
    plot_last_query_attention(model_24h, test_24h, device,
                              save_dir / "attention_last_query_24h.png")

    model_mse, base_mse = evaluate_per_variable(model_24h, test_24h, device)
    print("\nPer-variable test MSE (24 h):")
    print_per_variable(feature_names, model_mse, base_mse)

    for number, sample_index in enumerate((1000, 20000, 60000), start=1):
        plot_forecast(
            model_24h, test_24h, device, scaler, feature_names,
            "T (degC)", save_dir / f"forecast_24h_temperature_{number}.png",
            sample_index=sample_index
        )

    if run_sweep_too:
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
        run_sweep(train_24h, test_24h, input_dim, device, configs, seed)

    return next_mse, next_base, h24_mse, h24_base


def main():
    
    print("Script:", Path(__file__).resolve())

    args = parse_args()
    device = torch.device("cpu")

    csv_path = ensure_dataset(args.data_dir)
    feature_names, scaler, train_values, test_values = load_data(csv_path)

    datasets = (
        SlidingWindowDataset(train_values, INPUT_LENGTH, NEXT_TOKEN_LENGTH),
        SlidingWindowDataset(test_values, INPUT_LENGTH, NEXT_TOKEN_LENGTH),
        SlidingWindowDataset(train_values, INPUT_LENGTH, FORECAST_24H_LENGTH),
        SlidingWindowDataset(test_values, INPUT_LENGTH, FORECAST_24H_LENGTH),
    )

    print("\nSliding-window shapes:")
    x_next, y_next = datasets[0][0]
    x_24h, y_24h = datasets[2][0]
    print("Input:", tuple(x_next.shape))
    print("Next-token target:", tuple(y_next.shape))
    print("24-hour target:", tuple(y_24h.shape))
    print("Training windows (next / 24 h):", len(datasets[0]), len(datasets[2]))
    print("Testing windows (next / 24 h):", len(datasets[1]), len(datasets[3]))

    all_results = []

    for position, seed in enumerate(args.seeds):
        print(f"\n===== Seed {seed} =====")
        result = run_experiment(
            seed, args, device, feature_names, scaler, datasets,
            args.output_dir / f"seed_{seed}",
            run_sweep_too=(position == 0 and not args.skip_sweep)
        )
        all_results.append(result)

    results = np.array(all_results)
    labels = ["Next token", "Next token baseline",
              "24 h", "24 h baseline"]

    print("\n===== Summary (test MSE) =====")
    print(f"Seeds: {args.seeds}")
    for column, label in enumerate(labels):
        values = results[:, column]
        std = values.std(ddof=1) if len(values) > 1 else 0.0
        print(f"{label:<22}{values.mean():.4f} +/- {std:.4f}")

    print("\nFigures saved to:", args.output_dir)


if __name__ == "__main__":
    main()