print("\nChecagem temporal da primeira janela:")
print(
    "Entrada:",
    train_df["Date Time"].iloc[0],
    "até",
    train_df["Date Time"].iloc[input_length - 1]
)
print(
    "Alvo:",
    train_df["Date Time"].iloc[input_length],
    "até",
    train_df["Date Time"].iloc[input_length + forecast_length - 1]
)
print(
    "Intervalo entre observações:",
    train_df["Date Time"].iloc[1] - train_df["Date Time"].iloc[0]
)