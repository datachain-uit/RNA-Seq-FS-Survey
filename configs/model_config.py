MODEL_PARAMS = {
    "MLP": {
        "hidden1": 128,
        "hidden2": 64,
        "dropout": 0.3
    },
    "LSTM": {
        "hidden_dim": 64
    },
    "GRU": {
        "hidden_dim": 64
    },
    "CNN1D": {
        "num_filters": [32, 64],
        "kernel_size": 3
    },
    "Transformer": {
        "embed_dim": 128,
        "nhead": 4,
        "num_layers": 2
    },
    "VAE": {
        "latent_dim": 32
    },
    "GAN": {
        "latent_dim": 64
    }
}
