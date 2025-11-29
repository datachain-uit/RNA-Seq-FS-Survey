DL_TRAINING_CONFIG = {
    "batch_size": 64,
    "epochs": 50,
    "learning_rate": 1e-3,
    "optimizer": "adam",
    "device": "cuda"
}

DL_DATASET_CONFIG = {
    "train_file": "X_train.csv",
    "val_file": "X_val.csv",
    "test_file": "X_test.csv",
    "train_labels": "y_train.csv",
    "val_labels": "y_val.csv",
    "test_labels": "y_test.csv",
}
