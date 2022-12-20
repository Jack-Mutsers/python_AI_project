
def log_training_settings(init_lr, bs, epochs, session, datasets, continuation, storage):
    content = [
        "continuation: " + str(continuation),
        "",
        "initial training speed: " + str(init_lr),
        "batch size: " + str(bs),
        "epochs: " + str(epochs),
        "sessions: " + str(session),
        "",
        "dataset:",
    ]

    for dataset in datasets:
        name = dataset.replace("/", "\\")
        name = name.split("\\")[-1]
        content.append("- " + name)

    with open(storage + "training_settings.txt", 'w') as f:
        for row in content:
            f.write(row + '\n')

# continuation of: 2022-12-19_20-19-29/handwriting_perfect_letters_v2-2x50.model

# initial training speed: 8e-3 (0.008)
# batch size: 800
# epochs: 50
# sessions: 2

# dataset:
# - emnist_byclass_train_trimmed_letters_only
# - emnist_letters_train_trimmed_letters_only
# - letter_e
# - perfect_letters_sm (x30)
# - perfect_joined_letters_sm (x30)
# - typed_letters_sm (x30)
