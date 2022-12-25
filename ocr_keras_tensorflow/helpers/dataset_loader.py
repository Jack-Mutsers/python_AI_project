import pyimagesearch.dataset.helpers as helpers
import numpy as np

loaded_datasets = []

def load_dataset(load_mnist, load_AZ, datasets_to_load):
    data = np.empty([0,28,28], "uint8")
    labels = np.empty([0,])

    # load the A-Z and MNIST datasets, respectively
    print("[INFO] loading datasets...")
    if load_mnist:
        (digitsData, digitsLabels) = helpers.load_mnist_dataset()

        loaded_datasets.append("mnist_dataset")
        data = np.vstack([data, digitsData])
        labels = np.hstack([labels, digitsLabels])


    if len(load_AZ) > 0 and load_AZ["amount"] > 0:
        (azData, azLabels) = helpers.load_az_dataset(load_AZ["location"])

        # the MNIST dataset occupies the labels 0-9, so let's add 10 to every
        # A-Z label to ensure the A-Z characters are not incorrectly labeled as digits
        azLabels += load_AZ["offset"]

        loaded_datasets.append(load_AZ["location"])
        data = np.vstack([data, azData])
        labels = np.hstack([labels, azLabels])


    for dataset in datasets_to_load:
        if dataset["amount"] < 1:
            continue

        addition = ""
        if dataset["amount"] > 1:
            addition = " (x" + str(dataset["amount"]) + ")"

        loaded_datasets.append(dataset["location"] + addition)
        (loadedData, loadedLabels) = helpers.load_az_dataset(dataset["location"], flipped=dataset["flipped"])

        loadedLabels += dataset["offset"]

        for _ in range(0, dataset["amount"]):
            data = np.vstack([data, loadedData])
            labels = np.hstack([labels, loadedLabels])

    print("[INFO] datasets loaded.")

    return data, labels, loaded_datasets
