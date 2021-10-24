import pandas as pd
import config
import torch


def load_ann():
    source = pd.read_csv("./data/ann.csv")
    length = len(source)
    width = source.shape[1]
    dataset_train = source.iloc[:int(length*config.train_percentage), :]
    dataset_test = source.iloc[int(length*config.train_percentage):, :]
    dataset_a = pd.DataFrame(columns=source.columns)
    dataset_u = pd.DataFrame(columns=source.columns)
    for i in range(len(dataset_train)):
        label = source.iloc[i, width-1]
        if label == config.ann_anomaly_class and len(dataset_a) < config.anomaly_num:
            dataset_a = dataset_a.append(dataset_train.iloc[i, :])
        else:
            dataset_u = dataset_u.append(dataset_train.iloc[i, :])
    dataset_a = dataset_a.reset_index(drop=True)
    dataset_u = dataset_u.reset_index(drop=True)

    a_in_u = 0
    for i in range(len(dataset_u)):
        if dataset_u.iloc[i, width-1] != 3:
            a_in_u = a_in_u + 1
    dataset_u_backup = dataset_u.copy(deep=True)
    for i in range(len(dataset_u)):
        if a_in_u / len(dataset_u) < config.contamination_rate:
            break
        if dataset_u.iloc[i, width-1] != 3:
            dataset_u_backup = dataset_u_backup.drop(i)
            a_in_u = a_in_u - 1
    dataset_u_backup.reset_index(drop=True)
    dataset_u = dataset_u_backup

    dataset_a = torch.tensor(dataset_a.values)[:, :-1].float().to(config.device)
    dataset_u = torch.tensor(dataset_u.values)[:, :-1].float().to(config.device)
    test_label = torch.tensor(dataset_test.values)[:, -1].float().to(config.device)
    test_label = [0 if i == 3 else 1 for i in test_label]
    dataset_test = torch.tensor(dataset_test.values)[:, :-1].float().to(config.device)

    return dataset_a, dataset_u, dataset_test, test_label


def load_har():
    source = pd.read_csv("./data/har.csv")
    width = source.shape[1]
    class_two = pd.DataFrame(columns=source.columns)
    class_three = pd.DataFrame(columns=source.columns)
    temp = pd.DataFrame(columns=source.columns)
    for i in range(len(source)):
        label = source.iloc[i, width - 1]
        if label == 2 and len(class_two) < 150:
            class_two = class_two.append(source.iloc[i, :])
        elif label == 3 and len(class_three) < 150:
            class_three = class_three.append(source.iloc[i, :])
        elif label == 1 or label == 4 or label == 5 or label == 6:
            temp = temp.append(source.iloc[i, :])
    source = pd.concat([temp, class_two, class_three])
    source = source.sample(frac=1).reset_index(drop=True)

    length = len(source)
    dataset_train = source.iloc[:int(length*config.train_percentage), :]
    dataset_test = source.iloc[int(length*config.train_percentage):, :]
    dataset_a = pd.DataFrame(columns=source.columns)
    dataset_u = pd.DataFrame(columns=source.columns)
    for i in range(len(dataset_train)):
        label = source.iloc[i, width-1]
        if label == config.har_anomaly_class and len(dataset_a) < config.anomaly_num:
            dataset_a = dataset_a.append(dataset_train.iloc[i, :])
        else:
            dataset_u = dataset_u.append(dataset_train.iloc[i, :])
    dataset_a = dataset_a.reset_index(drop=True)
    dataset_u = dataset_u.reset_index(drop=True)

    a_in_u = 0
    for i in range(len(dataset_u)):
        if dataset_u.iloc[i, width - 1] == 2 or dataset_u.iloc[i, width - 1] == 3:
            a_in_u = a_in_u + 1
    dataset_u_backup = dataset_u.copy(deep=True)
    for i in range(len(dataset_u)):
        if a_in_u / len(dataset_u) < config.contamination_rate:
            break
        if dataset_u.iloc[i, width - 1] == 2 or dataset_u.iloc[i, width - 1] == 3:
            dataset_u_backup = dataset_u_backup.drop(i)
            a_in_u = a_in_u - 1
    dataset_u_backup.reset_index(drop=True)
    dataset_u = dataset_u_backup

    dataset_a = torch.tensor(dataset_a.values)[:, :-1].float().to(config.device)
    dataset_u = torch.tensor(dataset_u.values)[:, :-1].float().to(config.device)
    test_label = torch.tensor(dataset_test.values)[:, -1].float().to(config.device)
    test_label = [1 if i == 2 or i == 3 else 0 for i in test_label]
    dataset_test = torch.tensor(dataset_test.values)[:, :-1].float().to(config.device)

    return dataset_a, dataset_u, dataset_test, test_label


def load_cov():
    source = pd.read_csv("./data/cov.csv")
    length = len(source)
    width = source.shape[1]
    dataset_train = source.iloc[:int(length*config.train_percentage), :]
    dataset_test = source.iloc[int(length*config.train_percentage):, :]
    dataset_a = pd.DataFrame(columns=source.columns)
    dataset_u = pd.DataFrame(columns=source.columns)
    for i in range(len(dataset_train)):
        label = source.iloc[i, width-1]
        if label == config.cov_anomaly_class and len(dataset_a) < config.anomaly_num:
            dataset_a = dataset_a.append(dataset_train.iloc[i, :])
        elif label == 2 or label == 4 or label == 6:
            dataset_u = dataset_u.append(dataset_train.iloc[i, :])
    dataset_a = dataset_a.reset_index(drop=True)
    dataset_u = dataset_u.reset_index(drop=True)

    a_in_u = 0
    for i in range(len(dataset_u)):
        if dataset_u.iloc[i, width - 1] == 4 or dataset_u.iloc[i, width - 1] == 6:
            a_in_u = a_in_u + 1
    dataset_u_backup = dataset_u.copy(deep=True)
    for i in range(len(dataset_u)):
        if a_in_u / len(dataset_u) < config.contamination_rate:
            break
        if dataset_u.iloc[i, width - 1] == 4 or dataset_u.iloc[i, width - 1] == 6:
            dataset_u_backup = dataset_u_backup.drop(i)
            a_in_u = a_in_u - 1
    dataset_u_backup.reset_index(drop=True)
    dataset_u = dataset_u_backup

    dataset_a = torch.tensor(dataset_a.values)[:, :-1].float().to(config.device)
    dataset_u = torch.tensor(dataset_u.values)[:, :-1].float().to(config.device)
    test_label = torch.tensor(dataset_test.values)[:, -1].float().to(config.device)
    test_label = [0 if i == 2 else 1 for i in test_label]
    dataset_test = torch.tensor(dataset_test.values)[:, :-1].float().to(config.device)

    return dataset_a, dataset_u, dataset_test, test_label
