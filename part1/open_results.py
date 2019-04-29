def open_results(model):
    results = {}
    headers = []
    accuracies = []
    n_epochs = []

    with open('results_{}.csv'.format(model)) as csvfile:
        reader = csvfile.readlines()
        for line in reader:
            values = line.split(',')
            header = values[0]
            values = values[1:]
            values = [float(value.replace('[', '').replace(']', '')) for value in values]
            results[header] = values
            if header.startswith('acc'):
                accuracy = values[-1]
                accuracies.append(accuracy)
                n_epoch = len(values)
                n_epochs.append(n_epoch)
            headers.append(header)

    T = [int(header.split(' ')[-1]) for header in headers]
    T = set(T)
    T = sorted(list(T))

    return results, T, accuracies, n_epochs
