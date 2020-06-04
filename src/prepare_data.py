import csv

def main():
    i = 1
    FIELDNAMES = ['Number of Convolutional layers', 'Number of filters', 'Filter dimension', 'Strides', 'Number of Recurrent layers', 'Number of Neurons', 'Batch Normalization', 'Activation Function', 'Accuracy']

    CovLayers = ['1', '2', '3']  # Number of Convolutional layers
    Filters = ['32', '32, 32', '32, 32, 96']  # Number of filters
    FilterDimension = ['41x11', '41x11, 21x11', '41x11, 21x11, 21x11']  # Filter dimension
    Strides = ['2x2', '2x2, 2x1', '2x2, 2x1, 2x1']  # Strides

    RecurrentLayers = ['1', '3', '5', '7']  # Number of Recurrent layers
    Neurons = ['1024', '1536', '2048']  # ['1024', '1280', '1536', '1792', '2048']    # Number of Neurons
    BatchNorm = [True, False]  # Batch Normalization
    #Dropout = ['10%', '15%', '20%']  # Dropout
    # RNNCell = ['LSTM', 'GRU']  # RNN Cell
    ActivationFunction = ['Tanh', 'Relu']  # Activation Function
    # LearningRate = ['0.0001']  # Learning rate
    # SortaGrad = ['True', 'False']  # SortaGrad
    Accuracy = ['1']  # Accuracy

    rows = []

    with open('MetaKnowledge.csv', 'w') as target_csv_file:
        writer = csv.DictWriter(target_csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()

        for covLayers_ in CovLayers:
            for recurrentLayers_ in RecurrentLayers:
                for neurons_ in Neurons:
                    for batchNorm_ in BatchNorm:
                        # for dropout_ in Dropout:
                        #     for rnnCell_ in RNNCell:
                            for activationFunction_ in ActivationFunction:
                                for accuracy_ in Accuracy:
                                    #rows.append((covLayers_, filters_, filterDimension_, strides_))
                                    # for filename, file_size, transcript in rows:
                                    writer.writerow({'Number of Convolutional layers': covLayers_, 'Number of filters': Filters[int(covLayers_) - 1], 'Filter dimension': FilterDimension[int(covLayers_) - 1],
                                                     'Strides': Strides[int(covLayers_) - 1], 'Number of Recurrent layers': recurrentLayers_, 'Number of Neurons': neurons_, 'Batch Normalization': batchNorm_,
                                                     # 'Dropout': dropout_, 'RNN Cell': rnnCell_,
                                                    'Activation Function': activationFunction_, 'Accuracy': accuracy_})

if __name__ == '__main__':
    main()

