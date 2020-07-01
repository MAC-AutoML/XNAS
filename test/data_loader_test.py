from xnas.datasets.imagenet import XNAS_ImageFolder


if __name__ == "__main__":
    for backend in ['torch', 'dali_cpu', 'dali_gpu', 'custom']:
        print('Testing the dataloader with backend: {}'.format(backend))
        dataset = XNAS_ImageFolder('/gdata/Caltech256/256_ObjectCategories',
                                   [0.8, 0.2],
                                   backend=backend,
                                   dataset_name='custom')
        [train_, val_] = dataset.generate_data_loader()

        for i, (inputs, labels) in enumerate(train_):
            inputs = inputs.cuda()
            labels = labels.cuda()
            print(i)
        for i, (inputs, labels) in enumerate(val_):
            inputs = inputs.cuda()
            labels = labels.cuda()
            print(i)
        print('testing passed')
