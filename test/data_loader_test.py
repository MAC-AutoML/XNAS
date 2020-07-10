from xnas.datasets.imagenet import XNAS_ImageFolder
from xnas.datasets.cifar10 import XNAS_Cifar10


def image_folder_test():
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
            print(inputs)
            print(labels)
            break
        for i, (inputs, labels) in enumerate(val_):
            inputs = inputs.cuda()
            labels = labels.cuda()
            print(inputs)
            print(labels)
            break
        print('testing passed')


def cifar10_test():
    from xnas.core.config import cfg
    cfg.TRAIN.IM_SIZE = 32
    [train_, val_] = XNAS_Cifar10('/gdata/cifar10/cifar-10-batches-py', [0.8, 0.2])
    for i, (inputs, labels) in enumerate(train_):
        inputs = inputs.cuda()
        labels = labels.cuda()
        print(inputs)
        print(labels)
        break
    for i, (inputs, labels) in enumerate(val_):
        inputs = inputs.cuda()
        labels = labels.cuda()
        print(inputs)
        print(labels)
        break
    print('testing passed')


if __name__ == "__main__":
    cifar10_test()
    pass
