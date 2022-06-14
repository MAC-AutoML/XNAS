import json

with open('xnas/evaluations/NASBenchmacro/nas-bench-macro_cifar10.json', 'r+', encoding='utf-8') as data_file:
    data = json.load(data_file)


def evaluate(arch):
    # print(data[arch])
    print("{} accuracy for three test on CIFAR10: {}".format(arch, data[arch]['test_acc']))
    print("mean accuracy : {}, std : {},  params : {}, flops : {}".format(data[arch]['mean_acc'], data[arch]['std'],
                                                                          data[arch]['params'], data[arch]['flops']))
