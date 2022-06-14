from xnas.evaluations.NASBench201 import evaluate, index_to_genotype, distill

arch = index_to_genotype(2333)
result = evaluate(arch)
(
    cifar10_train,
    cifar10_test,
    cifar100_train,
    cifar100_valid,
    cifar100_test,
    imagenet16_train,
    imagenet16_valid,
    imagenet16_test,
) = distill(result)

print("cifar10 train %f test %f", cifar10_train, cifar10_test)
print("cifar100 train %f valid %f test %f", cifar100_train, cifar100_valid, cifar100_test)
print("imagenet16 train %f valid %f test %f", imagenet16_train, imagenet16_valid, imagenet16_test)


result = evaluate(arch, epoch=200)
(
    cifar10_train,
    cifar10_test,
    cifar100_train,
    cifar100_valid,
    cifar100_test,
    imagenet16_train,
    imagenet16_valid,
    imagenet16_test,
) = distill(result)

print("cifar10 train %f test %f", cifar10_train, cifar10_test)
print("cifar100 train %f valid %f test %f", cifar100_train, cifar100_valid, cifar100_test)
print("imagenet16 train %f valid %f test %f", imagenet16_train, imagenet16_valid, imagenet16_test)
