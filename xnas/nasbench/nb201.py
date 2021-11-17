import sys
import os

from nas_201_api import NASBench201API as API201


def Eval_nasbench201(theta, search_space, logger):
    nasbench201_path = 'benchmark/NAS-Bench-201-v1_0-e61699.pth'
    api_nasben201 = API201(nasbench201_path, verbose=False)

    # get result log
    with open("temp_eval_nasbench201.out", "w") as log_file:
        stdout_backup = sys.stdout
        sys.stdout = log_file

        geotype = search_space.genotype(theta)
        index = api_nasben201.query_index_by_arch(geotype)
        api_nasben201.show(index)

        sys.stdout = stdout_backup

    with open("temp_eval_nasbench201.out", "r") as f:
        fls = f.readlines()
        for i in fls:
            logger.info(i[:-1])

    os.remove("temp_eval_nasbench201.out")
