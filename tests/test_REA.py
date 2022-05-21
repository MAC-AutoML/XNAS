from xnas.algorithms.SPOS import REA

rea = REA(num_choice=4, child_len=10, population_size=5)

for i in range(30):
    child = rea.suggest()
    print("suggest: {}".format(child))
    rea.record(child, value=sum(child))
    print("===")
    for p in rea.population:
        print(p)
    input()
print(rea.final_best())