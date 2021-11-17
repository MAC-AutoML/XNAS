from xnas.core.plotting import graph_plot

if __name__ == "__main__":
    genotype = '|conv_3x3~0|+|conv_1x1~0|conv_1x1~1|+|none~0|conv_1x1~1|conv_1x1~2|+|conv_3x3~0|conv_3x3~1|none~2|none~3|'
    file_path = './experiment/architecture_plot_test/img'
    graph_plot(genotype, file_path, caption=None, search_space='nasbench_201', n_nodes=4)
    print("architecture plot test pass.")
