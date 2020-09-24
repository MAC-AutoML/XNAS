from xnas.core.plotting import graph_plot

if __name__ == "__main__":
    genotype = '|nor_conv_3x3~0|+|nor_conv_1x1~0|avg_pool_3x3~1|+|skip_connect~0|nor_conv_3x3~1|skip_connect~2|'
    file_path = './experiment/a.png'
    graph_plot(genotype, file_path, caption=None, search_space='nasbench_201', n_nodes=3)
