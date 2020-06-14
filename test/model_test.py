from model.mb_v3_cnn import get_super_net
if __name__ == '__main__':
    search_space = 'ofa'
    width_mult = 1.3
    conv_candidates = [
                '3x3_MBConv3', '3x3_MBConv6',
                '5x5_MBConv3', '5x5_MBConv6',
                '7x7_MBConv3', '7x7_MBConv6',
            ]
    depth = 4
    model = get_super_net(n_classes=1000, base_stage_width=search_space,
                                  width_mult=width_mult, conv_candidates=conv_candidates,
                                  depth=depth)
    a = model.config
    pass

