import os, sys
try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve


def download_url(url, model_dir='~/.torch/', overwrite=False):
    target_dir = url.split('/')[-1]
    model_dir = os.path.expanduser(model_dir)
    model_dir = os.path.join(model_dir, target_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    cached_file = model_dir
    sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
    for j in ['net.config', 'run.config']:
        if os.path.isfile(os.path.join(cached_file, j)):
            pass
        else:
            _url = url + '/' + j
            urlretrieve(url + '/' + j, os.path.join(cached_file, j))
    return cached_file


if __name__ == '__main__':
    specialized_network_list = [
        ################# FLOPs #################
        'flops@595M_top1@80.0_finetune@75',
        'flops@482M_top1@79.6_finetune@75',
        'flops@389M_top1@79.1_finetune@75',
        ################# Google pixel1 #################
        'pixel1_lat@143ms_top1@80.1_finetune@75',
        'pixel1_lat@132ms_top1@79.8_finetune@75',
        'pixel1_lat@79ms_top1@78.7_finetune@75',
        'pixel1_lat@58ms_top1@76.9_finetune@75',
        'pixel1_lat@40ms_top1@74.9_finetune@25',
        'pixel1_lat@28ms_top1@73.3_finetune@25',
        'pixel1_lat@20ms_top1@71.4_finetune@25',
        ################# Google pixel2 #################
        'pixel2_lat@62ms_top1@75.8_finetune@25',
        'pixel2_lat@50ms_top1@74.7_finetune@25',
        'pixel2_lat@35ms_top1@73.4_finetune@25',
        'pixel2_lat@25ms_top1@71.5_finetune@25',
        ################# Samsung note10 #################
        'note10_lat@64ms_top1@80.2_finetune@75',
        'note10_lat@50ms_top1@79.7_finetune@75',
        'note10_lat@41ms_top1@79.3_finetune@75',
        'note10_lat@30ms_top1@78.4_finetune@75',
        'note10_lat@22ms_top1@76.6_finetune@25',
        'note10_lat@16ms_top1@75.5_finetune@25',
        'note10_lat@11ms_top1@73.6_finetune@25',
        'note10_lat@8ms_top1@71.4_finetune@25',
        ################# Samsung note8 #################
        'note8_lat@65ms_top1@76.1_finetune@25',
        'note8_lat@49ms_top1@74.9_finetune@25',
        'note8_lat@31ms_top1@72.8_finetune@25',
        'note8_lat@22ms_top1@70.4_finetune@25',
        ################# Samsung S7 Edge #################
        's7edge_lat@88ms_top1@76.3_finetune@25',
        's7edge_lat@58ms_top1@74.7_finetune@25',
        's7edge_lat@41ms_top1@73.1_finetune@25',
        's7edge_lat@29ms_top1@70.5_finetune@25',
        ################# LG G8 #################
        'LG-G8_lat@24ms_top1@76.4_finetune@25',
        'LG-G8_lat@16ms_top1@74.7_finetune@25',
        'LG-G8_lat@11ms_top1@73.0_finetune@25',
        'LG-G8_lat@8ms_top1@71.1_finetune@25',
        ################# 1080ti GPU (Batch Size 64) #################
        '1080ti_gpu64@27ms_top1@76.4_finetune@25',
        '1080ti_gpu64@22ms_top1@75.3_finetune@25',
        '1080ti_gpu64@15ms_top1@73.8_finetune@25',
        '1080ti_gpu64@12ms_top1@72.6_finetune@25',
        ################# V100 GPU (Batch Size 64) #################
        'v100_gpu64@11ms_top1@76.1_finetune@25',
        'v100_gpu64@9ms_top1@75.3_finetune@25',
        'v100_gpu64@6ms_top1@73.0_finetune@25',
        'v100_gpu64@5ms_top1@71.6_finetune@25',
        ################# Jetson TX2 GPU (Batch Size 16) #################
        'tx2_gpu16@96ms_top1@75.8_finetune@25',
        'tx2_gpu16@80ms_top1@75.4_finetune@25',
        'tx2_gpu16@47ms_top1@72.9_finetune@25',
        'tx2_gpu16@35ms_top1@70.3_finetune@25',
        ################# Intel Xeon CPU with MKL-DNN (Batch Size 1) #################
        'cpu_lat@17ms_top1@75.7_finetune@25',
        'cpu_lat@15ms_top1@74.6_finetune@25',
        'cpu_lat@11ms_top1@72.0_finetune@25',
        'cpu_lat@10ms_top1@71.1_finetune@25',
    ]
    for i in specialized_network_list:
        download_url('https://hanlab.mit.edu/files/OnceForAll/ofa_specialized/'+i, '/Users/sherwood/Downloads/models')
