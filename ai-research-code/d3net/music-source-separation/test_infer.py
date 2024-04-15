import os
import yaml
import nnabla as nn
import numpy as np
from util import stft2time_domain, model_separate
from model import stft, spectogram, D3NetMSS

if __name__ == "__main__":
    # model_path_1 = "/ai-research-code/d3net/music-source-separation/assets/vocals_original.h5"
    # model_path_2 = "/Users/daniellebenbashat/PycharmProjects/audio/ai-research-code/d3net/music-source-separation/assets/vocals_batch15768.h5"
    model_path_1 = "/Users/daniellebenbashat/PycharmProjects/audio/ai-research-code/d3net/music-source-separation/assets/vocals_run2_epoch0.h5"
    model_path_2 = "/Users/daniellebenbashat/PycharmProjects/audio/ai-research-code/d3net/music-source-separation/assets/vocals_run2_epoch10.h5"
    model_path_3 = "/Users/daniellebenbashat/PycharmProjects/audio/ai-research-code/d3net/music-source-separation/assets/vocals_epoch42_run3.h5"
    nn.clear_parameters()
    nn.load_parameters(model_path_3)
    with nn.parameter_scope("XXX"):
        x=nn.get_current_parameter_scope()
        params3 = nn.get_parameters()

    nn.clear_parameters()
    nn.load_parameters(model_path_1)
    params1 = nn.get_parameters()

    nn.clear_parameters()
    nn.load_parameters(model_path_2)
    params2 = nn.get_parameters()

    targets1 = ['vocals']
    targets2 = ['VocalSourceSeparation/ai-research-code/d3net/music-source-separation/configs/vocals']
    targets = [targets1, targets2]

    # with nn.parameter_scope('VocalSourceSeparation/ai-research-code/d3net/music-source-separation/configs/VocalSourceSeparation/ai-research-code/d3net/music-source-separation/configs/'):
    nn.clear_parameters()
    with nn.parameter_scope(targets2[0]):
        nn.load_parameters(model_path_1)

    with nn.parameter_scope(targets2[0]):
        paramstest = nn.get_parameters()
    # diff_cnt, same_cnt = 0, 0
    diff_params = []
    same_params = []
    for key1, key2 in zip(params1.keys(), params2.keys()):
        if np.allclose(params1[key1].d, params2[key2].d):
            # same_cnt += 1
            # print(f"Layers {key1}, {key2} WITH the same params")
            # print(f"Layers {key1}, {key2} WITH the same params")
            same_params.append([key1, key2])
        else:
            # print(f"Layers {key1}, {key2} without the same params")
            print(f"{np.sum(np.abs(params1[key1].d-params2[key2].d))}, Layers {key1}, {key2} without the same params")
            # diff_cnt += 1
            diff_params.append([key1, key2])


    for key1, key3 in zip(params1.keys(), params3.keys()):
        if np.allclose(params1[key1].d, params3[key3].d):
            # same_cnt += 1
            # print(f"Layers {key1}, {key2} WITH the same params")
            # print(f"Layers {key1}, {key2} WITH the same params")
            same_params.append([key1, key3])
        else:
            # print(f"Layers {key1}, {key2} without the same params")
            print(f"{np.sum(np.abs(params1[key1].d-params3[key3].d))}, Layers {key1}, {key3} without the same params")
            # diff_cnt += 1
            diff_params.append([key1, key3])

    data_shape = (259, 2, 2049)
    data_shape = (1, 256, 2, 2049)
    audio_shape = (264600, 1)
    inp_stft = np.random.random(data_shape)

    out_stfts_1, out_stfts_2 = {}, {}
    inp_stft_contiguous = np.abs(np.ascontiguousarray(inp_stft))


    target1 = targets1[0]
    target2 = targets2[0]

    params = {}


    # Load the model weights for corresponding target
    model_path = os.path.abspath(model_path_2)
    # nn.load_parameters(model_path)
        # params = nn.get_parameters()
        # params = {name: value for name, value in params.items() if name.startswith(targets2[0])}
        # nn.clear_parameters()
    # nn.load_parameters(model_path)
    # with open("/Users/daniellebenbashat/PycharmProjects/audio/ai-research-code/d3net/music-source-separation/configs/vocals.yaml") as file:
    #     # Load target specific Hyper parameters
    #     hparams = yaml.load(file, Loader=yaml.FullLoader)
    # with nn.parameter_scope("XXX"):
    #     d3net = D3NetMSS(hparams, comm=None, input_mean=None,
    #                  input_scale=None, init_method='xavier')
    #
    # nn.clear_parameters()

    # Load the model weights for corresponding target
    model_path = os.path.abspath(model_path_1)
    nn.load_parameters(model_path_1)
        # params = nn.get_parameters()
        # params = {name: value for name, value in params.items() if name.startswith(targets2[0])}
        # nn.clear_parameters()
    target = target1
    with nn.parameter_scope(target2):
        params4 = nn.get_parameters()

    with open("/Users/daniellebenbashat/PycharmProjects/audio/ai-research-code/d3net/music-source-separation/configs/vocals.yaml") as file:
        # Load target specific Hyper parameters
        hparams = yaml.load(file, Loader=yaml.FullLoader)
    with nn.parameter_scope(target):
        d3net1 = D3NetMSS(hparams, comm=None, input_mean=None,
                         input_scale=None, init_method='xavier')
        out = d3net1(inp_stft)

    nn.clear_parameters()

    # Load the model weights for corresponding target
    model_path = os.path.abspath(model_path_2)
    nn.load_parameters(model_path)
        # params = nn.get_parameters()
        # params = {name: value for name, value in params.items() if name.startswith(targets2[0])}
        # nn.clear_parameters()
    target = target2
    with open("/Users/daniellebenbashat/PycharmProjects/audio/ai-research-code/d3net/music-source-separation/configs/vocals.yaml") as file:
        # Load target specific Hyper parameters
        hparams = yaml.load(file, Loader=yaml.FullLoader)
    with nn.parameter_scope(target):
        d3net2 = D3NetMSS(hparams, comm=None, input_mean=None,
                         input_scale=None, init_method='xavier')


    params = {}
    for i, (model_path, out_stfts) in enumerate(zip([model_path_1, model_path_2], [out_stfts_1, out_stfts_2])):
        for target in targets[i]:
            with nn.parameter_scope(target):
                # Load the model weights for corresponding target
                model_path = os.path.abspath(model_path)
                nn.load_parameters(model_path)
                ps = nn.get_parameters()
                ps = {name: value for name, value in ps.items() if name.startswith(target)}
                params[f"{model_path}"] = ps
                nn.clear_parameters()

    cnt = 0
    for key1, key2 in zip(params[model_path_1].keys(), params[model_path_2].keys()):
        if np.allclose(params[model_path_1][key1].d, params[model_path_2][key2].d):
            cnt += 1
            print(f"Layers {key1}, {key2} WITH the same params")
        # else:
            # print(f"Layers {key1}, {key2} without the same params")


    for model_path, out_stfts in zip([model_path_1, model_path_2], [out_stfts_1, out_stfts_2]):
        for target in targets:
            # Load the model weights for corresponding target
            model_path = os.path.abspath(model_path)
            nn.load_parameters(model_path)
            print(f"Loaded Parameters {model_path}")
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"configs/{target}.yaml")) as file:
                # Load target specific Hyper parameters
                hparams = yaml.load(file, Loader=yaml.FullLoader)
            with nn.parameter_scope(target):
                out_sep = model_separate(
                    inp_stft_contiguous, hparams, ch_flip_average=True)
                out_stfts[target] = out_sep * np.exp(1j * np.angle(inp_stft))

    # out_stfts = apply_mwf(out_stfts, inp_stft)
    data_shape = (1, 256, 2, 2049)
    mix_spec = np.random.random(data_shape)
    outs = {}
    for model_path, out_stfts in zip([model_path_1, model_path_2], [out_stfts_1, out_stfts_2]):
        for target in targets:
            with nn.parameter_scope(target):

                d3net = D3NetMSS(hparams, comm=None, input_mean=None,
                             input_scale=None, init_method='xavier')

                pred_spec = d3net(mix_spec)         # original MusDB: mix_spec.shape: (6, 256, 2, 2049), also the pred_spec: (6, 256, 2, 2049)
                outs[model_path] = pred_spec

    nn.clear_parameters()
