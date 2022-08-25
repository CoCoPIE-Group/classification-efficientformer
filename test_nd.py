#!/usr/bin/env python3
"""Script to test a pytorch model on ImageNet's validation set."""

import json
from xgen.xgen_run import xgen
from train_script_main import training_main
from xgen.utils.args_ai_map import get_old_config
from co_lib.utils import deep_update_dict


def inverse_clever_format(cfstr):
    num = cfstr[:-1]
    C = cfstr[-1]
    num = float(num)
    if C=="T":
        num = num*1e12
    elif C=="G":
        num = num * 1e9
    elif C == "M":
        num = num * 1e6
    elif C == "K":
        num = num * 1e3
    return num

if __name__ == '__main__':

    json_path = './mobilenet_nd_config/xgen.json'
    json_test_path =  './mobilenet_nd_config/xgen_test.json'
    # json_path = 'args_ai_template_sgpu.json'


    #if you are using new config

    old_json_path = 'args_ai_template_old.json'
    with open(json_path) as f:
        new = json.load(f)
    with open(json_test_path) as f:
        new_test = json.load(f)
    deep_update_dict(new,new_test)
    old = get_old_config(new)
    with open(old_json_path, 'w') as f:
        json.dump(old, f)
    json_path = old_json_path

    class CompilerSimulator:
        def __init__(self,base=None):
            self.base = base
        def __call__(self,onnx_path, quantized, pruning, output_path, **kwargs):

            res = {}
            # for simulation
            res['latency'] = 50
            if "internal_data" in kwargs:
                ttnzp = kwargs['internal_data']['total_nz_parameters']
                ttnzp = inverse_clever_format(ttnzp)
                if self.base is None:
                    self.base = ttnzp
                res['latency'] = 100*ttnzp/self.base
            res['output_dir'] = output_path

            return res

    run = CompilerSimulator()

    xgen(training_main, run, xgen_config_path=json_path, xgen_mode='compatible_testing')

    print('compatible_testing success!')
    old['origin']['common_train_epochs'] = 1
    with open(old_json_path, 'w') as f:
        json.dump(old, f)

    xgen(training_main, run, xgen_config_path=json_path, xgen_mode='customization')

    print('customization training 1 epoch success!')
    old['origin']['common_train_epochs'] = 0
    with open(old_json_path, 'w') as f:
        json.dump(old, f)

    xgen(training_main, run, xgen_config_path=json_path, xgen_mode='pruning')

    print('pruning 0 epoch success!')


    xgen(training_main, run, xgen_config_path=json_path, xgen_mode='scaling')

    print('scaling 0 epoch success!')
