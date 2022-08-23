#!/usr/bin/env python3
"""Script to test a pytorch model on ImageNet's validation set."""

import json
from xgen.xgen_run import xgen
from train_script_main import training_main
from xgen.utils.args_ai_map import get_old_config





if __name__ == '__main__':

    json_path = './mobilenet_config/xgen_test.json'
    # json_path = 'args_ai_template_sgpu.json'


    #if you are using new config

    old_json_path = 'args_ai_template_old.json'
    with open(json_path) as f:
        new = json.load(f)
    old = get_old_config(new)
    with open(old_json_path, 'w') as f:
        json.dump(old, f)
    json_path = old_json_path

    def run(onnx_path, quantized, pruning, output_path, **kwargs):
        import random
        res = {}
        # for simulation
        pr = kwargs['sp_prune_ratios']
        res['output_dir'] = output_path

        res['latency'] = 50

        return res


    # xgen(training_main, run, xgen_config_path=json_path, xgen_mode='compatible_testing')

    xgen(training_main, run, xgen_config_path=json_path, xgen_mode='customization')
