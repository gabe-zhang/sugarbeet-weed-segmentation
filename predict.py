""" Predict with semantic segmentation model.
"""
import argparse
import os
import time
from typing import Dict

import oyaml as yaml
import pytorch_lightning as pl
import torch
import torch_tensorrt  # noqa: F401
from callbacks import (
    PostprocessorrCallback,
    VisualizerCallback,
    get_postprocessors,
    get_visualizers,
)
from datasets import get_data_module
from modules import get_backbone, get_criterion, module
from pytorch_lightning import Trainer


def parse_args() -> Dict[str, str]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export_dir", required=True, help="Path to export dir which saves logs, metrics, etc.")
    parser.add_argument("--config", required=True, help="Path to configuration file (*.yaml)")
    parser.add_argument("--ckpt_path", required=True, help="Provide *.ckpt file to continue training.")

    args = vars(parser.parse_args())

    return args


def load_config(path_to_config_file: str) -> Dict:
    assert os.path.exists(path_to_config_file)

    with open(path_to_config_file) as istream:
        config = yaml.safe_load(istream)

    return config


def main():
    args = parse_args()
    cfg = load_config(args['config'])

    datasetmodule = get_data_module(cfg)
    criterion = get_criterion(cfg)

    # define backbone
    # network = get_backbone(cfg)

    class NetworkWrapper():
        def __init__(self, network, num_classes=3):
            self.network = network
            self.num_classes = num_classes
        
        def forward(self, image):
            return self.network.forward(image)
    network = torch.jit.load("models/erfnet_tensorrt.ts")
    network = NetworkWrapper(network)


    seg_module = module.SegmentationNetwork(network, 
                                            criterion, 
                                            cfg['train']['learning_rate'],
                                            cfg['train']['weight_decay'],
                                            predict_step_settings=cfg['predict']['step_settings'])

    # Add callbacks
    visualizer_callback = VisualizerCallback(get_visualizers(cfg), cfg['train']['vis_train_every_x_epochs'])
    postprocessor_callback = PostprocessorrCallback(
        get_postprocessors(cfg), cfg['train']['postprocess_train_every_x_epochs'])

    # Setup trainer
    trainer = Trainer(
        gpus=cfg['predict']['n_gpus'],
        default_root_dir=args['export_dir'],
        max_epochs=cfg['train']['max_epoch'],
        callbacks=[visualizer_callback, postprocessor_callback])
    start = time.time()
    # trainer.predict(seg_module, dataloaders=datasetmodule, ckpt_path=args['ckpt_path'])
    trainer.predict(seg_module, dataloaders=datasetmodule)
    total_time = time.time() - start
    print(f"{round(total_time / len(datasetmodule), 2)}s per image:"
          f"{round(total_time, 2)}s for {len(datasetmodule)} images.")

if __name__ == '__main__':
    main()
