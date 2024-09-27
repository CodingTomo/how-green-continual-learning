import json
import argparse
from trainer import train

def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    if args.only_inference == "y":
        if "tuned_epoch" in param.keys():
            param["tuned_epoch"] = 1
        if "init_epoch" in param.keys():
            param["init_epoch"] = 1
        if "epochs" in param.keys():
            param["epochs"] = 1
        if "later_epochs" in param.keys():
            param["later_epochs"] = 1
        if "init_epochs" in param.keys():
            param["init_epochs"] = 1
    args = vars(args) # Converting argparse Namespace to a dict.
    args.update(param) # Add parameters from json

    train(args)

def load_json(setting_path):
    with open(setting_path) as data_file:
        param = json.load(data_file)
    return param

def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple pre-trained incremental learning algorthms.')
    parser.add_argument('--config', type=str, default='./exps/simplecil.json',
                        help='Json file of settings.')
    parser.add_argument('--only_inference', type=str, default="n", choices=["y", "n"])
    return parser

if __name__ == '__main__':
    main()
