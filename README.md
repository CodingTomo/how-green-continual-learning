# How green is continual learning, really? Analyzing the Energy Consumption in Continual Training of Vision Foundation Models

![alt text](https://github.com/CodingTomo/how-green-continual-learning/blob/main/src/methodology.png?raw=true)

This repository provides the official code for the **Green Foundation Models (GreenFOMO)** workshop paper, presented at ECCV 2024. The paper investigates the energy consumption of continually training vision foundation models, benchmarking their environmental impact.

This project builds on [PILOT](https://github.com/sun-hailong/LAMDA-PILOT), integrating energy tracking using CodeCarbon to measure carbon emissions and energy usage during model training.

## Installation
1. Clone the repository:
```bash
   git clone https://github.com/CodingTomo/how-green-continual-learning.git
```
2. Install dependencies from [PILOT](https://github.com/sun-hailong/LAMDA-PILOT) following its instructions.
3. To track energy consumption, install the ```CodeCarbon``` package:
```
pip install codecarbon
```
For troubleshooting ```CodeCarbon```, refer to its [official repository](https://github.com/mlco2/codecarbon).


## Usage
To run training experiments with energy tracking:
```
python main.py --config exps/METHOD_NAME.yaml
```
Modify METHOD_NAME.yaml to switch between different continual learning methods.

## Dataset
- ImageNet-R: Follow setup instructions from the [PILOT repository](https://github.com/sun-hailong/LAMDA-PILOT).
- DomainNet (for Incremental Learning): Follow the instructions in the [DN4IL repository](https://github.com/NeurAI-Lab/DN4IL-dataset)

To replicate the paper experiments on DN4IL, the dn_split folder contains the splits used. Place these files in the dataset directory before training.

## Citing
If you use this repository in your research, please cite the following:
```
@inproceedings{green_fomo_2024,
  title={How Green is Continual Learning, Really? Analyzing the Energy Consumption in Continual Training of Vision Foundation Models},
  author={Your Name et al.},
  booktitle={Proceedings of the ECCV 2024 Workshop},
  year={2024}
}
```

```
@article{sun2023pilot,
  title={PILOT: Pre-training with Contrastive Learning of Global and Local Features for Incremental Object Detection},
  author={Sun, Hailong and others},
  journal={arXiv preprint arXiv:2301.12528},
  year={2023}
}
```

## Results
![alt text](https://github.com/CodingTomo/how-green-continual-learning/blob/main/src/train_energy_vs_accuracy_all.jpeg)

## License

This project is licensed under the MIT License. See the `LICENSE` file for more information.

## Contact
For any questions or issues, please open an issue in this repository.
