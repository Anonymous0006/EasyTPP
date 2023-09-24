# EasyTPP



`EasyTPP` is an easy-to-use development and application toolkit for [Temporal Point Process](https://mathworld.wolfram.com/TemporalPointProcess.html) (TPP), with key features in configurability, compatibility and reproducibility. We hope this project could benefit both researchers and practitioners with the goal of easily customized development and open benchmarking in TPP.
<span id='top'/>





## Features <a href='#top'>[Back to Top]</a>
<span id='features'/>

- **Configurable and customizable**: models are modularized and configurable，with abstract classes to support developing customized
  TPP models.
- **Compatible with both Tensorflow and PyTorch framework**: `EasyTPP` implements two equivalent sets of models, which can
  be run under Tensorflow (both Tensorflow 1.13.1 and Tensorflow 2.0) and PyTorch 1.7.0+ respectively. While the PyTorch models are more popular among researchers, the compatibility with Tensorflow is important for industrial practitioners.
- **Reproducible**: all the benchmarks can be easily reproduced.
- **Hyper-parameter optimization**: a pipeline of [optuna](https://github.com/optuna/optuna)-based HPO is provided.


## Model List <a href='#top'>[Back to Top]</a>
<span id='model-list'/>

We provide reference implementations of various state-of-the-art TPP papers:

| No  | Publication |     Model     | Paper                                                                                                                                    | Implementation                                                                                                             |
|:---:|:-----------:|:-------------:|:-----------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------|
|  1  |   KDD'16    |     RMTPP     | [Recurrent Marked Temporal Point Processes: Embedding Event History to Vector](https://www.kdd.org/kdd2016/papers/files/rpp1081-duA.pdf) | [Tensorflow](easy_tpp/model/tf_model/tf_rmtpp.py)<br/>[Torch](easy_tpp/model/torch_model/torch_rmtpp.py)                   |
|  2  | NeurIPS'17  |      NHP      | [The Neural Hawkes Process: A Neurally Self-Modulating Multivariate Point Process](https://arxiv.org/abs/1612.09328)                     | [Tensorflow](easy_tpp/model/tf_model/tf_nhp.py)<br/>[Torch](easy_tpp/model/torch_model/torch_nhp.py)                       |
|  3  | NeurIPS'19  |    FullyNN    | [Fully Neural Network based Model for General Temporal Point Processes](https://arxiv.org/abs/1905.09690)                                | [Tensorflow](easy_tpp/model/tf_model/tf_fullnn.py)<br/>[Torch](easy_tpp/model/torch_model/torch_fullynn.py)                |
|  4  |   ICML'20   |     SAHP      | [Self-Attentive Hawkes process](https://arxiv.org/abs/1907.07561)                                                                        | [Tensorflow](easy_tpp/model/tf_model/tf_sahp.py)<br/>[Torch](easy_tpp/model/torch_model/torch_sahp.py)                     |
|  5  |   ICML'20   |      THP      | [Transformer Hawkes process](https://arxiv.org/abs/2002.09291)                                                                           | [Tensorflow](easy_tpp/model/tf_model/tf_thp.py)<br/>[Torch](easy_tpp/model/torch_model/torch_thp.py)                       |
|  6  |   ICLR'20   | IntensityFree | [Intensity-Free Learning of Temporal Point Processes](https://arxiv.org/abs/1909.12127)                                                  | [Tensorflow](easy_tpp/model/tf_model/tf_intensity_free.py)<br/>[Torch](easy_tpp/model/torch_model/torch_intensity_free.py) |
|  7  |   ICLR'21   |    ODETPP     | [Neural Spatio-Temporal Point Processes (simplified)](https://arxiv.org/abs/2011.04583)                                                  | [Tensorflow](easy_tpp/model/tf_model/tf_ode_tpp.py)<br/>[Torch](easy_tpp/model/torch_model/torch_ode_tpp.py)               |
|  8  |   ICLR'22   |    AttNHP     | [Transformer Embeddings of Irregularly Spaced Events and Their Participants](https://arxiv.org/abs/2201.00044)                           | [Tensorflow](easy_tpp/model/tf_model/tf_attnhp.py)<br/>[Torch](easy_tpp/model/torch_model/torch_attnhp.py)                 |



## Dataset <a href='#top'>[Back to Top]</a>
<span id='dataset'/>

We preprocessed one synthetic and five real world datasets from widely-cited works that contain diverse characteristics in terms of their application domains and temporal statistics:
- Synthetic: a univariate Hawkes process simulated by [Tick](https://github.com/X-DataInitiative/tick) library.
- Retweet ([Zhou, 2013](http://proceedings.mlr.press/v28/zhou13.pdf)): timestamped user retweet events.
- Taxi ([Whong, 2014](https://chriswhong.com/open-data/foil_nyc_taxi/)): timestamped taxi pick-up events.
- StackOverflow ([Leskovec, 2014](https://snap.stanford.edu/data/)): timestamped user badge reward events in StackOverflow.
- Taobao ([Xue et al, 2022](https://arxiv.org/abs/2210.01753)): timestamped user online shopping behavior events in Taobao platform.
- Amazon ([Xue et al, 2022](https://nijianmo.github.io/amazon/)): timestamped user online shopping behavior events in Amazon platform.

In addition, we processed two non-anthropogenic datasets 
- [Earthquake](https://drive.google.com/drive/folders/1ubeIz_CCNjHyuu6-XXD0T-gdOLm12rf4): timestamped earthquake events over the Conterminous U.S from 1996 to 2023, processed from [USGS](https://www.usgs.gov/programs/earthquake-hazards/science/earthquake-data).
- [Volcano eruption](https://drive.google.com/drive/folders/1KSWbNi8LUwC-dxz1T5sOnd9zwAot95Tp?usp=drive_link): timestamped volcano eruption events over the world in recent hundreds of years, processed from [The Smithsonian Institution](https://volcano.si.edu/).


  All datasets are preprocess to the `Gatech` format dataset widely used for TPP researchers, and saved at [Google Drive](https://drive.google.com/drive/u/0/folders/1f8k82-NL6KFKuNMsUwozmbzDSFycYvz7) with a public access.

## Quick Start <a href='#top'>[Back to Top]</a>
<span id='quick-start'/>

We provide an end-to-end example for users to run a standard TPP model with `EasyTPP`.


### Step 1. Installation

First of all, we can install the package from the source code on Github.

```bash
git clone https://github.com/Anonymous0006/EasyTPP.git
cd EasyTemporalPointProcess
python setup.py install
```


### Step 2. Prepare datasets 

We need to put the datasets in a local directory before running a model and the datasets should follow a certain format.

Suppose we use the [taxi dataset](https://chriswhong.com/open-data/foil_nyc_taxi/) in the example.

### Step 3. Train the model


Before start training, we need to set up the config file for the pipeline. We provide a preset config file in [Example Config](https://github.com/Anonymous0006/EasyTPP/blob/main/examples/configs/experiment_config.yaml)

After the setup of data and config, the directory structure is as follows:

```bash

    data
     |______taxi
             |____ train.pkl
             |____ dev.pkl
             |____ test.pkl

    configs
     |______experiment_config.yaml

```

Then we start the training by simply running the script 

```python

import argparse
from easy_tpp.config_factory import Config
from easy_tpp.runner import Runner


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_dir', type=str, required=False, default='configs/experiment_config.yaml',
                        help='Dir of configuration yaml to train and evaluate the model.')

    parser.add_argument('--experiment_id', type=str, required=False, default='NHP_train',
                        help='Experiment id in the config file.')

    args = parser.parse_args()

    config = Config.build_from_yaml_file(args.config_dir, experiment_id=args.experiment_id)

    model_runner = Runner.build_from_config(config)

    model_runner.run()


if __name__ == '__main__':
    main()

```


## Documentation <a href='#top'>[Back to Top]</a>
<span id='doc'/>

The classes and methods of `EasyTPP` have been well documented so that users can generate the documentation by:

```shell
cd doc
pip install -r requirements.txt
make html
```
NOTE:
* The `doc/requirements.txt` is only for documentation by Sphinx, which can be automatically generated by Github actions `.github/workflows/docs.yml`. (Trigger by pull request.)

## Benchmark <a href='#top'>[Back to Top]</a>
<span id='benchmark'/>

In the [examples](https://github.com/Anonymous0006/EasyTPP/tree/main/examples) folder, we provide a [script](https://github.com/Anonymous0006/EasyTPP/blob/main/examples/benchmark_script.py) to benchmark the TPPs, with Taxi dataset as the input. 

To run the script, one should download the Taxi data following the above instructions. The [config](https://github.com/Anonymous0006/EasyTPP/blob/main/examples/configs/experiment_config.yaml) file is readily setup up. Then run


```shell
cd examples
python benchmark_script.py
```


## License <a href='#top'>[Back to Top]</a>

This project is licensed under the Apache License (Version 2.0). This toolkit also contains some code modified from other repos under other open-source licenses. See the [NOTICE](https://github.com/Anonymous0006/EasyTPP/blob/master/NOTICE) file for more information.