# Tiny_CD

This is the implementation of:

You can find the complete work here: 

## Installation and Requirements

The easiest way to reproduce our results is to create a folder named "TinyCD" on your device and then
clone this repository in "TinyCD":

```shell
git clone https://github.com/AndreaCodegoni/Tiny_CD.git
```

Then, you can create a virtual ``conda`` environment named ``TinyCD`` with the following cmd:

```shell
conda create --name TinyCD --file requirements.txt
conda activate TinyCD
```

## Dataset 

You can find the original datasets at this two links:

LEVIR-CD: https://justchenhao.github.io/LEVIR/

WHU-CD: https://study.rsgis.whu.edu.cn/pages/download/building_dataset.html

Then you have to process the data as described in the paper in order to obtain the following structure:

```
Change detection data set with pixel-level binary labels；
├─A
├─B
├─label
└─list
```

`A`: images of t1 phase;

`B`:images of t2 phase;

`label`: label maps;

`list`: contains `train.txt, val.txt and test.txt`, each file records the image names (XXX.png) in the change detection dataset.

If you prefer, you can download the pre-processed dataset using the following:

LEVIR-CD 
```cmd
wget https://www.dropbox.com/s/h9jl2ygznsaeg5d/LEVIR-CD-256.zip
```
WHU-CD
```cmd
wget https://www.dropbox.com/s/r76a00jcxp5d3hl/WHU-CD-256.zip
```

If you have any trouble with the datasets, feel free to contact us.


## Evaluate pretrained models

If you want to evaluate your trained model, or if you want to reproduce the paper results with the pretrained models that 
you can find in the "pretrained_models" folder, you can run:

```cmd
python test_ondata.py --datapath "Your_data_path" --modelpath "Your_path_to_pretrained_model"
```

## Train your model

You can re-train our model, or if you prefer you can play with the parameters of our model and then train it using 

```cmd
python training.py --datapath "Your_data_path" --log-path "Path_to_save_logs_and_models_checkpoints"
```

## References

We want to mention the following repository that greatly help us in our work:

- https://github.com/justchenhao/BIT_CD We have used this repository in the visual comparison and to report other state-of-the-art results on the two datasets.
- https://github.com/wgcban/ChangeFormer/ and https://github.com/wgcban/SemiCD for the datasets.

## License
Code is released for non-commercial and research purposes **only**. For commercial purposes, please contact the authors.