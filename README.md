# Multimodal Spectroscopic Dataset
This project is designed to work with  Multimodal-Spectroscopic-Dataset. It requires a specific conda environment and data files to run properly.

## Installation


Clone the repository or download the project files by running

```
git clone https://github.com/rxn4chemistry/multimodal-spectroscopic-dataset.git
```
#### Conda install 

Open a terminal or command prompt and navigate to the project directory.
Create a new conda environment named multispecdata with the required dependencies:
```
conda create -n multispecdata python=3.10
```
activate the created conda enviroment and insall the required dependencies

```
conda activate multispecdata
pip install -r requirements.txt
```

## Data Download
The project requires specific data files to run. You can download the data from the following link:
[https://zenodo.org/records/11611178](https://zenodo.org/records/11611178)

Extract the downloaded ZIP file into the project directory in the data folder.

## Replicating experiments

### XGBoost and CNN

To replicate the results presented in the paper for XGBoost and CNNs use execute the following two scripts:

```bash
run_cnn.sh
run_xgboost.sh
```

### Transformer models:

For the transformer models we present a set of different experiments including predicting the structure of a molecule from spectra, the spectra from the molecule. For all experiments the training data was generated using the following script:

```
python benchmark/generate_input.py 
  # Paths
  --analytical_data <Path to the folder containing the dataset> 
  --out_path <Path where the data will be saved>

  # Flags controlling input 
  --formula <Flag: Wether or not to include molecular formula in the input>
  --h_nmr <Flag: Wether or not to include 13C-NMR in the input>
  --c_nmr <Flag: Wether or not to include 1H-NMR in the input>
  --ir <Flag: Wether or not to include IR spectra in the input>
  --pos_msms <Flag: Wether or not to include positive MS/MS spectra in the input>
  --neg_msms <Flag: Wether or not to include negative MS/MS spectra in the input>

  # Flags controlling target
  --pred_spectra <Flag: By default the target is the molecular structure as SMILES. If this flag is used this is reversed with SMILES as input and spectra as output.>

  --seed <Default: 3245>
```

To start training the model use the following script:

```
python benchmark/start_training.py 
  --output_path <Path to the out_path specified when generating the data.>
  --template_path <Path to the OpenNMT config template to use. Default: ./benchmark/transformer_template.yaml>
  --seed <Seed. Default: 3245>
```

With a trained model inference can be run using the following OpenNMT command:

```
onmt_translate 
  -model <model_path> 
  -src <src_path> 
  -output <out_file> 
  -beam_size 10 
  -n_best 10 
  -min_length 5 
  -gpu 0
```

To analyse the results use the following script:

```
python benchmark/analyse_results.py 
  --pred_path <Path to the predictions made with onmt_translate>
  --test_path <Path to the ground truth test data>
```

We have provided a script `run_transformer.sh` to train models used in the structure prediction experiments for 1H-NMR, 13C-NMR and the combination of 1H-NMR and 13C-NMR.

## Citation
Cite our work as followes:
```
@article{alberts2024unraveling,
  title={Unraveling Molecular Structure: A Multimodal Spectroscopic Dataset for Chemistry},
  author={Alberts, Marvin and Schilter, Oliver and Zipoli, Federico and Hartrampf, Nina and Laino, Teodoro},
  year={2024},
  url={https://arxiv.org/abs/2407.17492}, 
}
```

## Acknowledgements
This publication was created as part of NCCR Catalysis (grant number 180544), a National Centre of Competence in Research funded by the Swiss National Science Foundation.