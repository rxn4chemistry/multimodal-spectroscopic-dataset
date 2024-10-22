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
Running the Project
After activating the multispecdata environment and downloading the data, you can run the the benchmark models with the following command:

```bash
run_transformer.sh
run_cnn.sh
run_xgboost.sh
```

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