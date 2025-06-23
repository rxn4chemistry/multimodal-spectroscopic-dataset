from pathlib import Path

import click
import pandas as pd
from tqdm.auto import tqdm
from typing import Tuple, List, Dict, Union

from rxn.chemutils.tokenization import tokenize_smiles
from sklearn.model_selection import train_test_split
import regex as re
from scipy.interpolate import interp1d
import numpy as np


def split_data(data: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame]:
    train, test = train_test_split(data, test_size=0.1, random_state=seed, shuffle=True)
    train, val = train_test_split(train, test_size=0.05, random_state=seed, shuffle=True)

    return train, test, val

def tokenize_formula(formula: str) -> list:
    return ' '.join(re.findall("[A-Z][a-z]?|\d+|.", formula)) + ' '

def process_hnmr(multiplets: List[Dict[str, Union[str, float, int]]]) -> str:

    multiplet_str = "1HNMR "
    for peak in multiplets:
        range_max = float(peak["rangeMax"]) 
        range_min = float(peak["rangeMin"]) 

        formatted_peak = ""
        formatted_peak = formatted_peak + "{:.2f} {:.2f} ".format(range_max, range_min)        
        formatted_peak = formatted_peak +  "{} {}H ".format(
                                                            peak["category"],
                                                            peak["nH"],
                                                        )
        js = str(peak["j_values"])
        if js != "None":
            split_js = js.split("_")
            split_js = list(filter(None, split_js))
            processed_js = ["{:.2f}".format(float(j)) for j in split_js]
            formatted_js = "J " + " ".join(processed_js)
            formatted_peak += formatted_js

        multiplet_str += formatted_peak.strip() + " | "

    # Remove last separating token
    multiplet_str = multiplet_str[:-2]
    return multiplet_str

def process_cnmr(carbon_nmr: List[Dict[str, Union[str, float, int]]]) -> str:
    nmr_string = "13CNMR "
    for peak in carbon_nmr:
        nmr_string += str(round(float(peak["delta (ppm)"]), 1)) + " "

    return nmr_string

def process_ir(ir: np.ndarray, interpolation_points: int = 400) -> str:
    original_x = np.linspace(400, 4000, 1800)
    interpolation_x = np.linspace(400, 4000, interpolation_points)

    
    interp = interp1d(original_x, ir)
    interp_ir = interp(interpolation_x)

    # Normalise
    interp_ir = interp_ir + abs(min(interp_ir))
    interp_ir = (interp_ir / max(interp_ir)) * 100 
    interp_ir = np.round(interp_ir, decimals=0).astype(int).astype(str)
    return 'IR ' + ' '.join(interp_ir) + ' '

def process_msms(msms: List[List[float]]) -> List[str]:
    msms_string = ''
    for peak in msms:
        msms_string = msms_string + "{:.1f} {:.1f} ".format(
            round(peak[0], 1), round(peak[1], 1)
        )
    return msms_string


def tokenise_data(
    data: pd.DataFrame,
    h_nmr: bool, 
    c_nmr: bool,
    ir: bool,
    pos_msms: bool, 
    neg_msms: bool,
    formula: bool
):
    input_list = list()

    for i in tqdm(range(len(data))):
        tokenized_formula = tokenize_formula(data.iloc[i]['molecular_formula'])
        
        if formula:
            tokenized_input = tokenized_formula
        else:
            tokenized_input = ''

        if h_nmr:
            h_nmr_string = process_hnmr(data.iloc[i]['h_nmr_peaks'])
            tokenized_input += h_nmr_string

        if c_nmr:
            c_nmr_string = process_cnmr(data.iloc[i]['c_nmr_peaks'])
            tokenized_input += c_nmr_string

        if ir:
            ir_string = process_ir(data.iloc[i]["ir_spectra"])
            tokenized_input += ir_string

        if pos_msms:
            pos_msms_string = ''
            pos_msms_string += "E0Pos " + process_msms(data.iloc[i]["msms_positive_10ev"])
            pos_msms_string += "E1Pos " + process_msms(data.iloc[i]["msms_positive_20ev"])
            pos_msms_string += "E2Pos " + process_msms(data.iloc[i]["msms_positive_40ev"])
            tokenized_input += pos_msms_string

        if neg_msms:
            neg_msms_string = ''
            neg_msms_string += "E0Neg " + process_msms(data.iloc[i]["msms_negative_10ev"])
            neg_msms_string += "E1Neg " + process_msms(data.iloc[i]["msms_negative_20ev"])
            neg_msms_string += "E2Neg " + process_msms(data.iloc[i]["msms_negative_40ev"])
            tokenized_input += neg_msms_string
        
        tokenized_target = tokenize_smiles(data.iloc[i]["smiles"])
        input_list.append({'source': tokenized_input.strip(), 'target': tokenized_target})

    input_df = pd.DataFrame(input_list)
    input_df = input_df.drop_duplicates(subset="source")

    return input_df


def save_set(data_set: pd.DataFrame, out_path: Path, set_type: str, pred_spectra: bool) -> None:
    out_path.mkdir(parents=True, exist_ok=True)

    smiles = list(data_set.target)
    spectra = data_set.source

    with (out_path / f"src-{set_type}.txt").open("w") as f:
        if pred_spectra:
            src = smiles
        else:
            src = spectra

        for item in src:
            f.write(f"{item}\n")
        
    with (out_path / f"tgt-{set_type}.txt").open("w") as f:
        if pred_spectra:
            tgt = spectra
        else:
            tgt = smiles

        for item in tgt:
            f.write(f"{item}\n")


@click.command()
@click.option(
    "--analytical_data",
    "-n",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the NMR dataframe",
)
@click.option(
    "--out_path",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output path",
)
@click.option("--h_nmr", is_flag=True)
@click.option("--c_nmr", is_flag=True)
@click.option("--ir", is_flag=True)
@click.option("--pos_msms", is_flag=True)
@click.option("--neg_msms", is_flag=True)
@click.option("--formula", is_flag=True)
@click.option("--pred_spectra", is_flag=True)
@click.option("--seed", type=int, default=3245)
def main(
    analytical_data: Path,
    out_path: Path,
    h_nmr: bool = False,
    c_nmr: bool = False,
    ir: bool = False,
    pos_msms: bool = False,
    neg_msms: bool = False,
    formula: bool = True,
    pred_spectra: bool = False,
    seed: int = 3245
):  
    
    # Make the training data
    tokenised_data = list()
    for parquet_file in tqdm.tqdm(analytical_data.glob("*.parquet"), total=245):
        print(parquet_file.stem)
        data = pd.read_parquet(parquet_file)
        tokenised_data.append(tokenise_data(data, h_nmr, c_nmr, ir, pos_msms, neg_msms, formula))
        del data

    tokenised_data = pd.concat(tokenised_data)

    train_set, test_set, val_set = split_data(tokenised_data, seed)

    # Save training data
    out_data_path = out_path / "data"
    save_set(test_set, out_data_path, "test", pred_spectra)
    save_set(train_set, out_data_path, "train", pred_spectra)
    save_set(val_set, out_data_path, "val", pred_spectra)

if __name__ == '__main__':
    main()