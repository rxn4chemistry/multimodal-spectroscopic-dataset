from rdkit import Chem, RDLogger
from pathlib import Path
import click
import pandas as pd
import tqdm.auto as tqdm

def canonicalise(smiles):
    RDLogger.DisableLog('rdApp.*')
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    else:
        return Chem.MolToSmiles(mol)

@click.command()
@click.option("--pred_path",
              type=click.Path(),
              )
@click.option("--test_path",
              type=click.Path())
def main(pred_path, test_path):

    with Path(test_path).open("r") as test_file:
        test = test_file.readlines()
    test = [canonicalise(smiles.replace(' ', '')) for smiles in tqdm.tqdm(test)]

    with Path(pred_path).open("r") as pred_file:
        pred = pred_file.readlines()
    pred = [canonicalise(smiles.replace(' ', '')) for smiles in tqdm.tqdm(pred)]
    pred = [pred[i*10 : (i+1)*10] for i in range(len(pred) // 10)]

    max_len = min(len(pred), len(test))

    pred_data = pd.DataFrame({'pred': pred[:max_len], 'target': test[:max_len]})
    pred_data['rank'] = pred_data.apply(lambda row : row['pred'].index(row['target']) if row['target'] in row['pred'] else 10, axis=1)

    for i in range(1, 11):
        print(f"Top {i}: {(pred_data['rank'] < i).sum() / len(pred_data):.5f}")

if __name__ == "__main__":
    main()
