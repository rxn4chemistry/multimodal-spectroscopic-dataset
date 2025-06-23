# Adapted from Guwon Jung: https://github.com/gj475/irchracterizationcnn

import numpy as np
import pickle
import pandas as pd

from keras import backend as K
from keras.models import Model
from keras.layers import Input, MaxPooling1D, Dropout, Activation
from keras.layers import Conv1D, Dense, Flatten, BatchNormalization
from keras.optimizers import Adam
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split 
from pathlib import Path
import click
from rdkit import Chem
from rdkit import RDLogger
from scipy.interpolate import interp1d


functional_groups = {
    'Acid anhydride': Chem.MolFromSmarts('[CX3](=[OX1])[OX2][CX3](=[OX1])'),
    'Acyl halide': Chem.MolFromSmarts('[CX3](=[OX1])[F,Cl,Br,I]'),
    'Alcohol': Chem.MolFromSmarts('[#6][OX2H]'),
    'Aldehyde': Chem.MolFromSmarts('[CX3H1](=O)[#6,H]'),
    'Alkane': Chem.MolFromSmarts('[CX4;H3,H2]'),
    'Alkene': Chem.MolFromSmarts('[CX3]=[CX3]'),
    'Alkyne': Chem.MolFromSmarts('[CX2]#[CX2]'),
    'Amide': Chem.MolFromSmarts('[NX3][CX3](=[OX1])[#6]'),
    'Amine': Chem.MolFromSmarts('[NX3;H2,H1,H0;!$(NC=O)]'),
    'Arene': Chem.MolFromSmarts('[cX3]1[cX3][cX3][cX3][cX3][cX3]1'),
    'Azo compound': Chem.MolFromSmarts('[#6][NX2]=[NX2][#6]'),
    'Carbamate': Chem.MolFromSmarts('[NX3][CX3](=[OX1])[OX2H0]'),
    'Carboxylic acid': Chem.MolFromSmarts('[CX3](=O)[OX2H]'),
    'Enamine': Chem.MolFromSmarts('[NX3][CX3]=[CX3]'),
    'Enol': Chem.MolFromSmarts('[OX2H][#6X3]=[#6]'),
    'Ester': Chem.MolFromSmarts('[#6][CX3](=O)[OX2H0][#6]'),
    'Ether': Chem.MolFromSmarts('[OD2]([#6])[#6]'),
    'Haloalkane': Chem.MolFromSmarts('[#6][F,Cl,Br,I]'),
    'Hydrazine': Chem.MolFromSmarts('[NX3][NX3]'),
    'Hydrazone': Chem.MolFromSmarts('[NX3][NX2]=[#6]'),
    'Imide': Chem.MolFromSmarts('[CX3](=[OX1])[NX3][CX3](=[OX1])'),
    'Imine': Chem.MolFromSmarts('[$([CX3]([#6])[#6]),$([CX3H][#6])]=[$([NX2][#6]),$([NX2H])]'),
    'Isocyanate': Chem.MolFromSmarts('[NX2]=[C]=[O]'),
    'Isothiocyanate': Chem.MolFromSmarts('[NX2]=[C]=[S]'),
    'Ketone': Chem.MolFromSmarts('[#6][CX3](=O)[#6]'),
    'Nitrile': Chem.MolFromSmarts('[NX1]#[CX2]'),
    'Phenol': Chem.MolFromSmarts('[OX2H][cX3]:[c]'),
    'Phosphine': Chem.MolFromSmarts('[PX3]'),
    'Sulfide': Chem.MolFromSmarts('[#16X2H0]'),
    'Sulfonamide': Chem.MolFromSmarts('[#16X4]([NX3])(=[OX1])(=[OX1])[#6]'),
    'Sulfonate': Chem.MolFromSmarts('[#16X4](=[OX1])(=[OX1])([#6])[OX2H0]'),
    'Sulfone': Chem.MolFromSmarts('[#16X4](=[OX1])(=[OX1])([#6])[#6]'),
    'Sulfonic acid': Chem.MolFromSmarts('[#16X4](=[OX1])(=[OX1])([#6])[OX2H]'),
    'Sulfoxide': Chem.MolFromSmarts('[#16X3]=[OX1]'),
    'Thial': Chem.MolFromSmarts('[CX3H1](=S)[#6,H]'),
    'Thioamide': Chem.MolFromSmarts('[NX3][CX3]=[SX1]'),
    'Thiol': Chem.MolFromSmarts('[#16X2H]')
}


def match_group(mol: Chem.Mol, func_group) -> int:
    if type(func_group) is Chem.Mol:
        n = len(mol.GetSubstructMatches(func_group))
    else:
        n = func_group(mol)
    return 0 if n == 0 else 1

def get_functional_groups(smiles: str) -> dict:
    RDLogger.DisableLog('rdApp.*')
    smiles = smiles.strip().replace(' ', '')
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: 
        return None
    func_groups = list()
    for func_group_name, smarts in functional_groups.items():
        func_groups.append(match_group(mol, smarts))

    return func_groups


def train_model(X_train, y_train, X_test, num_fgs, aug, num, weighted):
    """Trains final model with the best hyper-parameters."""
    # Input
    X_train = X_train.reshape(X_train.shape[0], 600, 1)

    # Shape of input data.
    input_shape = X_train.shape[1:]
    input_tensor = Input(shape=input_shape)

    # 1st CNN layer.
    x = Conv1D(filters=31,
               kernel_size=(11), 
               strides=1,
               padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x)

    # 2nd CNN layer.
    x = Conv1D(filters=62,
       kernel_size=(11),
       strides=1,
       padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x)

    # Flatten layer.
    x = Flatten()(x)

    # 1st dense layer.
    x = Dense(4927, activation='relu')(x)
    x = Dropout(0.48599073736368)(x)

    # 2nd dense layer.
    x = Dense(2785, activation='relu')(x)
    x = Dropout(0.48599073736368)(x)

    # 3rd dense layer.
    x = Dense(1574, activation='relu')(x)
    x = Dropout(0.48599073736368)(x)

    output_tensor = Dense(num_fgs, activation='sigmoid')(x)
    print('Model Construction')
    model = Model(inputs=input_tensor, outputs=output_tensor)
    model.summary()
    optimizer = Adam()

    if weighted == 1:

        def calculate_class_weights(y_true):
            number_dim = np.shape(y_true)[1]
            weights = np.zeros((2, number_dim))
            # Calculates weights for each label in a for loop.
            for i in range(number_dim):
                weights_n, weights_p = (y_train.shape[0]/(2 * (y_train[:,i] == 0).sum())), (y_train.shape[0]/(2 * (y_train[:,i] == 1).sum()))
                # Weights could be log-dampened to avoid extreme weights for extremly unbalanced data.
                weights[1, i], weights[0, i] = weights_p, weights_n

            return weights.T

        def get_weighted_loss(weights):
            def weighted_loss(y_true, y_pred):
                return K.mean((weights[:,0]**(1.0-y_true))*(weights[:,1]**(y_true))*K.binary_crossentropy(y_true, y_pred), axis=-1)
            return weighted_loss

        model.compile(optimizer=optimizer, loss=get_weighted_loss(calculate_class_weights(y_train)))

    else:

        model.compile(optimizer=optimizer, loss='binary_crossentropy')

    def custom_learning_rate_schedular(epoch):
        if epoch < 31:
            return 2.5e-4
        elif 31 <= epoch < 37:
            return 2.5000001187436283e-05
        elif 37 <= epoch < 42:
            return 2.5000001187436284e-06

    print('Start training')
    # Start training.

    prediction = model.predict(X_test)
    return (prediction > 0.5).astype(int)
    


def interpolate_to_600(spec):
    

    old_x = np.arange(len(spec))
    new_x = np.linspace(min(old_x), max(old_x), 600)

    interp = interp1d(old_x, spec)
    new_spec = interp(new_x)
    return new_spec

def make_msms_spectrum(spectrum):
    msms_spectrum = np.zeros(10000)
    for peak in spectrum:
        peak_pos = int(peak[0]*10)
        if peak_pos >= 10000:
            peak_pos = 9999

        msms_spectrum[peak_pos] = peak[1]
    
    return msms_spectrum


@click.command()
@click.option("--analytical_data", type=click.Path(exists=True, path_type=Path), required=False)
@click.option("--out_path", type=click.Path(exists=True, path_type=Path), required=False)
@click.option("--column", type=str, required=False)
@click.option("--seed", type=int, default=3245)
def main(analytical_data, out_path, column, seed):

    print("Loading Data")
    training_data = None

    if column == 'pos_msms':
        column = 'msms_positive_40ev'
    elif column == 'neg_msms':
        column = 'msms_negative_40ev'

    for i, parquet_file in enumerate(analytical_data.glob("*.parquet")):
        data = pd.read_parquet(parquet_file, columns=[column, 'smiles'])

        if 'msms' in column:
            data[column] = data[column].map(make_msms_spectrum)

        data['func_group'] = data.smiles.map(get_functional_groups)
        data[column] = data[column].map(interpolate_to_600)

        if training_data is None:
            training_data = data
        else:
            training_data = pd.concat((training_data, data))
        del data

        print("Loaded Data: ", i)

    train, test = train_test_split(training_data, test_size=0.1, random_state=seed) 

    X_train = np.stack(train[column].to_list())
    y_train = np.stack(train['func_group'].to_list())
    X_test = np.stack(test[column].to_list())
    y_test = np.stack(test['func_group'].to_list())

    # Train extended model.
    prediction = train_model(X_train, y_train, X_test, 37, 'e', 0, 0)

    print(f1_score(y_test, prediction, average='micro'))

    with open(out_path / "results.pickle", "wb") as file:
        pickle.dump({'pred': prediction, 'tgt': y_test}, file)


if __name__ == '__main__':
    main()
    
