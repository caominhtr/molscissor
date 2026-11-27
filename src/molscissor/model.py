import argparse
import numpy as np
import pandas as pd
import os
import random
import gc

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem import Descriptors
from rdkit.Chem.Fingerprints import FingerprintMols

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import math
from oddt.metrics import bedroc

os.environ['TF_USE_LEGACY_KERAS']='1'


parser = argparse.ArgumentParser(description='')
parser.add_argument('-aTrain', required=True, help='Path to active training SMILES file')
parser.add_argument('-iTrain', required=True, help='Path to inactive training SMILES file')
parser.add_argument('-aTest', required=True, help='Path to active test SMILES file')
parser.add_argument('-iTest', required=True, help='Path to inactive test SMILES file')

parser.add_argument('-model', required = True, help='RF, SVC, XGB, graphconv, chemberta')
parser.add_argument('-feature', default=None, help='morgan, torsion, daylight, rdkit')
parser.add_argument('-seed', required = True)

parser.add_argument('-id', default=None, help='ID')
parser.add_argument('-o', required=True, help='Output')
args = parser.parse_args()


def read_smiles_file(file_target):
    with open(file_target, 'r') as f:
        first_line = f.readline()

    if ',' in first_line:
        sep = ','
    elif '\t' in first_line:
        sep = '\t'
    else:
        sep = r' '

    df = pd.read_csv(
        file_target,
        sep=sep,
        header=None,
        names=['SMILES', 'ID'],
        engine='python'
    )
    return df



DESCRIPTOR_FUNCS = [func for _, func in Descriptors._descList]
NUM_DESCRIPTORS = len(DESCRIPTOR_FUNCS)

def get_descriptors_array(df):
    smiles_list = df['SMILES'].tolist()
    num_mols = len(smiles_list)
 
    desc_matrix = np.full((num_mols, NUM_DESCRIPTORS), np.nan, dtype=np.float32)
    
    for idx, smal in enumerate(smiles_list):
        if not isinstance(smal, str):
            continue
        mol = Chem.MolFromSmiles(smal)
        if mol is not None:
            desc_matrix[idx] = [func(mol) for func in DESCRIPTOR_FUNCS]
            
    return desc_matrix


def cal_rdkit(train, test):
    train_arr = get_descriptors_array(train)

    scaler = StandardScaler(copy=False)
    train_scaled = scaler.fit_transform(train_arr)
    
    del train, train_arr
    gc.collect()

    test_arr = get_descriptors_array(test)

    test_scaled = scaler.transform(test_arr)
   
    del test, test_arr
    gc.collect()

    return train_scaled, test_scaled

def generate_morgan(df, rad=2, bit_len=2048, smiles_col="SMILES"):
    fps = []

    for _, row in df.iterrows():
        item = row[smiles_col]
        mol = Chem.MolFromSmiles(item) if isinstance(item, str) else item

        if mol is not None:
            fp = np.array(
                AllChem.GetMorganFingerprintAsBitVect(mol, rad, nBits=bit_len)
            )
            fps.append(fp)

    return pd.DataFrame(np.array(fps), dtype=np.uint8)

def generate_daylight(df, bit_len=2048, smiles_col='SMILES'):
    n_samples = len(df)
    fps_array = np.zeros((n_samples, bit_len), dtype=np.uint8)
    
    for i, smiles in enumerate(df[smiles_col]):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                bv = FingerprintMols.FingerprintMol(mol)
                temp = tuple(bv.GetOnBits())
                if len(temp) > 0:
                    fps_array[i, np.array(temp)] = 1
        except Exception:
            pass # leaves as zeros if failed
            
    return pd.DataFrame(fps_array)


def generate_torsion(df, bit_len=2048, smiles_col='SMILES'):
    n_samples = len(df)
    
    fps_array = np.zeros((n_samples, bit_len), dtype=np.uint8)
    
    for i, smiles in enumerate(df[smiles_col]):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            bv = Chem.rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, bit_len)
            DataStructs.ConvertToNumpyArray(bv, fps_array[i])
            
    return pd.DataFrame(fps_array)


def cal_daylight(train_init, test_init):
    return generate_daylight(train_init), generate_daylight(test_init)

def cal_morgan(train_init, test_init, rad):
    train = generate_morgan(train_init, rad=rad)
    test = generate_morgan(test_init, rad=rad)

    return train, test

def cal_torsion(train_init, test_init):
    train = generate_torsion(train_init)
    test = generate_torsion(test_init)

    return train, test

def NEF(y_true, y_prob, x):
    df = pd.DataFrame({'True': y_true, 'Prob': y_prob})
    N = len(df)
    na = df['True'].sum() 
    x = int(x)
    thres = math.floor(N * x/100)
    
    max_value = 100/x if (na / N <= x/100) else (N / na)
    
    df_sorted_top = df.sort_values('Prob', ascending=False).head(thres)

    na_top = df_sorted_top['True'].sum()
    ef = (na_top / na) * (100/x)
    
    return ef / max_value

def compute_metrics(y_true, y_probs):
    auc = roc_auc_score(y_true, y_probs)

    nef_1 = NEF(y_true, y_probs, 1)

    df = pd.DataFrame({'True': y_true, 'Pred': y_probs})
    df = df.sort_values('Pred', ascending=False, kind='stable').reset_index(drop=True)

    bedroc_1 = bedroc(df['True'], df['Pred'], alpha=160.9)

    return [auc, nef_1, bedroc_1]

active_T = read_smiles_file(args.aTrain)
inactive_T = read_smiles_file(args.iTrain)
active_V = read_smiles_file(args.aTest)
inactive_V = read_smiles_file(args.iTest)

df_train = pd.concat([active_T, inactive_T]).reset_index(drop = True)
df_train['Label'] = np.concatenate([np.repeat(1, active_T.shape[0]),
                                    np.repeat(0, inactive_T.shape[0])])

df_test = pd.concat([active_V, inactive_V]).reset_index(drop = True)
df_test['Label'] = np.concatenate([np.repeat(1, active_V.shape[0]),
                                    np.repeat(0, inactive_V.shape[0])])



train_smiles = df_train["SMILES"].astype(str).tolist()
y_train = df_train["Label"].to_numpy()

test_smiles = df_test["SMILES"].astype(str).tolist()
y_test = df_test["Label"].to_numpy()


if args.feature == 'morgan':
    X_train_feature, X_test_feature = cal_morgan(df_train, df_test, 2)
elif args.feature == 'daylight':
    X_train_feature, X_test_feature = cal_daylight(df_train, df_test)
elif args.feature =='torsion':
    X_train_feature, X_test_feature = cal_torsion(df_train, df_test)   
elif args.feature == 'rdkit': 
    X_train_feature, X_test_feature = cal_rdkit(df_train, df_test) 
else:
    X_train_feature, X_test_feature = None, None



def train_and_predict(
    model_name,
    train_smiles,
    y_train,
    test_smiles,
    y_test,
    X_train_feature=None,
    X_test_feature=None
    ):

    if model_name in ["RF", "SVC", "XGB"]:
        if model_name == "RF":
            model = RandomForestClassifier(random_state=int(args.seed))
        elif model_name == "SVC":
            model = SVC(probability=True, random_state=int(args.seed))
        elif model_name == "XGB":
            model = XGBClassifier(random_state=int(args.seed))

        model.fit(X_train_feature, y_train)
        y_pred = model.predict_proba(X_test_feature)[:, 1]

        return y_pred

    elif model_name == 'MLP':
        import torch
        from DeepPurpose import CompoundPred
        from DeepPurpose.utils import generate_config, smiles2morgan, smiles2daylight, smiles2torsion
        

        def set_seed(seed):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        def drug(df_data, drug_encoding):
            if drug_encoding == 'morgan':
                unique = pd.Series(df_data['SMILES'].unique()).apply(smiles2morgan)
                unique_dict = dict(zip(df_data['SMILES'].unique(), unique))
                df_data['drug_encoding'] = [unique_dict[i] for i in df_data['SMILES']]
            elif drug_encoding == 'torsion':
                unique = pd.Series(df_data['SMILES'].unique()).apply(smiles2torsion)
                unique_dict = dict(zip(df_data['SMILES'].unique(), unique))
                df_data['drug_encoding'] = [unique_dict[i] for i in df_data['SMILES']]
            elif drug_encoding == 'daylight':
                unique = pd.Series(df_data['SMILES'].unique()).apply(smiles2daylight)
                unique_dict = dict(zip(df_data['SMILES'].unique(), unique))
                df_data['drug_encoding'] = [unique_dict[i] for i in df_data['SMILES']]
                
            return df_data

        df_train = pd.DataFrame({'SMILES': train_smiles, 
                                 'Label': y_train})
        df_test = pd.DataFrame({'SMILES': test_smiles, 
                                'Label': y_test})
        set_seed(int(args.seed))

        if args.feature in ['morgan','torsion','daylight']:
            df_train = drug(df_train, args.feature)
            df_test = drug(df_test, args.feature)
            config = generate_config(drug_encoding = args.feature, input_dim_drug=2048)
            

        elif args.feature == 'rdkit':
            df_train['drug_encoding'] = list(X_train_feature)
            df_test['drug_encoding'] = list(X_test_feature)
            config = generate_config(drug_encoding='rdkit_2d_normalized',input_dim_drug=217)

        model = CompoundPred.model_initialize(**config)
        model.train(df_train, df_train)

        y_pred = model.predict(df_test)

        return y_pred

    elif model_name == 'graphconv':
        import deepchem as dc
        import torch
        def set_global_seed(seed: int):
            random.seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        seed = int(args.seed)
        set_global_seed(seed)

        featurizer = dc.feat.ConvMolFeaturizer()
        X_tr = featurizer.featurize(train_smiles)
        v_tr = np.array([x is not None for x in X_tr])
        train_ds = dc.data.NumpyDataset(X=X_tr[v_tr], y=y_train[v_tr])

        X_te = featurizer.featurize(test_smiles)
        v_te = np.array([x is not None for x in X_te])
        test_ds = dc.data.NumpyDataset(X=X_te[v_te], y=y_test[v_te])

        model = dc.models.GraphConvModel(
            n_tasks=1,
            mode="classification",
            graphconv_layers=[512,512],
            dense_layer_size=2048,
            batch_size=256
        )

        model.fit(train_ds, nb_epoch=10)
        preds = model.predict(test_ds)
        y_pred = preds[:, 0, 1]
        return y_pred

    elif model_name.lower() in ['chemberta']:
        import torch
        from torch.utils.data import Dataset as TorchDataset
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            TrainingArguments,
            Trainer,
            DataCollatorWithPadding
        )
        def set_global_seed(seed: int):
            random.seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        seed = int(args.seed)
        set_global_seed(seed)

        def filter_and_canonicalize(smiles_list, labels=None):
            valid_smiles = []
            valid_labels = [] if labels is not None else None
            
            for idx, smile in enumerate(smiles_list):
                if isinstance(smile, str) and smile:
                    mol = Chem.MolFromSmiles(smile)
                    if mol is not None:
                        canonical = Chem.MolToSmiles(mol, canonical=True)
                        valid_smiles.append(canonical)
                        if labels is not None:
                            valid_labels.append(labels[idx])
                            
            if labels is not None:
                return valid_smiles, valid_labels
            return valid_smiles


        clean_train_smiles, clean_y_train = filter_and_canonicalize(train_smiles, y_train)
        clean_test_smiles = filter_and_canonicalize(test_smiles)

        model_ckpt = f'DeepChem/ChemBERTa-77M-MTR'
        tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

        class ChemBERTaDataset(TorchDataset):
            def __init__(self, smiles_list, labels=None):
                self.encodings = tokenizer(
                    smiles_list,
                    truncation=True,
                    max_length=512,
                    padding=False
                )
                self.labels = labels

            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                if self.labels is not None:
                    item['labels'] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
                return item

            def __len__(self):
                return len(self.encodings['input_ids'])

        train_ds = ChemBERTaDataset(clean_train_smiles, clean_y_train)
        test_ds = ChemBERTaDataset(clean_test_smiles)

        model = AutoModelForSequenceClassification.from_pretrained(model_ckpt,
                                                                   num_labels=2,
                                                                   problem_type="single_label_classification")
        device_available = torch.cuda.is_available()

        training_args = TrainingArguments(
            output_dir=f"./results_chemberta",
            eval_strategy="no",
            save_strategy="no",
            learning_rate=5e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=256,
            num_train_epochs=3,
            seed=seed,
            data_seed=seed,
            fp16=device_available,             
            dataloader_pin_memory=device_available, 
            logging_steps=50,
            report_to="none"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            processing_class=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
        )

        trainer.train()

        raw_preds = trainer.predict(test_ds)
        test_logits = raw_preds.predictions
        test_probs = torch.softmax(torch.tensor(test_logits), dim=-1)[:, 1].numpy()

        del model, trainer, train_ds, test_ds
        if device_available:
            torch.cuda.empty_cache()
        gc.collect()

        return test_probs
        


y_pred = train_and_predict(args.model,
                            train_smiles,
                            y_train,
                            test_smiles,
                            y_test,
                            X_train_feature,
                            X_test_feature)


A=compute_metrics(y_test, y_pred)

os.makedirs(args.o, exist_ok=True)
file_metric = os.path.join(args.o, "Metrics.txt")

with open(file_metric, 'w') as f:
    f.write(f"{args.id}," + ",".join(str(v) for v in A) + f",{args.seed}" "\n")
