import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd
from rdkit import DataStructs, Chem
from rdkit.Chem import AllChem, PandasTools
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from multiprocessing import Pool
import gc


parser = argparse.ArgumentParser(description='Calculate metrics')
parser.add_argument('-aTrain', required=True, help='Train Active')
parser.add_argument('-iTrain', required=True, help='Train Inactive')
parser.add_argument('-aTest', required=True, help='Test Active')
parser.add_argument('-iTest', required=True, help='Test Inactive')

parser.add_argument('-metric', required=True, help='AVE, maxTani, scaledlpi')

parser.add_argument('-o', required=True, help='path file')
args = parser.parse_args()

aTrain = args.aTrain
iTrain = args.iTrain 
aTest = args.aTest
iTest = args.iTest
metric = args.metric
output_path = args.o

def read_smiles_file(file_target):
    with open(file_target, 'r') as f:
        first_line = f.readline()

    if ',' in first_line:
        sep = ','
    elif '\t' in first_line:
        sep = '\t'
    else:
        sep = ' '  

    df = pd.read_csv(
        file_target,
        sep=sep,
        header=None,
        names=['SMILES', 'ID']
    )
    return df

if metric=='AVE':
    def get_fp(smiles_list):
        fps = []
        for x in smiles_list:
            mol = Chem.MolFromSmiles(x)
            if mol is not None:
                fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))
            else:
                continue
                
        return fps
        

    def Cdist(params):
        query_fps, ref_fps = params
        min_distances = []

        for q in query_fps:
            max_sim = max(DataStructs.BulkTanimotoSimilarity(q, ref_fps))
            min_distances.append(1.0 - max_sim)

        return np.array(min_distances, dtype=np.float64)



    def getMinDistances(fp1, fp2, numWorkers=1):
        if numWorkers > 1:
            pool = Pool(processes=numWorkers)
            interval = max(1, len(fp1) // numWorkers)
            tasks = [(fp1[r : r + interval], fp2) for r in range(0, len(fp1), interval)]
            chunks = list(tqdm(pool.imap(Cdist, tasks), total=len(tasks), desc="Calculating Min Dists", leave=False))
            pool.close()
            pool.join()
            return np.concatenate(chunks)
        else:
            return Cdist((fp1, fp2))

    def calculate_S(min_dist_array):
        return np.mean([np.mean(min_dist_array < t) for t in np.linspace(0, 1.0, 50)])

    active_T = read_smiles_file(aTrain)
    active_V = read_smiles_file(aTest)
    inactive_T = read_smiles_file(iTrain)
    inactive_V = read_smiles_file(iTest)

    activesTrain = [m for m in active_T['SMILES'] if m is not None]
    inactivesTrain = [m for m in inactive_T['SMILES'] if m is not None]
    activesTest = [m for m in active_V['SMILES'] if m is not None]
    inactivesTest = [m for m in inactive_V['SMILES'] if m is not None]

    activesTrainFP = get_fp(activesTrain)
    inactivesTrainFP = get_fp(inactivesTrain)
    activesTestFP = get_fp(activesTest)
    inactivesTestFP = get_fp(inactivesTest)

    aTest_aTrain_minD = getMinDistances(activesTestFP, activesTrainFP)
    print('Finish 1/4')
    aTest_iTrain_minD = getMinDistances(activesTestFP, inactivesTrainFP)
    print('Finish 2/4')
    iTest_iTrain_minD = getMinDistances(inactivesTestFP, inactivesTrainFP)
    print('Finish 3/4')
    iTest_aTrain_minD = getMinDistances(inactivesTestFP, activesTrainFP)
    print('Finish 4/4')


    aTest_aTrain_S = calculate_S(aTest_aTrain_minD)
    aTest_iTrain_S = calculate_S(aTest_iTrain_minD)
    iTest_iTrain_S = calculate_S(iTest_iTrain_minD)
    iTest_aTrain_S = calculate_S(iTest_aTrain_minD)

    with open(args.o, 'w') as outFH:
        print(
            "#ActTrain=", len(activesTrainFP), "#InactTrain=", len(inactivesTrainFP),
            "#ActTest=", len(activesTestFP), "#InactTest=", len(inactivesTestFP),
            "AA-AI=", aTest_aTrain_S - aTest_iTrain_S,
            "II-IA=", iTest_iTrain_S - iTest_aTrain_S,
            "(AA-AI)+(II-IA)=", aTest_aTrain_S - aTest_iTrain_S + iTest_iTrain_S - iTest_aTrain_S,
            file=outFH
        )


elif metric=='maxTani':
    def process_data(file_target):
        df = read_smiles_file(file_target)
        PandasTools.AddMoleculeColumnToFrame(df, 'SMILES', 'Structure')
        df = df[~df['Structure'].isna()].reset_index(drop=True)
        df['Morgan'] = df['Structure'].apply(lambda x: GetMorganFingerprintAsBitVect(x, 2, 2048))
        return df


    def get_max_tanimoto(train_fps, test_fps):
        max_sim = 0.0

        for fp in tqdm(train_fps):
        
            scores = DataStructs.BulkTanimotoSimilarity(fp, test_fps)
            current_max = max(scores)
            if current_max > max_sim:
                max_sim = current_max
                if max_sim == 1.0: 
                    break

        return max_sim


    active_T = process_data(aTrain)
    active_V = process_data(aTest)
    inactive_T = process_data(iTrain)
    inactive_V = process_data(iTest)

    act_train_fps = active_T['Morgan'].tolist()
    act_test_fps = active_V['Morgan'].tolist()
    inact_train_fps = inactive_T['Morgan'].tolist()
    inact_test_fps = inactive_V['Morgan'].tolist()

    score_active = get_max_tanimoto(act_train_fps, act_test_fps)
    score_inactive = get_max_tanimoto(inact_train_fps, inact_test_fps)

    with open(output_path, 'w') as f:
        f.write(f"Max Active Tanimoto: {score_active}\n")
        f.write(f"Max Inactive Tanimoto: {score_inactive}\n")

elif metric=='scaledlpi':

    def smiles_to_fingerprints(smiles_list: list, radius: int = 2, n_bits: int = 2048):
        fps = []
        valid_indices = []

        for idx, smiles in enumerate(smiles_list):
            if not isinstance(smiles, str) or not smiles.strip():
                continue
            mol = Chem.MolFromSmiles(str(smiles).strip())
            if mol is not None:
                fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits))
                valid_indices.append(idx)

        return fps, valid_indices


    def compute_scaled_datasail_leakage_eq20(
        train_smiles: list,
        test_smiles: list,
        kappa_train: list = None,
        kappa_test: list = None,
        batch_size: int = 1000,
    ) -> float:
        """
        Computes Scaled/Relative DataSAIL Leakage L_rel(pi) [Equation 20]:

                    sum_{x in Train, x' in Test} sim(x, x') * kappa(x) * kappa(x')
        L_rel(pi) = ------------------------------------------------------------------
                                    sum_{x, x' in D} sim(x, x')
        """
        train_fps, valid_tr = smiles_to_fingerprints(train_smiles)
        test_fps, valid_te = smiles_to_fingerprints(test_smiles)

        N_train, N_test = len(train_fps), len(test_fps)

        if kappa_train is None:
            kappa_tr = np.ones(N_train, dtype=np.float64)
        else:
            kappa_tr = np.array(kappa_train, dtype=np.float64)[valid_tr]

        if kappa_test is None:
            kappa_te = np.ones(N_test, dtype=np.float64)
        else:
            kappa_te = np.array(kappa_test, dtype=np.float64)[valid_te]

        cross_leakage_weighted = 0.0
        cross_sim_unweighted = 0.0

        for i in range(0, N_train, batch_size):
            train_chunk = train_fps[i : i + batch_size]
            kappa_tr_chunk = kappa_tr[i : i + batch_size]

            chunk_sims = np.zeros((len(train_chunk), N_test), dtype=np.float32)
            for row_idx, train_fp in enumerate(train_chunk):
                chunk_sims[row_idx, :] = DataStructs.BulkTanimotoSimilarity(train_fp, test_fps)

            chunk_kappa_prod = np.outer(kappa_tr_chunk, kappa_te)
            cross_leakage_weighted += np.sum(chunk_sims * chunk_kappa_prod)

            cross_sim_unweighted += np.sum(chunk_sims)

            del chunk_sims, chunk_kappa_prod
            gc.collect()

        train_internal_sim = 0.0
        for i in range(0, N_train, batch_size):
            chunk_a = train_fps[i : i + batch_size]
            for fp_a in chunk_a:
                train_internal_sim += np.sum(DataStructs.BulkTanimotoSimilarity(fp_a, train_fps))

        test_internal_sim = 0.0
        for i in range(0, N_test, batch_size):
            chunk_b = test_fps[i : i + batch_size]
            for fp_b in chunk_b:
                test_internal_sim += np.sum(DataStructs.BulkTanimotoSimilarity(fp_b, test_fps))

        total_unweighted_sim = train_internal_sim + test_internal_sim + (2.0 * cross_sim_unweighted)

        del train_fps, test_fps
        gc.collect()

        return float(cross_leakage_weighted / total_unweighted_sim) if total_unweighted_sim > 0 else 0.0

    train_active_df = read_smiles_file(args.aTrain)
    train_inactive_df = read_smiles_file(args.iTrain)
    test_active_df = read_smiles_file(args.aTest)
    test_inactive_df = read_smiles_file(args.iTest)

    train_df = pd.concat([train_active_df, train_inactive_df]).reset_index(drop = True)
    test_df = pd.concat([test_active_df, test_inactive_df]).reset_index(drop=True)

    train_smiles = train_df["SMILES"].tolist()
    test_smiles = test_df["SMILES"].tolist()

    scaled_leakage = compute_scaled_datasail_leakage_eq20(
        train_smiles=train_smiles,
        test_smiles=test_smiles,
    )

    out_df = pd.DataFrame([{"Scaled_DataSAIL_Leakage_Eq20": scaled_leakage}])
    out_df.to_csv(args.o, index=False, header = False)

else:
    print("Metric should be AVE, maxTani, or scaledlpi")