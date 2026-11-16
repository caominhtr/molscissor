import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import CanonSmiles, PandasTools, DataStructs, AllChem
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
import gc
from tqdm import tqdm
import os
import itertools
from numba import njit, jit
import math
from ortools.algorithms.python import knapsack_solver
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

def read_smiles_file(file_target, cluster=False):
    with open(file_target, 'r') as f:
        first_line = f.readline()

    if ',' in first_line:
        sep = ','
    elif '\t' in first_line:
        sep = '\t'
    else:
        sep = ' '  
    
    if cluster:
        df = pd.read_csv(
            file_target,
            sep=sep,
            header=None,
            names=['SMILES', 'ID', 'Cluster']
        )
    else:
        df = pd.read_csv(
            file_target,
            sep=sep,
            header=None,
            names=['SMILES', 'ID']
        )
    return df


def canon_smiles(smiles):
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        return Chem.CanonSmiles(smiles)
    except Exception:
        return None

def read_dataframe(df):
    df_new = df.copy()
    df_new["SMILES"] = df_new["SMILES"].apply(canon_smiles)
    df_new = df_new.dropna(subset=["SMILES"]).reset_index(drop=True)
    PandasTools.AddMoleculeColumnToFrame(df_new, smilesCol="SMILES", molCol="Structure")

    df_new = df_new[df_new["Structure"].notna()].reset_index(drop=True)

    df_new["Morgan"] = df_new["Structure"].apply(
        lambda x: AllChem.GetMorganFingerprintAsBitVect(x, radius=2, nBits=2048)
    )

    return df_new


def write_file(outdir, name_file, content):
    file_destination = os.path.join(outdir, name_file)

    with open(file_destination, "w") as f:
        f.write(f"{content}")

@njit
def find(parent, i):
    if parent[i] == i:
        return i
    parent[i] = find(parent, parent[i])
    return parent[i]
@njit
def union(parent, i, j):
    root_i = find(parent, i)
    root_j = find(parent, j)
    if root_i != root_j:
        parent[root_i] = root_j



class linkage_clustering:
    def __init__(self):
        pass

    '''
    Tanimoto based on Morgan 2 2048
    '''

    def get_clusters_tanimoto(self, df, threshold):
        fps = df["Morgan"].tolist()
        num_fps = len(fps)

        assigned = np.zeros(num_fps, dtype=bool)
        clusters = []
        unassigned_indices = np.arange(num_fps)

        for i in tqdm(range(num_fps)):
            if assigned[i]:
                continue

            remaining_mask = ~assigned
            remaining_indices = unassigned_indices[remaining_mask]
            remaining_fps = [fps[idx] for idx in remaining_indices]

            current_fp = fps[i]
            sims = np.array(DataStructs.BulkTanimotoSimilarity(current_fp, remaining_fps), dtype=np.float32)
            relative_hits = np.where(sims >= threshold)[0]
            cluster_indices = remaining_indices[relative_hits]

            clusters.append(cluster_indices.tolist())
            assigned[cluster_indices] = True

            if i % 1000 == 0:
                gc.collect()
                
        return clusters, [len(c) for c in clusters]

    '''
    Merge
    '''
    def merge_overlapping_clusters(self, df, clusters, threshold, global_min=None, global_max=None, feat_scaled=None):
        n = len(clusters)
        parent = np.arange(n, dtype=np.int32)
        all_fps = df["Morgan"].tolist() 
        

        for i in tqdm(range(n)):
            fps_i = [all_fps[idx] for idx in clusters[i] if all_fps[idx] is not None]

            for j in range(i):
                if find(parent, i) == find(parent, j):
                    continue

                found_match = False

                fps_j = [all_fps[idx] for idx in clusters[j] if all_fps[idx] is not None]
                for fp in fps_i:
                    sims = DataStructs.BulkTanimotoSimilarity(fp, fps_j)
                    if any(s >= threshold for s in sims):
                        found_match = True
                        break

                if found_match:
                    union(parent, i, j)

            del fps_i

        new_groups = {}
        for i in range(n):
            root = find(parent, i)
            if root not in new_groups:
                new_groups[root] = []
            new_groups[root].extend(clusters[i])

        final_clusters = [list(set(indices)) for indices in new_groups.values()]
        final_lens = [len(c) for c in final_clusters]

        return final_clusters, final_lens


class allocation:
    def __init__(self):
            pass

    def allocate_normal(self, list_cluster, len_cluster, n, num_cpd):
        '''
        Partition1 (p1): training set
        Partition2 (p2): test set
        '''
        n = int(n)
        len_cluster = np.asarray(len_cluster)

        max_len = len_cluster.max()
        limit_test = num_cpd / n
        limit_train_max = (n * num_cpd) / (n + 1)

        if max_len > limit_train_max:
            return None, None

        mask_over = (len_cluster > limit_test) & (len_cluster <= limit_train_max)
        mask_below = len_cluster <= limit_test

        if len_cluster[mask_over].sum() > limit_train_max:
            return None, None

        idx_below = np.where(mask_below)[0]
        sort_val_idx = np.argsort(len_cluster[idx_below], kind='stable')[::-1]
        sorted_idx = idx_below[sort_val_idx]

        selected_idx = sorted_idx[::(n + 1)]

        test_indice = []
        cumsum = 0.0

        for idx in selected_idx:
            l = len_cluster[idx]
            if cumsum + l > limit_test:
                break
            test_indice.extend(list_cluster[idx])
            cumsum += l

        test_set = set(test_indice)
        train_indice = [i for i in range(num_cpd) if i not in test_set]

        if len(test_indice) < num_cpd / (n+1):
            return None, None

        return train_indice, test_indice

    def neighbor_fit(self, reference_feat, n_reference):

        scale = StandardScaler()
        ref_feat_scaled = scale.fit_transform(reference_feat[n_reference])
        nn = NearestNeighbors(n_neighbors=len(n_reference), algorithm='ball_tree')
        nn.fit(ref_feat_scaled)

        return nn, scale


    def allocate_knapsack(self, list_cluster_query, len_cluster_query, n_query, query_feat, fold, nn, scale):
        len_cluster_arr = np.array(len_cluster_query)
        threshold = int(n_query // fold)
    
        mask_below = len_cluster_arr <= threshold
        indices_below = np.where(mask_below)[0]
        
        if len(indices_below) == 0:
            return list(range(n_query)), []

        sub_list_cluster = [list_cluster_query[i] for i in indices_below]

        sim_scores = np.array([np.max(nn.kneighbors(scale.transform(query_feat[cluster]))[0].max(axis = 1))
            for cluster in sub_list_cluster], dtype = 'int64')

        max_val = np.max(sim_scores) + 1
        profits = (max_val - sim_scores).tolist()
        weights = [len_cluster_arr[i].item() for i in indices_below] 
        capacity = [threshold]

        solver = knapsack_solver.KnapsackSolver(
            knapsack_solver.SolverType.KNAPSACK_DYNAMIC_PROGRAMMING_SOLVER, 
            'ClusterSelection'
        )
        
        solver.init(profits, [weights], capacity)
        solver.solve()


        test_new_list = []
        for i in range(len(profits)):
            if solver.best_solution_contains(i):
                test_new_list.extend(sub_list_cluster[i])
        
        test_new = np.array(test_new_list, dtype=np.int64)

        train_mask = np.ones(n_query, dtype=bool)
        if len(test_new) > 0:
            train_mask[test_new] = False
        
        train_new = np.where(train_mask)[0].tolist()

        return train_new, test_new.tolist()


    def allocate_cycle(self, list_cluster_A, len_cluster_A, list_cluster_I, len_cluster_I, n_active, n_inactive, size_number, active_feat_matrix, inactive_feat_matrix, cycle = 1):
        aT_init, aV_init = self.allocate_normal(list_cluster_A, len_cluster_A, size_number, n_active)

        try:
            for i in range(cycle):
                nn_a, scale_a = self.neighbor_fit(active_feat_matrix, aV_init)
                iT, iV = self.allocate_knapsack(list_cluster_I,
                                        len_cluster_I,
                                        n_inactive,
                                        inactive_feat_matrix,
                                        size_number,
                                        nn_a, scale_a)
                del nn_a, scale_a

                nn_i, scale_i = self.neighbor_fit(inactive_feat_matrix, iV)
                aT, aV = self.allocate_knapsack(list_cluster_A,
                                        len_cluster_A,
                                        n_active,
                                        active_feat_matrix,
                                        size_number,
                                        nn_i, scale_i)
                
                del nn_i, scale_i

                aT_init = aT
                aV_init = aV

                gc.collect()
        
            return aT, aV, iT, iV
        
        except:
            return None, None, None, None