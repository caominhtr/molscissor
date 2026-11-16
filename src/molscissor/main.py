import argparse
import os
import warnings
import subprocess
import tempfile
import pandas as pd
import numpy as np
import optuna
import pickle
from optuna.trial import TrialState


warnings.filterwarnings("ignore", category=DeprecationWarning)

# from . import new_AVE_cal
# from . import new_DOE_cal
# from . import new_max_tanimoto
# from . import new_model

from .decoy_utils import calc_dataset_props_all, doe_score
from .utils import read_smiles_file,read_dataframe
from .utils import linkage_clustering
from .utils import allocation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-active", required=True, help="Input SMILES file for active (.smi)")
    parser.add_argument("-inactive", required=True, help="Input SMILES file for inactive (.smi)")
    parser.add_argument("-algo", required=True, type=str, help="clustering algorithm: molscissor, bitbirch, other")

    parser.add_argument("-no_tune", action="store_false", help="(only for MolSCISSOR) disable tuning")
    parser.add_argument("-continue_tune", action="store_true", help="(only for MolSCISSOR) continue tuning from last trial")

    parser.add_argument("-upperActive", type=float, default=0.7, help="(for tune only) MolSCISSOR upper similarity for Actives")
    parser.add_argument("-upperInactive", type=float, default=0.7, help="(for tune only) MolSCISSOR upper similarity for Inactives")

    parser.add_argument("-thresholdActive", type=float, help="(for no tune) user-defined threshold for Actives")
    parser.add_argument("-thresholdInactive", type=float, help="(for no tune) user-defined threshold for Inactives")

    parser.add_argument("-k", type=int, default=4, help="Ratio of Train/Test from k:1 to (k-1):1")
    parser.add_argument("-cycle", type=int, default=1, help="Number of cycles for allocation")
    parser.add_argument("-iter", type=int, default=100, help="Number of Optuna optimization iterations")
    parser.add_argument("-seed", type=int, default=42, help="Seed for reproduction")
    parser.add_argument("-namestudy", type=str, default=f'job_0', help="Name of the Optuna SQLite database file/ bitbirch folder")

    parser.add_argument("-outdir", required=True, help="Output directory")
    args = parser.parse_args()

    smi_file_active = args.active
    smi_file_inactive = args.inactive
    
    upper_active = args.upperActive
    upper_inactive = args.upperInactive

    threshold_active = args.thresholdActive
    threshold_inactive = args.thresholdInactive
    
    algorithm = args.algo
    size_number = args.k
    cycle = args.cycle
    iteration = args.iter
    seed = args.seed
    name_study  = args.namestudy
    outdir = args.outdir

    #########################################
    if algorithm in ['molscissor', 'bitbirch']:
        df_active = read_smiles_file(smi_file_active)
        df_inactive = read_smiles_file(smi_file_inactive)
    elif algorithm == 'other':
        df_active = read_smiles_file(smi_file_active, cluster=True)
        df_inactive = read_smiles_file(smi_file_inactive, cluster=True)


    current_scenario = allocation()

    ###########################################

    if algorithm=='molscissor':

        df_new_active = read_dataframe(df_active)
        active_feat_matrix = calc_dataset_props_all(df_new_active)

        df_new_inactive = read_dataframe(df_inactive)
        inactive_feat_matrix = calc_dataset_props_all(df_new_inactive)

        n_active = df_new_active.shape[0]
        n_inactive = df_new_inactive.shape[0]

        clustering = linkage_clustering()

        if not args.no_tune:

            final_thres_active = threshold_active
            final_thres_inactive = threshold_inactive

            if final_thres_active is None or final_thres_inactive is None:
                print("Please provide thresholds for each class")

            else:
                clusters_active_1, len_clusters_A_1 = clustering.get_clusters_tanimoto(df_new_active, final_thres_active)
                list_cluster_A, len_cluster_A = clustering.merge_overlapping_clusters(df_new_active, clusters_active_1, final_thres_active)

                del clusters_active_1, len_clusters_A_1

                clusters_inactive_1, len_clusters_I_1 = clustering.get_clusters_tanimoto(df_new_inactive, final_thres_inactive)
                list_cluster_I, len_cluster_I = clustering.merge_overlapping_clusters(df_new_inactive, clusters_inactive_1, final_thres_inactive)

                aT, aV, iT, iV = current_scenario.allocate_cycle(list_cluster_A, len_cluster_A,
                                                                                list_cluster_I, len_cluster_I,
                                                                                n_active, n_inactive,
                                                                                size_number,
                                                                                active_feat_matrix, inactive_feat_matrix,
                                                                                cycle = cycle)
                results = [aT, aV, iT, iV]
                                    
                if any(v is None for v in results) or any(len(v) == 0 for v in results):
                    print("Please choose other thresholds")

                    aT, aV, iT, iV = None, None, None, None
                else:
                    ratio = (len(aT) + len(iT)) / (len(aV) + len(iV))

                    if ratio < size_number - 1 or ratio > size_number:
                        print("Please choose other split ratio (k)")

                        aT, aV, iT, iV = None, None, None, None
                    else:
                        print("Done!")

        else:

            def objective(trial, upper_active, upper_inactive):

                thres_active = trial.suggest_float("thres_active", 0, upper_active)
                thres_inactive = trial.suggest_float("thres_inactive", 0, upper_inactive)

                clusters_active_1, len_clusters_A_1 = clustering.get_clusters_tanimoto(df_new_active, thres_active)
                clusters_inactive_1, len_clusters_I_1 = clustering.get_clusters_tanimoto(df_new_inactive, thres_inactive)

                if np.max(len_clusters_A_1)/n_active > size_number/(size_number+1) or np.max(len_clusters_I_1)/n_inactive > size_number/(size_number+1):
                    del clusters_active_1, len_clusters_A_1
                    del clusters_inactive_1, len_clusters_I_1
                    return (max(upper_active, 100*(upper_active-thres_active)), max(upper_inactive, 100*(upper_inactive - thres_inactive)))
                
                else:
                    list_cluster_A, len_cluster_A = clustering.merge_overlapping_clusters(df_new_active, clusters_active_1, thres_active)
                    list_cluster_I, len_cluster_I = clustering.merge_overlapping_clusters(df_new_inactive, clusters_inactive_1, thres_inactive)

                    del clusters_active_1, len_clusters_A_1
                    del clusters_inactive_1, len_clusters_I_1
            
                    active_train, active_test, inactive_train, inactive_test = current_scenario.allocate_cycle(
                        list_cluster_A, len_cluster_A,
                        list_cluster_I, len_cluster_I,
                        n_active, n_inactive,
                        size_number,
                        active_feat_matrix, inactive_feat_matrix,
                        cycle = cycle
                        )
                    
                    results = [active_train, active_test, inactive_train, inactive_test]
                    
                    if any(v is None for v in results) or any(len(v) == 0 for v in results):
                        del list_cluster_A, len_cluster_A, list_cluster_I, len_cluster_I
                        del active_train, active_test, inactive_train, inactive_test
                        del results

                        return (max(upper_active, 100*(upper_active-thres_active)), max(upper_inactive, 100*(upper_inactive - thres_inactive)))
                    else:
                        ratio = (len(active_train) + len(inactive_train)) / (len(active_test) + len(inactive_test))

                        if ratio < size_number - 1 or ratio > size_number:
                            del list_cluster_A, len_cluster_A, list_cluster_I, len_cluster_I
                            del active_train, active_test, inactive_train, inactive_test
                            del results
                            return (max(upper_active, 100*(upper_active-thres_active)), max(upper_inactive, 100*(upper_inactive - thres_inactive)))
                        else:
                
                            del list_cluster_A, len_cluster_A, list_cluster_I, len_cluster_I
                            del active_train, active_test, inactive_train, inactive_test
                            del results

                            return thres_active, thres_inactive
            
            os.makedirs(outdir, exist_ok=True)
            db_path = os.path.join(outdir, f"optuna_study_{name_study}.db")

            study_obj = optuna.create_study(
                study_name=f"{name_study}",
                storage=f"sqlite:///{db_path}", 
                load_if_exists=True,
                directions=["minimize","minimize"],
                sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=int(iteration/10), multivariate=True),
            )

            if args.continue_tune:
                completed_trials = len(study_obj.get_trials(states=[TrialState.COMPLETE]))
                remaining_trials = iteration - completed_trials
                print(f"Resuming study: {completed_trials} trials already completed. Running {remaining_trials} remaining trials...")

                study_obj.optimize(
                    lambda trial: objective(trial, upper_active, upper_inactive),
                    n_trials=remaining_trials,
                    gc_after_trial=True,
                    callbacks=[optuna.study.MaxTrialsCallback(iteration, states=(TrialState.COMPLETE,))]
                )
            else:
                study_obj.optimize(
                    lambda trial: objective(trial, upper_active, upper_inactive),
                    n_trials=iteration,
                    gc_after_trial=True
                )

            best_trials = study_obj.best_trials

            best_doe = 999
            aT, aV, iT, iV = [], [], [], []
            final_thres_active = None
            final_thres_inactive = None

            for i, trial in enumerate(best_trials):
                optimized_thres_active = trial.values[0]
                optimized_thres_inactive = trial.values[1]

                clusters_active_1, len_clusters_A_1 = clustering.get_clusters_tanimoto(df_new_active, optimized_thres_active)
                list_cluster_A, len_cluster_A = clustering.merge_overlapping_clusters(df_new_active, clusters_active_1, optimized_thres_active)

                del clusters_active_1, len_clusters_A_1

                clusters_inactive_1, len_clusters_I_1 = clustering.get_clusters_tanimoto(df_new_inactive, optimized_thres_inactive)
                list_cluster_I, len_cluster_I = clustering.merge_overlapping_clusters(df_new_inactive, clusters_inactive_1, optimized_thres_inactive)

                aT_try, aV_try, iT_try, iV_try = current_scenario.allocate_cycle(list_cluster_A, len_cluster_A,
                                                                                list_cluster_I, len_cluster_I,
                                                                                n_active, n_inactive,
                                                                                size_number,
                                                                                active_feat_matrix, inactive_feat_matrix,
                                                                                cycle = cycle)

                del list_cluster_A, len_cluster_A, list_cluster_I, len_cluster_I

                active_feat = active_feat_matrix[aV_try]
                inactive_feat = inactive_feat_matrix[iV_try]
                doe_result = doe_score(active_feat, inactive_feat)

                if doe_result < best_doe:
                    best_doe = doe_result
                    
                    aT = aT_try
                    aV = aV_try
                    iT = iT_try
                    iV = iV_try

                    final_thres_active = optimized_thres_active
                    final_thres_inactive = optimized_thres_inactive

                # print(optimized_thres_active, optimized_thres_inactive, doe_result)

                del active_feat, inactive_feat, aT_try, aV_try, iT_try, iV_try

         ################################################
        #Extract and save best scores
        if all([aT, aV, iT, iV]):
            os.makedirs(outdir, exist_ok=True)

            final_test_A = os.path.join(outdir, "Test_Active.smi")
            final_test_I = os.path.join(outdir, "Test_Inactive.smi")
            final_train_A = os.path.join(outdir, "Train_Active.smi")
            final_train_I = os.path.join(outdir, "Train_Inactive.smi")
            final_metrics = os.path.join(outdir, "Metrics.txt")

            df_new_active.loc[aT, ["SMILES", "ID"]].to_csv(final_train_A, index=False, header=False)
            df_new_active.loc[aV, ["SMILES", "ID"]].to_csv(final_test_A, index=False, header=False)
            df_new_inactive.loc[iT, ["SMILES", "ID"]].to_csv(final_train_I, index=False, header=False)
            df_new_inactive.loc[iV, ["SMILES", "ID"]].to_csv(final_test_I, index=False, header=False)

            with open(final_metrics, "w") as f:
                f.write(f"Training: Active ({len(aT)}), Inactive ({len(iT)})\n")
                f.write(f"Test: Active ({len(aV)}), Inactive ({len(iV)})\n")
                f.write(f"Maximum Threshold: Active: {final_thres_active}, Inactive: {final_thres_inactive}\n")


            print(f"Done! Outputs are saved at {outdir}")

    ###################################################
    elif algorithm=='bitbirch':

        df_new_active = read_dataframe(df_active)
        active_feat_matrix = calc_dataset_props_all(df_new_active)

        df_new_inactive = read_dataframe(df_inactive)
        inactive_feat_matrix = calc_dataset_props_all(df_new_inactive)

        n_active = df_new_active.shape[0]
        n_inactive = df_new_inactive.shape[0]

        subprocess.run(["mkdir", f"{name_study}"])

        with tempfile.NamedTemporaryFile(mode='w', suffix='.smi', delete=False) as tmp:
            tmp.write('\n'.join(df_new_active['SMILES'].astype(str).tolist()))
            tmp_path_active = tmp.name
        
        subprocess.run(["bb", "fps-from-smiles",tmp_path_active, "--out-dir", f"{name_study}/active_fps", "--no-verbose"],check=True)
        os.remove(tmp_path_active)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.smi', delete=False) as tmp:
            tmp.write('\n'.join(df_new_inactive['SMILES'].astype(str).tolist()))
            tmp_path_inactive = tmp.name

        subprocess.run(["bb", "fps-from-smiles", tmp_path_inactive, "--out-dir", f"{name_study}/inactive_fps", "--no-verbose"],check=True)
        os.remove(tmp_path_inactive)

        cmd_active = f"bb thresh {name_study}/active_fps/*.npy | grep 'optimal threshold' | rev | cut -d ' ' -f1 | rev"
        result_active = subprocess.run(cmd_active, shell=True, capture_output=True, text=True)
        threshold_active = result_active.stdout.strip()

        cmd_inactive = f"bb thresh {name_study}/inactive_fps/*.npy | grep 'optimal threshold' | rev | cut -d ' ' -f1 | rev"
        result_inactive = subprocess.run(cmd_inactive, shell=True, capture_output=True, text=True)
        threshold_inactive = result_inactive.stdout.strip()

        subprocess.run(["bb", "run", f"{name_study}/active_fps", 
                        "--threshold", threshold_active, 
                        "--out-dir", f"{name_study}/active_cluster", 
                        "--no-verbose"],check=True)
        
        subprocess.run(["bb", "run", f"{name_study}/inactive_fps", 
                        "--threshold", threshold_inactive, 
                        "--out-dir", f"{name_study}/inactive_cluster", 
                        "--no-verbose"],check=True)
        
        
        list_cluster_A = pickle.load(open(f"{name_study}/active_cluster/clusters.pkl", "rb"))
        len_cluster_A = [len(ca) for ca in list_cluster_A]
    
        list_cluster_I = pickle.load(open(f"{name_study}/inactive_cluster/clusters.pkl", "rb"))
        len_cluster_I = [len(ci) for ci in list_cluster_I]

        aT, aV, iT, iV = current_scenario.allocate_cycle(list_cluster_A, len_cluster_A,
                                                        list_cluster_I, len_cluster_I,
                                                        n_active, n_inactive,
                                                        size_number,
                                                        active_feat_matrix, inactive_feat_matrix,
                                                        cycle = cycle)

        results = [aT, aV, iT, iV]
                                        
        if any(v is None for v in results) or any(len(v) == 0 for v in results):
            print("Please choose other split ratio (k)")

            aT, aV, iT, iV = None, None, None, None
        else:
            ratio = (len(aT) + len(iT)) / (len(aV) + len(iV))

            if ratio < size_number - 1 or ratio > size_number:
                print("Please choose other split ratio (k)")

                aT, aV, iT, iV = None, None, None, None
            else:
                print("Done!")

        if all([aT, aV, iT, iV]):
            os.makedirs(outdir, exist_ok=True)

            final_test_A = os.path.join(outdir, "Test_Active.smi")
            final_test_I = os.path.join(outdir, "Test_Inactive.smi")
            final_train_A = os.path.join(outdir, "Train_Active.smi")
            final_train_I = os.path.join(outdir, "Train_Inactive.smi")
            final_metrics = os.path.join(outdir, "Metrics.txt")

            df_new_active.loc[aT, ["SMILES", "ID"]].to_csv(final_train_A, index=False, header=False)
            df_new_active.loc[aV, ["SMILES", "ID"]].to_csv(final_test_A, index=False, header=False)
            df_new_inactive.loc[iT, ["SMILES", "ID"]].to_csv(final_train_I, index=False, header=False)
            df_new_inactive.loc[iV, ["SMILES", "ID"]].to_csv(final_test_I, index=False, header=False)

            with open(final_metrics, "w") as f:
                f.write(f"Training: Active ({len(aT)}), Inactive ({len(iT)})\n")
                f.write(f"Test: Active ({len(aV)}), Inactive ({len(iV)})\n")
                f.write(f"Optimal threshold: Active: {threshold_active}, Inactive: {threshold_inactive}\n")

            subprocess.run(["mv", f"{name_study}", outdir], check=True)

            print(f"Done! Outputs are saved at {outdir}")
  
    ###################################################
    elif algorithm=='other':

        df_new_active = read_dataframe(df_active)
        active_feat_matrix = calc_dataset_props_all(df_new_active)

        df_new_inactive = read_dataframe(df_inactive)
        inactive_feat_matrix = calc_dataset_props_all(df_new_inactive)

        n_active = df_new_active.shape[0]
        n_inactive = df_new_inactive.shape[0]
        
        
        list_cluster_A = [list(idx_active) for idx_active in df_active.groupby('Cluster').indices.values()]
        len_cluster_A = [len(ca) for ca in list_cluster_A]
    
        list_cluster_I = [list(idx_inactive) for idx_inactive in df_inactive.groupby('Cluster').indices.values()]
        len_cluster_I = [len(ci) for ci in list_cluster_I]

        aT, aV, iT, iV = current_scenario.allocate_cycle(list_cluster_A, len_cluster_A,
                                                        list_cluster_I, len_cluster_I,
                                                        n_active, n_inactive,
                                                        size_number,
                                                        active_feat_matrix, inactive_feat_matrix,
                                                        cycle = cycle)

        results = [aT, aV, iT, iV]
                                        
        if any(v is None for v in results) or any(len(v) == 0 for v in results):
            print("Please choose other split ratio (k) or other clustering results")

            aT, aV, iT, iV = None, None, None, None
        else:
            ratio = (len(aT) + len(iT)) / (len(aV) + len(iV))

            if ratio < size_number - 1 or ratio > size_number:
                print("Please choose other split ratio (k) or other clustering results")

                aT, aV, iT, iV = None, None, None, None
            else:
                print("Done!")

        if all([aT, aV, iT, iV]):
            os.makedirs(outdir, exist_ok=True)

            final_test_A = os.path.join(outdir, "Test_Active.smi")
            final_test_I = os.path.join(outdir, "Test_Inactive.smi")
            final_train_A = os.path.join(outdir, "Train_Active.smi")
            final_train_I = os.path.join(outdir, "Train_Inactive.smi")
            final_metrics = os.path.join(outdir, "Metrics.txt")

            df_new_active.loc[aT, ["SMILES", "ID", "Cluster"]].to_csv(final_train_A, index=False, header=False)
            df_new_active.loc[aV, ["SMILES", "ID", "Cluster"]].to_csv(final_test_A, index=False, header=False)
            df_new_inactive.loc[iT, ["SMILES", "ID", "Cluster"]].to_csv(final_train_I, index=False, header=False)
            df_new_inactive.loc[iV, ["SMILES", "ID", "Cluster"]].to_csv(final_test_I, index=False, header=False)

            with open(final_metrics, "w") as f:
                f.write(f"Training: Active ({len(aT)}), Inactive ({len(iT)})\n")
                f.write(f"Test: Active ({len(aV)}), Inactive ({len(iV)})\n")
                f.write(f"Optimal threshold: Active: {threshold_active}, Inactive: {threshold_inactive}\n")

            print(f"Done! Outputs are saved at {outdir}")
    
    ##############################################################
    
    else:
        print("Please provide available clustering algorithm: molscissor or bitbirch or other")

    

if __name__ == "__main__":
    main()
