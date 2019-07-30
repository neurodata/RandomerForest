
import os
import gc
import argparse
import time

from helper import selectKmK, read_kfold_data, read_data, setup, setupCLF

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from rerf.rerfClassifier import rerfClassifier

NJOBS = int(os.environ["NCORES"])
print(f"njobs:{NJOBS}")

##
parser = argparse.ArgumentParser(description = "Run UCI datasets through classification and track statistics.")
parser.add_argument('dataset_name', metavar='N', type=str, help='Dataset Name')
parser.add_argument('classifier_name', metavar='C', type=str, help='Classifier')

args = parser.parse_args()

dataset_name = "wine"
classifier_name = "rerf"

dataset_name = args.dataset_name
classifier_name = args.classifier_name

#DATA_NAMES = [ "abalone", "acute_inflammation", "acute_nephritis", "adult", "annealing", "arrhythmia", "audiology_std", "balance_scale", "balloons", "bank", "blood", "breast_cancer", "breast_cancer_wisc_diag", "breast_cancer_wisc_prog", "breast_cancer_wisc", "breast_tissue", "car", "cardiotocography_10clases", "cardiotocography_3clases", "chess_krvk", "chess_krvkp", "congressional_voting", "conn_bench_sonar_mines_rocks", "conn_bench_vowel_deterding", "connect_4", "contrac", "credit_approval", "cylinder_bands", "dermatology", "echocardiogram", "ecoli", "energy_y1", "energy_y2", "fertility", "flags", "glass", "haberman_survival", "hayes_roth", "heart_cleveland", "heart_hungarian", "heart_switzerland", "heart_va", "hepatitis", "hill_valley", "horse_colic", "ilpd_indian_liver", "image_segmentation", "ionosphere", "iris", "led_display", "lenses", "letter", "libras", "low_res_spect", "lung_cancer", "lymphography", "magic", "mammographic", "miniboone", "molec_biol_promoter", "molec_biol_splice", "monks_1", "monks_2", "monks_3", "mushroom", "musk_1", "musk_2", "nursery", "oocytes_merluccius_nucleus_4d", "oocytes_merluccius_states_2f", "oocytes_trisopterus_nucleus_2f", "oocytes_trisopterus_states_5b", "optical", "ozone", "page_blocks", "parkinsons", "pendigits", "pima", "pittsburg_bridges_MATERIAL", "pittsburg_bridges_REL_L", "pittsburg_bridges_SPAN", "pittsburg_bridges_T_OR_D", "pittsburg_bridges_TYPE", "planning", "plant_margin", "plant_shape", "plant_texture", "post_operative", "primary_tumor", "ringnorm", "seeds", "semeion", "soybean", "spambase", "spect", "spectf", "statlog_australian_credit", "statlog_german_credit", "statlog_heart", "statlog_image", "statlog_landsat", "statlog_shuttle", "statlog_vehicle", "steel_plates", "synthetic_control", "teaching", "thyroid", "tic_tac_toe", "titanic", "trains", "twonorm", "vertebral_column_2clases", "vertebral_column_3clases", "wall_following", "waveform_noise", "waveform", "wine_quality_red", "wine_quality_white", "wine", "yeast", "zoo"]



#trainX, trainY, testX, testY = read_data(dataset_name)

data = read_kfold_data(dataset_name)

NTREES, MTRY, MTRY_MULT = setup(classifier_name, data["X"].shape[1])


for k in range(len(data['folds'])):
    trainX, trainY, testX, testY = selectKmK(data, k)
    for key in MTRY:
        for mult in MTRY_MULT:
            outFile = f"RESULTS/{classifier_name}_{dataset_name}_fold{k+1}_{NTREES}trees_mtry{key}_mult{mult}.tsv"
            try:
                clf = setupCLF(classifier_name, NTREES, MTRY[key], mult, NJOBS)
                trainTimeStart = time.time()
                clf.fit(trainX, trainY)
                trainTimeStop = time.time()
    
                trainTime = trainTimeStop - trainTimeStart
    
                testTimeStart = time.time()
                Yhat = clf.predict(testX)
                testTimeStop = time.time()
    
                testTime = testTimeStop - testTimeStart
    
                testError = np.mean(Yhat != testY)
    
                df = pd.DataFrame({"Classifier":[classifier_name],
                    "dataset":[dataset_name],"fold":[k+1], "NTREES":[NTREES],
                    "testError": [testError], "mtry": [f"{MTRY[key]}+{key}i"],
                    "mtrymult": [mult], "trainTime":[trainTime],
                    "testTime":[testTime]})
    
                out = df.to_csv(index = False, sep = "\t")
                print(f"\n{out}\n")
                with open(outFile, "w") as f:
                    f.write(out)
    
                del clf
            except Exception as e:
                print(f"{e}\n")
                print("Failed: " + outFile)
                None
    
