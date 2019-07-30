
import os
import csv
import math
import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from rerf.rerfClassifier import rerfClassifier

NJOBS = int(os.environ["NCORES"])

#DATA_NAMES = [ "abalone", "acute_inflammation", "acute_nephritis", "adult", "annealing", "arrhythmia", "audiology_std", "balance_scale", "balloons", "bank", "blood", "breast_cancer", "breast_cancer_wisc_diag", "breast_cancer_wisc_prog", "breast_cancer_wisc", "breast_tissue", "car", "cardiotocography_10clases", "cardiotocography_3clases", "chess_krvk", "chess_krvkp", "congressional_voting", "conn_bench_sonar_mines_rocks", "conn_bench_vowel_deterding", "connect_4", "contrac", "credit_approval", "cylinder_bands", "dermatology", "echocardiogram", "ecoli", "energy_y1", "energy_y2", "fertility", "flags", "glass", "haberman_survival", "hayes_roth", "heart_cleveland", "heart_hungarian", "heart_switzerland", "heart_va", "hepatitis", "hill_valley", "horse_colic", "ilpd_indian_liver", "image_segmentation", "ionosphere", "iris", "led_display", "lenses", "letter", "libras", "low_res_spect", "lung_cancer", "lymphography", "magic", "mammographic", "miniboone", "molec_biol_promoter", "molec_biol_splice", "monks_1", "monks_2", "monks_3", "mushroom", "musk_1", "musk_2", "nursery", "oocytes_merluccius_nucleus_4d", "oocytes_merluccius_states_2f", "oocytes_trisopterus_nucleus_2f", "oocytes_trisopterus_states_5b", "optical", "ozone", "page_blocks", "parkinsons", "pendigits", "pima", "pittsburg_bridges_MATERIAL", "pittsburg_bridges_REL_L", "pittsburg_bridges_SPAN", "pittsburg_bridges_T_OR_D", "pittsburg_bridges_TYPE", "planning", "plant_margin", "plant_shape", "plant_texture", "post_operative", "primary_tumor", "ringnorm", "seeds", "semeion", "soybean", "spambase", "spect", "spectf", "statlog_australian_credit", "statlog_german_credit", "statlog_heart", "statlog_image", "statlog_landsat", "statlog_shuttle", "statlog_vehicle", "steel_plates", "synthetic_control", "teaching", "thyroid", "tic_tac_toe", "titanic", "trains", "twonorm", "vertebral_column_2clases", "vertebral_column_3clases", "wall_following", "waveform_noise", "waveform", "wine_quality_red", "wine_quality_white", "wine", "yeast", "zoo"]


def read_data(name):
    f_train = "../Data/Benchmarks/" + name + "_train.dat"
    f_test = "../Data/Benchmarks/" + name + "_test.dat"


    train = genfromtxt(f_train, delimiter = "\t")
    trainY = train[:, -1]
    trainX = train[:, :(train.shape[1] - 1)]


    test = genfromtxt(f_test, delimiter = "\t")
    testY = test[:, -1]
    testX = test[:, :(test.shape[1] -1)]

    #data = {'trainX': trainX, 'trainY': trainY, 'testX': testX, 'testY': testY}

    return(trainX, trainY, testX, testY)



def read_kfold_data(name):
    kfolds = "processed/cv_partitions/" + name + "_partitions.txt"

    with open(kfolds, 'r') as f:
        reader = csv.reader(f) 
        folds = list(map(tuple, reader))


    f_train = "processed/data/" + name + ".csv"

    data = genfromtxt(f_train, delimiter = ",")

    Y = data[:, -1]
    X = data[:, :(data.shape[1] - 1)]

    out = {'X':X, 'Y':Y, 'folds': folds}

    return(out)


def classRerpresentation(trainY):

    uni = np.bincount(trainY)

    return(None)

def selectKmK(data, k):

    ## Get list of indices to hold out
    ## The indices were 1-indexed.
    K = np.asarray(data['folds'][k], 'int') - 1
    
    outK = np.ones(data["X"].shape[0], np.bool)
    outK[K] = 0

    trainX = data["X"][outK,:]
    trainY = np.asarray(data["Y"][outK], "int")

    Kin = []
    for j in [ki for ki in range(len(data['folds'])) if ki != k]:
        Kin += data['folds'][j]

    inK = np.ones(data["X"].shape[0], np.bool)
    inK[np.asarray(Kin, 'int') - 1] = 0

    testX = data["X"][inK,:]
    testY = np.asarray(data["Y"][inK], 'int')

    #npuy  np.unique(trainY)
    #Ymap = {uy[0]:uy[1] for uy in zip(range(npuy.shape[0]), npuy)}

    return(trainX, trainY, testX, testY)



def setup(classifier_name, num_feat):

    NTREES = 500

    if classifier_name == "rerf":
        MTRY = {str(i):math.ceil(num_feat**i) for i in [1/4, 1/2, 3/4, 1, 2]}
        MTRY_MULT = [i + 1 for i in range(5)]
    elif (classifier_name == "RF" or
          classifier_name == "SKRF" or
          classifier_name == "SKX"):

        MTRY_MULT = [1]
        MTRY = {str(i):math.ceil(num_feat**i) for i in [1/4, 1/2, 3/4, 1]}

    return(NTREES, MTRY, MTRY_MULT)



def setupCLF(classifier_name, NTREES, MTRY, MTRY_MULT, NJOBS):

    RS = 317
    print(f"RANDOM_STATE={RS}\n")
    if classifier_name == "rerf":
        clf = rerfClassifier(n_estimators = NTREES, projection_matrix = "RerF",\
                        max_features = MTRY, feature_combinations = MTRY_MULT,\
                                n_jobs = NJOBS, random_state = RS)
    elif classifier_name == "RF":
        clf = rerfClassifier(n_estimators = NTREES, projection_matrix = "Base",\
                        max_features = MTRY, n_jobs = NJOBS, random_state = RS)
    elif classifier_name == "SKRF":
        clf = RandomForestClassifier(n_estimators = NTREES, max_features = MTRY,\
                        n_jobs = NJOBS, random_state = RS)
    elif classifier_name == "SKX":
        clf = ExtraTreesClassifier(n_estimators = NTREES, max_features = MTRY,\
                n_jobs = NJOBS, random_state = RS)

    return(clf)

def stratifiedPartitions(dataset_name, k = 5, random_state=0):

    partition1 = f"processed/cv_partitions/{dataset_name}_partitions.txt"

    #with open(partition1, "r") as fi:
        
    outFile = f"processed/stratified_cv_partitions/{dataset_name}_stratefiedPartitions.txt"
    f_train = f"processed/data/{dataset_name}.csv"

    Y = genfromtxt(f_train, delimiter = ",")[:, -1]

    sss = StratifiedShuffleSplit(n_splits=k, random_state=random_state)
    sss.get_n_splits(Y, Y)

    print(f"{dataset_name}\n")
    with open(outFile, "a") as f:
        for train_index, _ in sss.split(Y, Y):
            writer = csv.writer(f, delimiter = ",")
            writer.writerow(list(train_index))

    return(None)




    


    
