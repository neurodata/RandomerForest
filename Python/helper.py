
from numpy import genfromtxt

DATA_NAMES = [ "abalone", "acute_inflammation", "acute_nephritis", "adult", "annealing", "arrhythmia", "audiology_std", "balance_scale", "balloons", "bank", "blood", "breast_cancer", "breast_cancer_wisc_diag", "breast_cancer_wisc_prog", "breast_cancer_wisc", "breast_tissue", "car", "cardiotocography_10clases", "cardiotocography_3clases", "chess_krvk", "chess_krvkp", "congressional_voting", "conn_bench_sonar_mines_rocks", "conn_bench_vowel_deterding", "connect_4", "contrac", "credit_approval", "cylinder_bands", "dermatology", "echocardiogram", "ecoli", "energy_y1", "energy_y2", "fertility", "flags", "glass", "haberman_survival", "hayes_roth", "heart_cleveland", "heart_hungarian", "heart_switzerland", "heart_va", "hepatitis", "hill_valley", "horse_colic", "ilpd_indian_liver", "image_segmentation", "ionosphere", "iris", "led_display", "lenses", "letter", "libras", "low_res_spect", "lung_cancer", "lymphography", "magic", "mammographic", "miniboone", "molec_biol_promoter", "molec_biol_splice", "monks_1", "monks_2", "monks_3", "mushroom", "musk_1", "musk_2", "nursery", "oocytes_merluccius_nucleus_4d", "oocytes_merluccius_states_2f", "oocytes_trisopterus_nucleus_2f", "oocytes_trisopterus_states_5b", "optical", "ozone", "page_blocks", "parkinsons", "pendigits", "pima", "pittsburg_bridges_MATERIAL", "pittsburg_bridges_REL_L", "pittsburg_bridges_SPAN", "pittsburg_bridges_T_OR_D", "pittsburg_bridges_TYPE", "planning", "plant_margin", "plant_shape", "plant_texture", "post_operative", "primary_tumor", "ringnorm", "seeds", "semeion", "soybean", "spambase", "spect", "spectf", "statlog_australian_credit", "statlog_german_credit", "statlog_heart", "statlog_image", "statlog_landsat", "statlog_shuttle", "statlog_vehicle", "steel_plates", "synthetic_control", "teaching", "thyroid", "tic_tac_toe", "titanic", "trains", "twonorm", "vertebral_column_2clases", "vertebral_column_3clases", "wall_following", "waveform_noise", "waveform", "wine_quality_red", "wine_quality_white", "wine", "yeast", "zoo"]


def read_data(name):
    f_train = "../Data/Benchmarks/" + name + "_train.dat"
    f_test = "../Data/Benchmarks/" + name + "_test.dat"


    train = genfromtxt(f_train, delimiter = "\t")
    Y = train[:, -1]
    X = train[:, :(train.shape[1] - 2)]


    test = genfromtxt(f_test, delimiter = "\t")

    data = {'trainX': X, 'trainY': Y, 'test': test}

    return(data)


