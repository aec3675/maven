import os
import torch
import pickle
import numpy as np
from src.dataloader import (
    load_data,
    NoisyDataLoader,
)
from src.models_multimodal import (
    load_model,
)
from src.utils import (
    get_valid_dir,
    set_seed,
    get_linear_predictions,
    get_knn_predictions,
    get_embs,
    is_subset,
    process_data_loader,
    print_metrics_in_latex,
    calculate_metrics,
    get_checkpoint_paths,
    mergekfold_results,
    save_normalized_conf_matrices,
    plot_pred_vs_true,
    get_class_dependent_predictions,
    generate_radar_plots,
    filter_classes,
    get_mlp_predictions,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load models
set_seed(0)

# KNNparameters = [1,2,3,5,7,8,9]
KNNparameters = [1,3,5,7,9]

directories = [
    # "models/multipeak-finetune-cFrF",
    # "models/multipeak-finetune-noweights-cFrF",
    # "models/multipeak-finetune-weights0210-cTrF",
    # "models/multipeak-finetune-weights0208-cTrF-newinit/", 
    "models/clip_finetune",                             #<-maven results
]  
names = [
    # "multipeak-finetune-cFrF",
    # "multipeak-finetune-noweights-cFrF",
    # "multipeak-finetune-weights0210-cTrF",
    # "multipeak-finetune-weights0208-cTrF-newinit/",
    "clip_finetune",                                    #<-maven results
]
models = []

paths = []
ids = []
labels = []
# Finding all checkpoints
for id, (directory, label) in enumerate(zip(directories, names)):
    paths_to_checkpoint, name, id = get_checkpoint_paths(directory, label, id)
    paths.extend(paths_to_checkpoint)
    ids.extend(id)
    labels.extend(name)

print(paths)

for i, path in enumerate(paths):
    print(f"loading {labels[i]}")
    print(path)
    models.append(load_model(path))

print("finished loading models")


# Data preprocessing

#Maven dirs
data_dirs = [
    # "/home/thelfer1/scr4_tedwar42/thelfer1/ZTFBTS/",
    "ZTFBTS/",
    "/ocean/projects/phy230064p/shared/ZTFBTS/",
    "data/ZTFBTS/",
]
#Our dirs
# data_dirs = [
#         "/Users/pnr5sh/Documents/phd/maven/data/test/all",
#         "data/test/all",
#         "test/all",
#         "all",
#     ]
data_dir = get_valid_dir(data_dirs)

#Maven dirs
data_dirs = [
    "ZTFBTS_spectra/",
    "data/ZTFBTS_spectra/",
    # "/n/home02/gemzhang/Storage/multimodal/ZTFBTS_spectra/",
    # "/n/home02/gemzhang/Storage/multimodal/ZTFBTS_spectra/",
]
#Our dira
# data_dirs = [
#         "/Users/pnr5sh/Documents/phd/maven/data/test/all_spectra",
#         "data/test/all_spectra",
#         "test/all_spectra",
#         "all_spectra",
#     ]
spectra_dir = get_valid_dir(data_dirs)


# Default to 1 if the environment variable is not set
cpus_per_task = int(os.getenv("SLURM_CPUS_PER_TASK", 1))

# Assuming you want to leave one CPU for overhead
num_workers = 0 #max(1, cpus_per_task - 1)
print(f"Using {num_workers} workers for data loading", flush=True)

# Keeping track of all metrics
regression_metrics_list = []
classification_metrics_list = []
collect_classification_results = []
collect_regression_results = []


for output, label, id in zip(models, labels, ids):
    (
        model,
        combinations,
        regression,
        classification,
        n_classes,
        cfg,
        cfg_extra_args,
        train_filenames,
        val_filenames,
    ) = output

    set_seed(cfg["seed"])

    # Spectral data is cut to this length
    dataset_train, nband, filenames_read, _ = load_data(
        data_dir,
        spectra_dir,
        max_data_len_spec=cfg_extra_args["max_spectral_data_len"],
        combinations=cfg_extra_args["combinations"],
        spectral_rescalefactor=cfg_extra_args["spectral_rescalefactor"],
        filenames=train_filenames,
        n_classes=n_classes,
    )

    # Check that the filenames read are a subset of the training filenames from the already trained models
    assert is_subset(filenames_read, train_filenames)

    dataset_val, nband, filenames_read, _ = load_data(
        data_dir,
        spectra_dir,
        max_data_len_spec=cfg_extra_args["max_spectral_data_len"],
        combinations=cfg_extra_args["combinations"],
        spectral_rescalefactor=cfg_extra_args["spectral_rescalefactor"],
        filenames=val_filenames,
        n_classes=n_classes,
    )

    # Check that the filenames read are a subset of the training filenames from the already trained models
    assert is_subset(filenames_read, val_filenames)

    # val_fraction = cfg_extra_args.get("val_fraction", cfg_extra_args["val_fraction"])
    ## Iterate over data
    # number_of_samples = len(dataset)
    # n_samples_val = int(val_fraction * number_of_samples)

    train_loader_no_aug = NoisyDataLoader(
        dataset_train,
        batch_size=cfg["batchsize"],
        noise_level_img=0,
        noise_level_mag=0,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        combinations=cfg_extra_args["combinations"],
    )

    print('TRAIN', train_loader_no_aug.dataset)

    val_loader_no_aug = NoisyDataLoader(
        dataset_val,
        batch_size=cfg["batchsize"],
        noise_level_img=0,
        noise_level_mag=0,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        combinations=cfg_extra_args["combinations"],
    )

    model = model.to(device)
    model.eval()
    #y_pred empty is regression is not True
    y_true, y_true_label, y_pred, lc_data = process_data_loader(
        val_loader_no_aug,
        regression,
        classification,
        device,
        model,
        combinations=cfg_extra_args["combinations"],
    )
    y_true_train, y_true_train_label, _, _ = process_data_loader(
        train_loader_no_aug,
        regression,
        classification,
        device,
        model,
        combinations=cfg_extra_args["combinations"],
    )

    print("===============================")
    print(f"Model: {label}")
    print(f"Using data modalities: {cfg_extra_args['combinations']}")

    def format_combinations(combinations):
        if len(combinations) > 1:
            return ", ".join(combinations[:-1]) + " and " + combinations[-1]
        elif combinations:
            return combinations[0]
        return ""

    if regression:
        metrics, results = calculate_metrics(
            y_true,
            y_true_label,
            y_pred,
            lc_data,
            label,
            format_combinations(cfg_extra_args["combinations"]),
            id=id,
            task="regression",
        )
        collect_regression_results.append(results)
        regression_metrics_list.append(metrics)

    elif classification:
        metrics, results = calculate_metrics(
            y_true,
            y_true_label,
            y_pred,
            lc_data,
            label,
            format_combinations(cfg_extra_args["combinations"]),
            id=id,
            task="classification",
        )
        classification_metrics_list.append(metrics)
        collect_classification_results.append(results)
    else:        
        embs_list, combs = get_embs(
            model, val_loader_no_aug, cfg_extra_args["combinations"], ret_combs=True
        )
        embs_list_train = get_embs(model, train_loader_no_aug, combinations)
        # looping over different amount of classes to predict
        for n_classes in ["three", "two"]: #"five", "three", 
            if n_classes == "three":
                subclasses = torch.tensor(
                    [1, 3, 4]
                )  # Selecting subclasses 1,3 and 4 correspnding to 'SN II', 'SN Ia', 'SN Ibc'
                embs_list, y_true_label, lc_data = filter_classes(
                    embs_list, y_true_label, lc_data, subclasses
                )
                embs_list_train, y_true_train_label, _ = filter_classes(
                    embs_list_train, y_true_train_label, None, subclasses
                )
            if n_classes == "two":
                subclasses = torch.tensor(
                    [0, 1]
                )
                embs_list, y_true_label, lc_data = filter_classes(
                    embs_list, y_true_label, lc_data, subclasses
                )
                embs_list_train, y_true_train_label, _ = filter_classes(
                    embs_list_train, y_true_train_label, None, subclasses
                )
            # loop over different combinations of modalities
            for i in range(len(embs_list)):
                # print(f"Train set linear regression R2 value for {combs[i]}: {get_linearR2(embs_list_train[i], y_true_train)}")
                print(f"---- {combs[i]} input ---- ")
                for task in ["regression", "classification"]:
                    # Regression only for five classes
                    if task == "regression" and n_classes == "five":
                        y_pred_linear = get_linear_predictions(
                            embs_list_train[i],
                            y_true_train,
                            embs_list[i],
                            y_true,
                            task=task,
                        )

                        metrics, results = calculate_metrics(
                            y_true,
                            y_true_label,
                            y_pred_linear,
                            lc_data,
                            label + "+Linear",
                            combs[i],
                            id=id,
                            task=task,
                        )
                        regression_metrics_list.append(metrics)
                        collect_regression_results.append(results)

                        for kneighbours in KNNparameters:
                            y_pred_knn = get_knn_predictions(
                                embs_list_train[i],
                                y_true_train,
                                embs_list[i],
                                y_true,
                                task=task,
                                k = kneighbours,
                            )

                            metrics, results = calculate_metrics(
                                y_true,
                                y_true_label,
                                y_pred_knn,
                                lc_data,
                                label + f"+KNN{kneighbours}",
                                combs[i],
                                id=id,
                                task=task,
                            )
                            regression_metrics_list.append(metrics)
                            collect_regression_results.append(results)

                    elif task == "classification":
                        y_pred_linear = get_linear_predictions(
                            embs_list_train[i],
                            y_true_train_label,
                            embs_list[i],
                            y_true_label,
                            task=task,
                        )
                        metrics, results = calculate_metrics(
                            y_true,
                            y_true_label,
                            y_pred_linear,
                            lc_data,
                            label + f"+Linear+{n_classes}",
                            combs[i],
                            id=id,
                            task=task,
                        )
                        classification_metrics_list.append(metrics)
                        collect_classification_results.append(results)
                        
                        # #TODO: get a MLP prediction 
                        # y_pred_mlp = get_mlp_predictions(
                        #     embs_list_train[i],
                        #     y_true_train_label,
                        #     embs_list[i],
                        #     y_true_label,
                        #     task=task,
                        # )
                        # metrics, results = calculate_metrics(
                        #     y_true,
                        #     y_true_label,
                        #     y_pred_mlp,
                        #     lc_data,
                        #     label + f"+MLP+{n_classes}",
                        #     combs[i],
                        #     id=id,
                        #     task=task,
                        # )
                        # classification_metrics_list.append(metrics)
                        # collect_classification_results.append(results)
                        
                        for kneighbours in KNNparameters: #TODO: make sure neighbors different when KNN even
                            y_pred_knn = get_knn_predictions(
                                embs_list_train[i],
                                y_true_train_label,
                                embs_list[i],
                                y_true_label,
                                task=task,
                                k = kneighbours,
                            )
                            metrics, results = calculate_metrics(
                                y_true,
                                y_true_label,
                                y_pred_knn,
                                lc_data,
                                label + f"+KNN{kneighbours}+{n_classes}",
                                combs[i],
                                id=id,
                                task=task,
                            )
                            collect_classification_results.append(results)
                            classification_metrics_list.append(metrics)

            # for concatenated pairs of modalities
            for i in range(len(embs_list)):
                for j in range(i + 1, len(embs_list)):
                    emb_concat = torch.cat([embs_list[i], embs_list[j]], dim=1)
                    emb_train = torch.cat(
                        [embs_list_train[i], embs_list_train[j]], dim=1
                    )
                    print(f"---- {combs[i]} and {combs[j]} input ---- ")
                    for task in ["regression", "classification"]:
                        # Regression only for five classes
                        if task == "regression" and n_classes == "five":
                            y_pred_linear = get_linear_predictions(
                                emb_train,
                                y_true_train,
                                emb_concat,
                                y_true,
                                task=task,
                            )

                            metrics, results = calculate_metrics(
                                y_true,
                                y_true_label,
                                y_pred_linear,
                                lc_data,
                                label + "+Linear",
                                combs[i] + " and " + combs[j],
                                id=id,
                                task=task,
                            )
                            regression_metrics_list.append(metrics)
                            collect_regression_results.append(results)

                            for kneighbours in KNNparameters:
                                y_pred_knn = get_knn_predictions(
                                    emb_train,
                                    y_true_train,
                                    emb_concat,
                                    y_true,
                                    task=task,
                                    k = kneighbours,
                                )
                                
                                metrics, results = calculate_metrics(
                                    y_true,
                                    y_true_label,
                                    y_pred_knn,
                                    lc_data,
                                    label + f"+KNN{kneighbours}",
                                    combs[i] + " and " + combs[j],
                                    id=id,
                                    task=task,
                                )
                                collect_regression_results.append(results)
                                regression_metrics_list.append(metrics)
                        elif task == "classification":
                            y_pred_linear = get_linear_predictions(
                                emb_train,
                                y_true_train_label,
                                emb_concat,
                                y_true_label,
                                task=task,
                            )

                            metrics, results = calculate_metrics(
                                y_true,
                                y_true_label,
                                y_pred_linear,
                                lc_data,
                                label + f"+Linear+{n_classes}",
                                combs[i] + " and " + combs[j],
                                id=id,
                                task=task,
                            )
                            classification_metrics_list.append(metrics)
                            collect_classification_results.append(results)

                            #TODO: get MLP predictions
                            # y_pred_mlp = get_linear_predictions(
                            #     emb_train,
                            #     y_true_train_label,
                            #     emb_concat,
                            #     y_true_label,
                            #     task=task,
                            # )

                            # metrics, results = calculate_metrics(
                            #     y_true,
                            #     y_true_label,
                            #     y_pred_mlp,
                            #     lc_data,
                            #     label + f"+MLP+{n_classes}",
                            #     combs[i] + " and " + combs[j],
                            #     id=id,
                            #     task=task,
                            # )
                            # classification_metrics_list.append(metrics)
                            # collect_classification_results.append(results)

                            for kneighbours in KNNparameters:
                                y_pred_knn = get_knn_predictions(
                                    emb_train,
                                    y_true_train_label,
                                    emb_concat,
                                    y_true_label,
                                    task=task,
                                    k = kneighbours,
                                )
                                metrics, results = calculate_metrics(
                                    y_true,
                                    y_true_label,
                                    y_pred_knn,
                                    lc_data,
                                    label + f"+KNN{kneighbours}+{n_classes}",
                                    combs[i] + " and " + combs[j],
                                    id=id,
                                    task=task,
                                )
                                collect_classification_results.append(results)
                                classification_metrics_list.append(metrics)
    print("===============================")
    

exit()
class_names = {
    0: ("SLSN-I", "blue"),
    1: ("SN II", "green"),
    2: ("SN IIn", "teal"),
    3: ("SN Ia", "purple"),
    4: ("SN Ibc", "orange"),
}
# class_names = {
#     0: ("single-peak", "blue"),
#     1: ("multipeak", "green"),
# }


os.makedirs("evaluation_metrics_2", exist_ok=True)


# Convert metrics list to a DataFrame
if len(collect_classification_results) > 0:
    # print_metrics_in_latex(classification_metrics_list)
    # Save metric to file
    with open("evaluation_metrics_2/classification_metrics_list.pkl", "wb") as file:
        pickle.dump(classification_metrics_list, file)
    with open("evaluation_metrics_2/collect_classification_results.pkl", "wb") as file:
        pickle.dump(collect_classification_results, file)
    merged_classification = mergekfold_results(collect_classification_results)
    save_normalized_conf_matrices(merged_classification, class_names, "confusion_plots")

# NOTE: ignoring for now
# if len(collect_regression_results) > 0:
#     # print_metrics_in_latex(regression_metrics_list)
#     with open("evaluation_metrics/regression_metrics_list.pkl", "wb") as file:
#         pickle.dump(regression_metrics_list, file)
#     with open("evaluation_metrics/collect_regression_results.pkl", "wb") as file:
#         pickle.dump(collect_regression_results, file)
#     merged_regression = mergekfold_results(collect_regression_results)
#     folder_name = "plots"
#     plot_pred_vs_true(merged_regression, "plots", class_names)
#     # Spiderplots for regression
#     spiderplot_data = get_class_dependent_predictions(merged_regression, class_names)
#     import pandas as pd

#     df = pd.DataFrame(spiderplot_data)
#     output_dir = "radar_plots"
#     range_dict = {"L1": [0, 0.2], "L2": [0, 0.2], "R2": [-1, 1], "OLF": None}

#     generate_radar_plots(df, output_dir, range_dict)
