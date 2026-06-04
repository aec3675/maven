import os
import torch
import pickle
import numpy as np
import pandas as pd
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

# to read in
weight2read = 1.0
model_label2read = 'multipeak-finetune-cFrF-picked-kfolds-aug'

directories = [
    # "models/multipeak-finetune-cFrF",
    # "models/multipeak-finetune-noweights-cFrF",
    # "models/multipeak-finetune-weights0210-cTrF",
    # "models/multipeak-finetune-weights0208-cTrF-newinit/", 
    # "models/multipeak-finetune-cFrF-picked-kfolds",
    # "models/multipeak-finetune-cFrF-picked-kfolds-aug",
    f"models/{model_label2read}"
    # "models/clip_finetune",                             #<-maven results
]  
names = [
    # "multipeak-finetune-cFrF",
    # "multipeak-finetune-noweights-cFrF",
    # "multipeak-finetune-weights0210-cTrF",
    # "multipeak-finetune-weights0208-cTrF-newinit/",
    # "multipeak-finetune-cFrF-picked-kfolds",
    # "multipeak-finetune-cFrF-picked-kfolds-aug",
    f"{model_label2read}"
    # "clip_finetune",                                    #<-maven results
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
    # print(path)
    models.append(load_model(path))

print("finished loading models")


# Data preprocessing
#Our dirs
data_dirs = [
        "/Users/pnr5sh/Documents/phd/maven/data/test/all",
        "data/test/all",
        "test/all",
        "all",
    ]
data_dir = get_valid_dir(data_dirs)

# Our dirs
data_dirs = [
        "/Users/pnr5sh/Documents/phd/maven/data/test/all_spectra",
        "data/test/all_spectra",
        "test/all_spectra",
        "all_spectra",
    ]
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

# filename_idx = 0
kfold = 0
best_loss_epochs = [124, 99, 110, 89, 118]
objs, true_labels, pred_labels, datatypes ,kfolds = [],[],[],[],[]

for output, label, id_ in zip(models, labels, ids):
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

    print('n_classes', n_classes) # must be 2 for our data, 5 for maven data

    for fidx in range(len(val_filenames)):

        val_filename = val_filenames[fidx:fidx+1]

        dataset_val, nband, filenames_read, _ = load_data(
            data_dir,
            spectra_dir,
            max_data_len_spec=cfg_extra_args["max_spectral_data_len"],
            combinations=cfg_extra_args["combinations"],
            spectral_rescalefactor=cfg_extra_args["spectral_rescalefactor"],
            filenames=val_filename,
            n_classes=n_classes,
            kfolds=None,
        )

        # Check that the filenames read are a subset of the training filenames from the already trained models
        assert is_subset(filenames_read, val_filename)

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
        
        #y_pred empty if regression is not True
        y_true, y_true_label, y_pred, lc_data = process_data_loader(
            val_loader_no_aug,
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
        
        embs_list, combs = get_embs(
            model, val_loader_no_aug, cfg_extra_args["combinations"], ret_combs=True
        )
        # looping over different amount of classes to predict
        for n_classes in ["two"]: #"five", "three",  "two"
            if n_classes == "two":
                subclasses = torch.tensor(
                    [0,1] #NOTE: change back when multipeak [0, 1]
                )
                embs_list, y_true_label, lc_data = filter_classes(
                    embs_list, y_true_label, lc_data, subclasses
                )

            # loop over different combinations of modalities
            for i in range(len(embs_list)):
                # print(f"Train set linear regression R2 value for {combs[i]}: {get_linearR2(embs_list_train[i], y_true_train)}")
                print(f"---- {combs[i]} input ---- ")
                for task in ["regression", "classification"]:
                    if task == "classification":                    
                        y_pred_mlp = get_mlp_predictions(
                            None, # copy as placeholder; won't actually be used
                            None, # copy as placeholder; won't actually be used
                            embs_list[i],
                            y_true_label,
                            task=task,
                            save_model=False,
                            load_model=True,
                            load_model_path=f"models/mlp-states/5fold-longrun/{model_label2read}+MLP+two_{combs[i]}_kfold{kfold}_w{weight2read}_epoch{best_loss_epochs[kfold]}",
                            seed=cfg["seed"],
                        )
                        
                        objs.append(val_filename)
                        true_labels.append(y_true_label)
                        pred_labels.append(y_pred_mlp)
                        datatypes.append(combs[i])
                        kfolds.append(kfold)

            # for concatenated pairs of modalities
            for i in range(len(embs_list)):
                for j in range(i + 1, len(embs_list)):
                    emb_concat = torch.cat([embs_list[i], embs_list[j]], dim=1)
                    print(f"---- {combs[i]} and {combs[j]} input ---- ")
                    for task in ["regression", "classification"]:
                        # Regression only for five classes
                        if task == "classification":
                            y_pred_mlp = get_mlp_predictions(
                                None, # copy as placeholder; wont actually be used
                                None, # copy as placeholder; wont actually be used
                                emb_concat,
                                y_true_label,
                                task=task,
                                save_model=False,
                                load_model=True,
                                load_model_path=f"models/mlp-states/5fold-longrun/{model_label2read}+MLP+two_{combs[i]}and{combs[j]}_kfold{kfold}_w{weight2read}_epoch{best_loss_epochs[kfold]}",
                                seed=cfg["seed"],
                            )
                            
                            objs.append(val_filename)
                            true_labels.append(y_true_label)
                            pred_labels.append(y_pred_mlp)
                            datatypes.append(f'{combs[i]}and{combs[j]}')
                            kfolds.append(kfold)
    
    kfold += 1
    print("===============================")

objs = np.array(objs).flatten()
true_labels = np.array(true_labels).flatten()
pred_labels = np.array(pred_labels).flatten()
datatypes = np.array(datatypes).flatten()
kfolds = np.array(kfolds).flatten()

print(np.shape(objs), type(objs))
print(np.shape(true_labels), type(true_labels))
print(np.shape(pred_labels), type(pred_labels))
print(np.shape(datatypes), type(datatypes))
print(np.shape(kfolds), type(kfolds))

datadict = {
    'SN': objs,
    'true_label': true_labels,
    'prediction': pred_labels,
    'modality': datatypes,
    'kfold_of_pred': kfolds, 
}

single_obj_predictions_df = pd.DataFrame.from_dict(datadict)
single_obj_predictions_df.to_csv(f'./{label}+MLP+{n_classes}_w{weight2read}_200epochs_single_objs.csv', index=False)
