import json
import pandas as pd
import torch
import numpy as np
import os
import random
import argparse
import ast
import sys

from src.utils import setup_logging, get_logger

logger = get_logger(__name__)

# Utils
from utils.utils import (
    DataLoader,
    compute_pna_degrees,
    virtual_screening,
    CustomWeightedRandomSampler,
)
from utils.dataset import *  # data
from utils.trainer import Trainer
from utils.metrics import *

# Preprocessing
from utils import protein_init, ligand_init

# Model
from models.net import net

# --- Argument Parsing Helpers ---


def tuple_type(s):
    try:
        value = ast.literal_eval(s)
        if not isinstance(value, tuple):
            raise ValueError
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError(f"Invalid tuple value: {s}")
    return value


def list_type(s):
    try:
        value = ast.literal_eval(s)
        if not isinstance(value, list):
            raise ValueError
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError(f"Invalid list value: {s}")
    return value


def parse_args() -> argparse.Namespace:
    """Configures and parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Model Training and Evaluation Script")

    ### Seed and device
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--device", type=str, default="cuda:0", help='e.g., "cuda:0" or "cpu"'
    )
    parser.add_argument("--config_path", type=str, default="config.json")

    ### Data and Pre-processing
    parser.add_argument(
        "--datafolder", type=str, default="./dataset/pdb2020", help="Protein data path"
    )
    parser.add_argument(
        "--result_path",
        type=str,
        default="./result/PDB2020_BENCHMARK/",
        help="Path to save results",
    )
    parser.add_argument(
        "--save_interpret", type=bool, default=True, help="Save interpretation results"
    )

    ### Task Type
    parser.add_argument(
        "--regression_task", type=bool, help="True if regression else False"
    )
    parser.add_argument(
        "--classification_task", type=bool, help="True if classification else False"
    )
    parser.add_argument(
        "--mclassification_task",
        type=int,
        help="Number of multiclassification, 0 if no multiclass task",
    )

    ### Training Schedule
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of training epochs"
    )
    parser.add_argument(
        "--evaluate_epoch", type=int, default=1, help="Evaluate every N epochs"
    )
    parser.add_argument(
        "--total_iters",
        type=int,
        default=None,
        help="Total training iterations (overrides epochs)",
    )
    parser.add_argument(
        "--evaluate_step", type=int, default=500, help="Evaluate every N iterations"
    )

    ### Optimizer Params
    parser.add_argument("--lrate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--eps", type=float, default=1e-8, help="AdamW epsilon")
    parser.add_argument(
        "--betas", type=tuple_type, default="(0.9,0.999)", help="AdamW betas"
    )

    ### Batching and Sampling
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--sampling_col", type=str, default="", help="Column for weighted sampling"
    )

    ### Finetuning / Loading
    parser.add_argument(
        "--trained_model_path",
        type=str,
        default="",
        help="Path to a pretrained model directory",
    )
    parser.add_argument(
        "--finetune_modules",
        type=list_type,
        default=None,
        help="List of modules to finetune",
    )

    parser.add_argument("--nb_mode", type=bool, default=False)

    args = parser.parse_args()

    if args.epochs is not None and args.total_iters is not None:
        logger.info(
            "If epochs and total iters are both not None, then we only use iters."
        )
        args.epochs = None

    return args


# --- Setup Functions ---


def load_and_merge_config(args: argparse.Namespace) -> dict:
    """Loads the base JSON config and overwrites it with args."""

    # Logic fix: If finetuning, load config from model dir. Else, load from config_path.
    if args.trained_model_path:
        logger.info(f"Loading config from: {args.trained_model_path}")
        config_file = os.path.join(args.trained_model_path, "config.json")
    else:
        logger.info(f"Loading config from: {args.config_path}")
        config_file = args.config_path

    logger.info(f"Loading configuration from: {config_file}")
    with open(config_file, "r") as f:
        config = json.load(f)

    # Overwrite config with command-line arguments
    config["optimizer"]["lrate"] = args.lrate
    config["optimizer"]["eps"] = args.eps
    config["optimizer"]["betas"] = args.betas
    config["tasks"]["regression_task"] = args.regression_task
    config["tasks"]["classification_task"] = args.classification_task
    config["tasks"]["mclassification_task"] = args.mclassification_task

    return config


def setup_environment(args: argparse.Namespace):
    """Sets random seeds, creates directories, and sets device."""

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    # Set device
    device = torch.device(args.device)

    # Create result paths
    model_path = os.path.join(args.result_path, f"save_model_seed{args.seed}")
    interpret_path = os.path.join(
        args.result_path, f"interpretation_result_seed{args.seed}"
    )

    os.makedirs(args.result_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(interpret_path, exist_ok=True)

    # Save args
    with open(os.path.join(args.result_path, "model_params.txt"), "w") as f:
        f.write(str(args))

    logger.info(f"Set seed to {args.seed}")
    logger.info(f"Using device: {device}")
    logger.info(f"Results will be saved to: {args.result_path}")

    return device, model_path, interpret_path


# --- Data Loading Functions ---


def load_dataframes(datafolder):
    """Loads the train, validation, and test dataframes."""
    train_df = pd.read_csv(os.path.join(datafolder, "train.csv"))
    test_df = pd.read_csv(os.path.join(datafolder, "test.csv"))

    valid_path = os.path.join(datafolder, "valid.csv")
    valid_df = None
    if os.path.exists(valid_path):
        valid_df = pd.read_csv(valid_path)

    return train_df, test_df, valid_df


def load_or_init_graphs(datafolder, train_df, test_df, valid_df):
    """Loads pre-computed graph data or initializes it if not found."""

    # Get unique proteins and ligands
    if valid_df is not None:
        protein_seqs = list(
            set(
                train_df["Protein"].tolist()
                + test_df["Protein"].tolist()
                + valid_df["Protein"].tolist()
            )
        )
        ligand_smiles = list(
            set(
                train_df["Ligand"].tolist()
                + test_df["Ligand"].tolist()
                + valid_df["Ligand"].tolist()
            )
        )
    else:
        protein_seqs = list(
            set(train_df["Protein"].tolist() + test_df["Protein"].tolist())
        )
        ligand_smiles = list(
            set(train_df["Ligand"].tolist() + test_df["Ligand"].tolist())
        )

    # Load or initialize protein graphs
    protein_path = os.path.join(datafolder, "protein.pt")
    if os.path.exists(protein_path):
        logger.info("Loading Protein Graph data...")
        protein_dict = torch.load(protein_path)
    else:
        logger.info("Initialising Protein Sequence to Protein Graph...")
        protein_dict = protein_init(protein_seqs)
        torch.save(protein_dict, protein_path)

    # Load or initialize ligand graphs
    ligand_path = os.path.join(datafolder, "ligand.pt")
    if os.path.exists(ligand_path):
        logger.info("Loading Ligand Graph data...")
        ligand_dict = torch.load(ligand_path)
    else:
        logger.info("Initialising Ligand SMILES to Ligand Graph...")
        ligand_dict = ligand_init(ligand_smiles)
        torch.save(ligand_dict, ligand_path)

    torch.cuda.empty_cache()
    return protein_dict, ligand_dict


def prepare_dataloaders(args, protein_dict, ligand_dict, train_df, test_df, valid_df):
    """Creates and returns train, validation, and test DataLoaders."""

    # Setup training sampler
    train_shuffle = True
    train_sampler = None
    if args.sampling_col:
        train_weights = torch.from_numpy(train_df[args.sampling_col].values)
        train_sampler = CustomWeightedRandomSampler(
            train_weights, len(train_weights), replacement=True
        )
        train_shuffle = False
        logger.info(
            f"Using CustomWeightedRandomSampler on column '{args.sampling_col}'. Shuffle is False."
        )

    # Create datasets
    train_dataset = ProteinMoleculeDataset(
        train_df, ligand_dict, protein_dict, device=args.device
    )
    test_dataset = ProteinMoleculeDataset(
        test_df, ligand_dict, protein_dict, device=args.device
    )

    # Create loaders
    follow_batch_keys = ["mol_x", "clique_x", "prot_node_aa"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        follow_batch=follow_batch_keys,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        follow_batch=follow_batch_keys,
    )

    valid_loader = None
    if valid_df is not None:
        valid_dataset = ProteinMoleculeDataset(
            valid_df, ligand_dict, protein_dict, device=args.device
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            follow_batch=follow_batch_keys,
        )

    return train_loader, valid_loader, test_loader


# --- Model and Trainer Functions ---


def get_pna_degrees(args, train_loader, model_path):
    """Loads or computes PNA degrees."""

    if not args.trained_model_path:
        # Compute degrees from scratch
        degree_path = os.path.join(args.datafolder, "degree.pt")
        if not os.path.exists(degree_path):
            logger.info("Computing training data degrees for PNA...")
            mol_deg, clique_deg, prot_deg = compute_pna_degrees(train_loader)
            degree_dict = {
                "ligand_deg": mol_deg,
                "clique_deg": clique_deg,
                "protein_deg": prot_deg,
            }
            torch.save(degree_dict, degree_path)
        else:
            logger.info("Loading pre-computed PNA degrees...")
            degree_dict = torch.load(degree_path)

        # Save degrees to model result directory
        torch.save(degree_dict, os.path.join(model_path, "degree.pt"))

    else:
        # Load degrees from the trained model directory
        logger.info(
            f"Loading PNA degrees from trained model path: {args.trained_model_path}"
        )
        degree_dict = torch.load(os.path.join(args.trained_model_path, "degree.pt"))

    return (
        degree_dict["ligand_deg"],
        degree_dict["clique_deg"],
        degree_dict["protein_deg"],
    )


def initialize_model(config, mol_deg, prot_deg, device, args):
    """Initializes the network model and loads pretrained weights if specified."""

    model = net(
        mol_deg,
        prot_deg,
        # MOLECULE
        mol_in_channels=config["params"]["mol_in_channels"],
        prot_in_channels=config["params"]["prot_in_channels"],
        prot_evo_channels=config["params"]["prot_evo_channels"],
        hidden_channels=config["params"]["hidden_channels"],
        pre_layers=config["params"]["pre_layers"],
        post_layers=config["params"]["post_layers"],
        aggregators=config["params"]["aggregators"],
        scalers=config["params"]["scalers"],
        total_layer=config["params"]["total_layer"],
        K=config["params"]["K"],
        heads=config["params"]["heads"],
        dropout=config["params"]["dropout"],
        dropout_attn_score=config["params"]["dropout_attn_score"],
        # output
        regression_head=config["tasks"]["regression_task"],
        classification_head=config["tasks"]["classification_task"],
        multiclassification_head=config["tasks"]["mclassification_task"],
        device=device,
    ).to(device)

    model.reset_parameters()

    if args.trained_model_path:
        param_dict = os.path.join(args.trained_model_path, "model.pt")
        model.load_state_dict(torch.load(param_dict, map_location=device), strict=False)
        logger.info(f"Pretrained model loaded from: {param_dict}")

    nParams = sum([p.nelement() for p in model.parameters()])
    logger.info(f"Model loaded with {nParams} parameters.")

    return model


def initialize_trainer(model, config, train_loader, args, device, model_path):
    """Initializes the training engine."""

    # Determine evaluation metric
    if config["tasks"]["regression_task"]:
        evaluation_metric = "rmse"
    elif config["tasks"]["classification_task"]:
        evaluation_metric = "roc"
    elif config["tasks"]["mclassification_task"]:
        evaluation_metric = "macro_f1"
    else:
        raise Exception(
            "No valid interaction property prediction task specified in config."
        )

    engine = Trainer(
        model=model,
        lrate=config["optimizer"]["lrate"],
        min_lrate=config["optimizer"]["min_lrate"],
        wdecay=config["optimizer"]["weight_decay"],
        betas=config["optimizer"]["betas"],
        eps=config["optimizer"]["eps"],
        amsgrad=config["optimizer"]["amsgrad"],
        clip=config["optimizer"]["clip"],
        steps_per_epoch=len(train_loader),
        num_epochs=args.epochs,
        total_iters=args.total_iters,
        warmup_iters=config["optimizer"]["warmup_iters"],
        lr_decay_iters=config["optimizer"]["lr_decay_iters"],
        schedule_lr=config["optimizer"]["schedule_lr"],
        regression_weight=1,
        classification_weight=1,
        evaluate_metric=evaluation_metric,
        result_path=args.result_path,
        runid=args.seed,
        finetune_modules=args.finetune_modules,
        device=device,
    )

    # Save the config to the model directory
    with open(os.path.join(model_path, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    return engine


# --- Main Execution ---


def run_training(engine, train_loader, valid_loader, test_loader, args):
    """Runs the main training loop."""
    logger.info("-" * 50)
    logger.info("Start training model")

    if args.epochs:
        engine.train_epoch(
            train_loader,
            val_loader=valid_loader,
            test_loader=test_loader,
            evaluate_epoch=args.evaluate_epoch,
        )
    else:
        engine.train_step(
            train_loader,
            val_loader=valid_loader,
            test_loader=test_loader,
            evaluate_step=args.evaluate_step,
        )

    logger.info("Finished training model")
    logger.info("-" * 50)


def run_evaluation(
    model, model_path, test_df, test_loader, interpret_path, ligand_dict, args
):
    """Loads the best model and runs final evaluation/screening."""
    logger.info("Loading best checkpoint and predicting test data")
    logger.info("-" * 50)

    best_model_file = os.path.join(model_path, "model.pt")
    model.load_state_dict(torch.load(best_model_file, map_location=args.device))

    screen_df = virtual_screening(
        test_df,
        model,
        test_loader,
        result_path=interpret_path,
        save_interpret=args.save_interpret,
        ligand_dict=ligand_dict,
        device=args.device,
    )

    pred_file = os.path.join(args.result_path, f"test_prediction_seed{args.seed}.csv")
    screen_df.to_csv(pred_file, index=False)
    logger.info(f"Test predictions saved to: {pred_file}")


def main():
    """Main execution flow."""
    # 1. Setup
    args = parse_args()
    try:
        setup_logging(args.log_fn, args.log_level)

        config = load_and_merge_config(args)
        device, model_path, interpret_path = setup_environment(args)

        # 2. Load Data
        train_df, test_df, valid_df = load_dataframes(args.datafolder)
        protein_dict, ligand_dict = load_or_init_graphs(
            args.datafolder, train_df, test_df, valid_df
        )
        train_loader, valid_loader, test_loader = prepare_dataloaders(
            args, protein_dict, ligand_dict, train_df, test_df, valid_df
        )

        # 3. Initialize Model and Trainer
        mol_deg, clique_deg, prot_deg = get_pna_degrees(args, train_loader, model_path)
        model = initialize_model(config, mol_deg, prot_deg, device, args)
        engine = initialize_trainer(
            model, config, train_loader, args, device, model_path
        )

        # 4. Run Training
        run_training(engine, train_loader, valid_loader, test_loader, args)

        # 5. Run Final Evaluation
        run_evaluation(
            model, model_path, test_df, test_loader, interpret_path, ligand_dict, args
        )

    except Exception as e:
        logger.exception("Script failed: %s", e)
        sys.exit(1)

    logger.info("Script finished.")


if __name__ == "__main__":
    main()
