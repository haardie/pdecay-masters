import os
import csv
from pathlib import Path
import logging
import wandb

logging.basicConfig(
    level=logging.INFO, 
    format='[%(levelname)s] %(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

signal_root = "/mnt/lustre/helios-shared/GAMS/dune/pdk-root/pdk_decays"
background_root = "/mnt/lustre/helios-shared/GAMS/dune/pdk-root/atmonu_decays"

logging.info("Listing signal decay directories in: %s", signal_root)
signal_decay_dirs = [
    os.path.join(signal_root, dir_name)
    for dir_name in os.listdir(signal_root)
    if os.path.isdir(os.path.join(signal_root, dir_name))
]
logging.info("Found %d signal decay directories", len(signal_decay_dirs))

logging.info("Listing background decay directories in: %s", background_root)
background_decay_dirs = [
    os.path.join(background_root, dir_name)
    for dir_name in os.listdir(background_root)
    if os.path.isdir(os.path.join(background_root, dir_name))
]
logging.info("Found %d background decay directories", len(background_decay_dirs))


decay_dirs = signal_decay_dirs + background_decay_dirs
# decay_dirs = [decay_dirs[0]]  # For testing

PLANE_IDX = 0

OUTPUT_FILE = "metadata_plane0.csv"

def collect_metadata(decay_paths, plane_idx, output_file):

    count = 0
    logging.info("Starting metadata collection from %d decay directory(ies).", len(decay_paths))
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file_path", "class_label", "decay_mode"])

        for decay_dir in decay_paths:
            decay_dir_path = Path(decay_dir)
            if not decay_dir_path.is_dir():
                logging.debug("Skipping non-directory path: %s", decay_dir)
                continue

            class_label = 1 if "pdk_decays" in str(decay_dir) else 0

            decay_mode = decay_dir_path.name
            logging.debug("Processing decay directory: %s (label: %d, mode: %s)",
                          decay_dir, class_label, decay_mode)

            for event_dir in decay_dir_path.iterdir():
                if not event_dir.is_dir():
                    continue

                plane_path = event_dir / f"plane{plane_idx}"
                if plane_path.is_dir():
                    for npz_file in plane_path.iterdir():
                        if npz_file.suffix == ".npz":
                            writer.writerow([str(npz_file), class_label, decay_mode])
                            count += 1

    logging.info("Finished scanning. Collected metadata for %d .npz file(s).", count)


if __name__ == "__main__":
    wandb_run_name = f"metadata_plane{PLANE_IDX}"
    run = wandb.init(project="dune_data", name=wandb_run_name)
    logging.info("Starting metadata collection.")
    logging.info("Searching for plane index: plane%d", PLANE_IDX)
    logging.info("Output CSV file will be: %s", OUTPUT_FILE)

    collect_metadata(decay_dirs, PLANE_IDX, OUTPUT_FILE)

    logging.info("Done. Metadata saved to %s", OUTPUT_FILE)
    run.finish()
