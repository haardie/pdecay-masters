import os
import csv
import logging
import shutil
import time
import wandb
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_decay_map(decay_map_pth):
    """Load the decay map from CSV into a dictionary."""
    decay_map = {}
    with open(decay_map_pth, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            event, _, decay = row
            decay_map[event] = decay.strip()
    logging.info(f"Loaded decay map with {len(decay_map)} entries")
    return decay_map

def create_struct(rootdir, cls, decays):
    """Create the directory structure for decays."""
    base_path = f"{rootdir}/{cls}_decays"
    os.makedirs(base_path, exist_ok=True)
    
    for d in decays:
        decay_path = os.path.join(base_path, d)
        os.makedirs(decay_path, exist_ok=True)

def move_event(evt_path, dst):
    """Move an event folder to the correct location."""
    try:
        if not os.path.exists(dst):
            shutil.move(evt_path, dst)  
            # logging.info(f"Moved {evt_path} -> {dst}")
        # else:
            # logging.warning(f"Skipping {evt_path}, {dst} already exists")
    except Exception as e:
        logging.error(f"Error moving {evt_path} -> {dst}: {e}")

def mv_evts(rootdir, cls, decays, max_workers=32):
    """Move events to the appropriate decay directories."""
    decay_map_pth = f'{rootdir}/{cls}-npz/{cls}_decay_map.csv'
    decay_map = load_decay_map(decay_map_pth)

    src_root = f'{rootdir}/{cls}-npz'
    futures = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for batchdir in os.scandir(src_root):
            if not batchdir.is_dir():
                continue

            for evtdir in os.scandir(batchdir.path):
                if not evtdir.is_dir():
                    continue

                evt_key = evtdir.name 
                if evt_key in decay_map:
                    decay = decay_map[evt_key]
                    if decay and decay in decays:
                        dst = f'{rootdir}/{cls}_decays/{decay}/{evt_key}'
                        futures.append(executor.submit(move_event, evtdir.path, dst))
                else:
                    logging.warning(f"Event {evt_key} not found in decay map")

        for future in as_completed(futures):
            future.result()

wandb_run_name = f'move_decays_{time.time()}'
run = wandb.init(project="dune_data", name=wandb_run_name)

rootdir = "/mnt/lustre/helios-shared/GAMS/dune/pdk-root"
cls = "atmonu"
decays_atmonu = ['mu_atmonu', 'n_atmonu', 'el_atmonu', 'p_atmonu', 'other_atmonu']

create_struct(rootdir, cls, decays_atmonu)
mv_evts(rootdir, cls, decays_atmonu, max_workers=16)

run.finish()
