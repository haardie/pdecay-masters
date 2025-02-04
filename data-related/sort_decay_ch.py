import os
import re
import csv
import logging
from concurrent.futures import ThreadPoolExecutor
import wandb
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

def map_decays(top_dir: str, log_dir: str) -> dict:
    """Maps events to log files."""
    decay_map = {}
    log_index = {log.split("_")[2]: log for log in os.listdir(log_dir) if log.endswith(".log")}

    for batch_dir in os.scandir(top_dir):
        if batch_dir.is_dir():
            for evt_dir in os.scandir(batch_dir.path):
                if evt_dir.is_dir():
                    evt_name = os.path.basename(evt_dir.path)
                    evt_batch_no = evt_name.split("_")[2]
                    if evt_batch_no in log_index:
                        decay_map[evt_name] = [log_index[evt_batch_no], ""]
    return decay_map

def extract_decay_pdk(log_path: str, decay_map: dict, evt_name: str) -> dict:
    logname = os.path.basename(log_path)
    log_batch_no = logname.split("_")[2]
    evt_batch_no = evt_name.split("_")[2]
    evt_no = evt_name.split("_")[4]

    logging.info(f"Processing log file: {log_path}")
    logging.info(
        f"log_batch_no: {log_batch_no}, evt_batch_no: {evt_batch_no}, evt_no: {evt_no}"
    )

    if log_batch_no == evt_batch_no:
        kaon_prod = False
        with open(log_path, "r") as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                if re.match(rf"\s*{evt_no}\s*\t", line):
                    logging.info(f"Found event number {evt_no} in line: {line.strip()}")
                    if "p -> K" in line:
                        kaon_prod = True
                if kaon_prod and idx + 1 < len(lines):
                    next_line = lines[idx + 1].strip()
                    if "K ->" in next_line:
                        match = re.search(r"K -> (.+)", next_line)
                        if match:
                            mode = match.group(1)
                            mode = re.sub(r"\([\d.e+-]+\s*MeV\)", "", mode)
                            mode = re.sub(r"\s*\+\s*", "_", mode)
                            decay_map[evt_name][1] = mode
                            logging.info(
                                f"Extracted decay mode: {mode} for event: {evt_name}"
                            )
                            kaon_prod = False
                            break
            if kaon_prod:
                decay_map[evt_name][1] = "no_Kdecay"
                logging.info(
                    f"No decay mode found for event: {evt_name}, set to 'no_Kdecay'"
                )
    return decay_map

def generalize_atmonu_mode(mode: str) -> str:
    mode = mode.strip()
    # logging.info(f"Generalizing mode: {mode}")
    if mode.startswith("n"):
        return "n_atmonu"
    elif mode.startswith("el"):
        return "el_atmonu"
    elif mode.startswith("mu"):
        return "mu_atmonu"
    elif mode.startswith("p"):
        return "p_atmonu"
    else:
        return "other_atmonu"

def extract_decay_atmonu(log_path: str, decay_map: dict, evt_name: str) -> dict:
    logname = os.path.basename(log_path)
    log_batch_no = logname.split("_")[2]
    evt_batch_no = evt_name.split("_")[2]
    evt_no = evt_name.split("_")[4]

    # logging.info(f"Processing log file: {log_path}")
    # logging.info(
    #     f"log_batch_no: {log_batch_no}, evt_batch_no: {evt_batch_no}, evt_no: {evt_no}"
    # )

    if log_batch_no == evt_batch_no:
        with open(log_path, "r") as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                if re.match(rf"\s*{evt_no}\s*\t", line):
                    # logging.info(f"Found event number {evt_no} in line: {line.strip()}")
                    match = re.search(r"p -> (.+)", line)
                    if match:
                        mode = match.group(1)
                        # logging.info(f"Found interaction mode: {mode}")
                        mode = re.sub(r"\([\d.e+-]+\s*MeV\)", "", mode)
                        mode = generalize_atmonu_mode(mode)
                        # logging.info(f"Generalized mode: {mode}")
                        decay_map[evt_name][1] = mode
                        break
                    # else:
                        # logging.info(
                        #     f"No interaction mode found in line: {line.strip()}"
                        # )

    # logging.info(f"Completed processing log file: {log_path}")
    return decay_map

def parallel_decay_extraction(decay_map: dict, log_dir: str, extract_func, num_workers: int = 64):
    """Processes decay extraction in parallel."""
    def process_event(evt_name):
        log_file = decay_map[evt_name][0]
        log_path = os.path.join(log_dir, log_file)
        if os.path.exists(log_path):
            return extract_func(log_path, decay_map, evt_name)
        return decay_map

    count = 0
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(process_event, decay_map.keys())
        for result in results:
            count += 1
            if count % 10000 == 0:
                logging.info(f"Processed {count} files.")
            decay_map.update(result)

def save_decay_map(decay_map: dict, output_path: str):
    """Saves the decay map to a CSV file."""
    with open(output_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["event", "log", "decay"])
        for evt_name, (log_file, decay) in decay_map.items():
            writer.writerow([evt_name, log_file, decay])

def main(cls_func_map: dict, cls_log_dir_map: dict, cls_evt_dir_map: dict):
    logging.info("Starting main function")
    for cls in ["atmonu", "pdk"]:
        logging.info(f"Processing classification: {cls}")

        log_dir = cls_log_dir_map[cls]
        top_dir = cls_evt_dir_map[cls]
        output_path = os.path.join(top_dir, f"{cls}_decay_map.csv")

        decay_map = map_decays(top_dir, log_dir)
        logging.info(f"Mapped {len(decay_map)} events.")

        extract_func = cls_func_map[cls]
        parallel_decay_extraction(decay_map, log_dir, extract_func, num_workers=8)

        save_decay_map(decay_map, output_path)
        logging.info(f"Saved decay map to: {output_path}")

    logging.info("Completed main function")

cls_func_map = {"pdk": extract_decay_pdk, "atmonu": extract_decay_atmonu}
cls_log_dir_map = {
    "pdk": "/mnt/lustre/helios-shared/GAMS/dune/pdk-root/pdk-logs",
    "atmonu": "/mnt/lustre/helios-shared/GAMS/dune/pdk-root/atmonu-logs",
}
cls_evt_dir_map = {
    "pdk": "/mnt/lustre/helios-shared/GAMS/dune/pdk-root/pdk-npz",
    "atmonu": "/mnt/lustre/helios-shared/GAMS/dune/pdk-root/atmonu-npz",
}

# Run
if __name__ == "__main__":
    wandb_run_name = f"assign_decays_{int(time.time())}"
    run = wandb.init(project="dune_data", name=wandb_run_name)

    main(cls_func_map, cls_log_dir_map, cls_evt_dir_map)
    logging.info("Finished run.")

    run.finish()
