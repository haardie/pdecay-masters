import re
import os
import csv
import logging
import logging.config

# set up logging configuration
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default'],
            'level': 'DEBUG',
            'propagate': True
        }
    }
})

def map_decays(evt_dir: str, log_dir: str) -> dict:
    # mapping decays for each event
    map = {}
    evt_name = os.path.basename(evt_dir)
    evt_batch_no = evt_name.split('_')[2]
    evt_no = evt_name.split('_')[4]  # event number in the format files_PDK_23_larcv_0
    logging.info(f"Mapping decay for event: {evt_name}")

    for log in os.listdir(log_dir):
        if log.endswith('.log'):
            log_path = os.path.join(log_dir, log)
            logname = os.path.basename(log_path)
            log_batch_no = logname.split('_')[2]
            if log_batch_no == evt_batch_no:
                logging.info(f"Matched log file: {logname} with event: {evt_name}")
                # initialize with logname and empty decay_mode
                map[evt_name] = [logname, '']  
                break
    return evt_name, map

def map_to_csv(evt_map: dict, save_pth: str):
    logging.info(f"Saving event map to CSV at: {save_pth}")
    with open(save_pth, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['event', 'log', 'decay_mode'])
        writer.writeheader()
        for event, log_info in evt_map.items():
            if log_info[1]:  # only write rows with a decay mode
                logging.info(f"Writing row: event={event}, log={log_info[0]}, decay_mode={log_info[1]}")
                writer.writerow({'event': event, 'log': log_info[0], 'decay_mode': log_info[1]})
            else:
                logging.warning(f"Skipping row for event {event} due to empty decay mode")

def extract_decay_pdk(log_path: str, decay_map: dict, evt_name: str) -> dict:
    logname = os.path.basename(log_path)
    log_batch_no = logname.split('_')[2]
    evt_batch_no = evt_name.split('_')[2]
    evt_no = evt_name.split('_')[4]

    logging.info(f"Processing log file: {log_path}")
    logging.info(f"log_batch_no: {log_batch_no}, evt_batch_no: {evt_batch_no}, evt_no: {evt_no}")

    if log_batch_no == evt_batch_no:
        kaon_prod = False
        with open(log_path, 'r') as f:
            lines = f.readlines()
            # look for the event number
            for idx, line in enumerate(lines):
                if re.match(fr'\s*{evt_no}\s*\t', line):
                    logging.info(f"Found event number {evt_no} in line: {line.strip()}")
                    if 'p -> K' in line:
                        kaon_prod = True

                if kaon_prod and idx + 1 < len(lines):
                    next_line = lines[idx + 1].strip()
                    if 'K ->' in next_line:
                        match = re.search(r'K -> (.+)', next_line)
                        if match:
                            mode = match.group(1)
                            mode = re.sub(r'\([\d.e+-]+\s*MeV\)', '', mode)  # clean up mode
                            mode = re.sub(r'\s*\+\s*', '_', mode)
                            decay_map[evt_name][1] = mode
                            logging.info(f"Extracted decay mode: {mode} for event: {evt_name}")
                            kaon_prod = False  # reset after extraction
                            break

            if kaon_prod:
                decay_map[evt_name][1] = 'no_Kdecay'
                logging.info(f"No decay mode found for event: {evt_name}, set to 'no_Kdecay'")

    return decay_map

def generalize_atmonu_mode(mode: str) -> str:
    mode = mode.strip()
    logging.info(f"Generalizing mode: {mode}")
    if mode.startswith('n'):
        return 'n'
    elif mode.startswith('el'):
        return 'el'
    elif mode.startswith('mu'):
        return 'mu'
    elif mode.startswith('p'):
        return 'p'
    else:
        return 'other'

def extract_decay_atmonu(log_path: str, decay_map: dict, evt_name: str) -> dict:
    logname = os.path.basename(log_path)
    log_batch_no = logname.split('_')[2]
    evt_batch_no = evt_name.split('_')[2]
    evt_no = evt_name.split('_')[4]

    logging.info(f"Processing log file: {log_path}")
    logging.info(f"log_batch_no: {log_batch_no}, evt_batch_no: {evt_batch_no}, evt_no: {evt_no}")

    if log_batch_no == evt_batch_no:
        with open(log_path, 'r') as f:
            lines = f.readlines()

            # look for the event number
            for idx, line in enumerate(lines):
                if re.match(fr'\s*{evt_no}\s*\t', line):
                    logging.info(f"Found event number {evt_no} in line: {line.strip()}")
                    match = re.search(r'p -> (.+)', line)
                    if match:
                        mode = match.group(1)
                        logging.info(f"Found interaction mode: {mode}")
                        mode = re.sub(r'\([\d.e+-]+\s*MeV\)', '', mode)
                        mode = generalize_atmonu_mode(mode)
                        logging.info(f"Generalized mode: {mode}")
                        decay_map[evt_name][1] = mode
                        break
                    else:
                        logging.info(f"No interaction mode found in line: {line.strip()}")

    logging.info(f"Completed processing log file: {log_path}")
    return decay_map

def main(cls_func_map: dict, cls_log_dir_map: dict, cls_evt_dir_map: dict) -> None:
    logging.info("Starting main function")

    # iterate through each classification ('pdk' or 'atmonu')
    for cls in cls_func_map.keys():
        if cls == 'atmonu':

            logging.info(f"Processing classification: {cls}")
            logging.info("-" * 40)

            log_dir = cls_log_dir_map[cls]
            evt_dir = cls_evt_dir_map[cls]
            extract_func = cls_func_map[cls]

            for evt in os.listdir(evt_dir):
                evt_path = os.path.join(evt_dir, evt)
                if os.path.isdir(evt_path):
                    logging.info(f"Processing batch directory: {evt_path}")
                    logging.info("-" * 20)

                    for subdir in os.listdir(evt_path):
                        subdir_path = os.path.join(evt_path, subdir)
                        if os.path.isdir(subdir_path):
                            logging.info(f"Processing subdirectory: {subdir_path}")
                            logging.info("-" * 10)

                            evt_name, decay_map = map_decays(subdir_path, log_dir)

                            for log_file in decay_map.keys():
                                log_path = os.path.join(log_dir, log_file)
                                log_path = log_path[:-4]
                                log_path += 'evtinfo.log'
                                if os.path.exists(log_path):
                                    logging.info(f"Processing log file: {log_path}")
                                    decay_map = extract_func(log_path, decay_map, evt_name)

                            save_pth = os.path.join(subdir_path, f'{subdir}_decay_map.csv')
                            map_to_csv(decay_map, save_pth)
                            logging.info(f"Saved decay map to: {save_pth}")
                            logging.info("-" * 10)

                    logging.info("-" * 20)

            logging.info("-" * 40)
    logging.info("Completed main function")

# run the main function
cls_func_map = {'pdk': extract_decay_pdk, 'atmonu': extract_decay_atmonu}
cls_log_dir_map = {'pdk': '/mnt/lustre/helios-shared/GAMS/dune/pdk-root/pdk-logs',
                   'atmonu': '/mnt/lustre/helios-shared/GAMS/dune/pdk-root/atmonu-logs'}
cls_evt_dir_map = {'pdk': '/mnt/lustre/helios-shared/GAMS/dune/pdk-root/pdk-npz',
                   'atmonu': '/mnt/lustre/helios-shared/GAMS/dune/pdk-root/atmonu-npz'}

main(cls_func_map, cls_log_dir_map, cls_evt_dir_map)
