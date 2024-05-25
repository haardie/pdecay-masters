import os
import paramiko
from scp import SCPClient

def create_client(host, port, user, key_path):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.WarningPolicy)
    print(f"Connecting to {host} as {user} using key {key_path}")
    client.connect(host, port, username=user, key_filename=key_path)
    print(f"Connected to {host}")
    return client

def list_dir(dir_path):
    print(f"Listing contents of {dir_path}:")
    for line in os.listdir(dir_path):
        print(line)

def list_first_numeric_subdir(base_dir):
    subdirs = os.listdir(base_dir)
    numeric_subdirs = [sub for sub in subdirs if sub.isdigit()]

    if numeric_subdirs:
        first_num_subdir = os.path.join(base_dir, numeric_subdirs[0])
        print(f"Listing contents of the first numeric subdir: {first_num_subdir}")
        list_dir(first_num_subdir)
    else:
        print(f"No numeric subdirs found in {base_dir}")

def file_exists(ssh, file_path, size):
    stdin, stdout, stderr = ssh.exec_command(f'stat -c%s {file_path}')
    remote_size = stdout.read().strip()
    return remote_size and int(remote_size) == size

def transfer_files(src_base, src_subdir, dest_ssh, dest_dir):
    src_dir = os.path.join(src_base, src_subdir)
    subdirs = os.listdir(src_dir)
    numeric_subdirs = [sub for sub in subdirs if sub.isdigit()]

    file_count = 0
    for num_subdir in numeric_subdirs:
        num_subdir_path = os.path.join(src_dir, num_subdir)
        print(f"Checking numeric subdir: {num_subdir_path}")

        files = os.listdir(num_subdir_path)

        with SCPClient(dest_ssh.get_transport()) as dest_scp:
            for file in files:
                if file.endswith('.h5'):
                    full_path = f"{num_subdir_path}/{file}"
                    print(f"Checking file {full_path}")

                    src_size = os.stat(full_path).st_size
                    dest_path = os.path.join(dest_dir, file)

                    if file_exists(dest_ssh, dest_path, src_size):
                        print(f"File {file} already exists on destination and is the same size. Skipping.")
                        continue

                    print(f"Transferring file {full_path} to {dest_dir}")
                    dest_scp.put(full_path, dest_path)
                    file_count += 1
    print(f"Transferred {file_count} files from {src_dir} to {dest_dir}")

dest_user = 'gartmann'
dest_ip = 'helios.fjfi.cvut.cz'
dest_key = '/home/gartman/.ssh/id_ed25519'

base_dir = '/mnt/nfs19/pec/dune/pdk/franc_anna/data/jobs/nopresel/'
atmonu_src = os.path.join(base_dir, 'AtmoNu')
pdk_src = os.path.join(base_dir, 'PDK')

atmonu_dest = '/mnt/lustre/helios-shared/GAMS/dune/pdk-atmonu-h5/atmonu'
pdk_dest = '/mnt/lustre/helios-shared/GAMS/dune/pdk-atmonu-h5/pdk'

try:
    dest_ssh = create_client(dest_ip, 22, dest_user, dest_key)

    transfer_files(base_dir, 'AtmoNu', dest_ssh, atmonu_dest)
    transfer_files(base_dir, 'PDK', dest_ssh, pdk_dest)

    dest_ssh.close()
    print("SSH connection closed.")
    print("File transfer completed successfully.")

except Exception as e:
    print(f"An error occurred: {e}")
