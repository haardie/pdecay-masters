import os
import paramiko
from scp import SCPClient
import argparse


def create_client(host, port, user, key_path):
    """
    Create an SSH client and connect to the specified host.

    Args:
        host (str): The hostname or IP address of the remote server.
        port (int): The port number to connect to on the remote server.
        user (str): The username to use for the SSH connection.
        key_path (str): The path to the private key file for authentication.

    Returns:
        paramiko.SSHClient: The connected SSH client.
    """
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.WarningPolicy)
    print(f"Connecting to {host} as {user} using key {key_path}")
    client.connect(host, port, username=user, key_filename=key_path)
    print(f"Connected to {host}")
    return client


def file_exists(local_path, remote_size):
    """
    Check if a local file exists and matches the specified size.

    Args:
        local_path (str): The path to the local file.
        remote_size (int): The size of the remote file to compare against.

    Returns:
        bool: True if the local file exists and matches the remote size, False otherwise.
    """
    if os.path.exists(local_path):
        local_size = os.stat(local_path).st_size
        return local_size == remote_size
    return False


def transf(src_ssh, src_base, src_subdir, dest_dir):
    """
    Transfer log files from a remote directory to a local directory.

    Args:
        src_ssh (paramiko.SSHClient): The SSH client connected to the remote server.
        src_base (str): The base directory on the remote server.
        src_subdir (str): The subdirectory under the base directory to search for files.
        dest_dir (str): The local directory to transfer files to.

    Returns:
        None
    """
    src_dir = os.path.join(src_base, src_subdir)

    # Execute command on remote to list subdirectories
    stdin, stdout, stderr = src_ssh.exec_command(f'ls -1 {src_dir}')
    subdirs = stdout.read().decode().splitlines()
    numeric_subdirs = [sub for sub in subdirs if sub.isdigit()]

    file_count = 0
    for num_subdir in numeric_subdirs:
        num_subdir_path = os.path.join(src_dir, num_subdir)
        print(f"Checking numeric subdir: {num_subdir_path}")

        # List files in the remote directory
        stdin, stdout, stderr = src_ssh.exec_command(f'ls -1 {num_subdir_path}')
        files = stdout.read().decode().splitlines()

        with SCPClient(src_ssh.get_transport()) as dest_scp:
            for file in files:
                if file.endswith('.log') and 'evt' in file:
                    root_file = [file for file in files if file.endswith('.root')][0]
                    root_basename = os.path.basename(root_file).split('.')[0]

                    full_path = f"{num_subdir_path}/{file}"
                    print(f"Checking file {full_path}")

                    new_filename = f"{root_basename}_{file}"
                    local_file_path = os.path.join(dest_dir, new_filename)

                    if os.path.exists(local_file_path):
                        print(f"File {local_file_path} already exists locally. Skipping.")
                        continue

                    print(f"Transferring file {full_path} to {local_file_path}")

                    try:
                        dest_scp.get(full_path, local_file_path)
                        file_count += 1
                    except Exception as scp_error:
                        print(f"Error transferring file {full_path}: {scp_error}")

    print(f"Transferred {file_count} files.")


# ==================================#
if __name__ == '__main__':
  
    parser = argparse.ArgumentParser(description="Transfer log files from remote server to local.")
    parser.add_argument('--src_user', required=True, help='Username for SSH connection')
    parser.add_argument('--src_ip', required=True, help='IP address of the remote server')
    parser.add_argument('--src_key', required=True, help='Path to SSH private key')
    parser.add_argument('--base_dir', required=True, help='Base directory on the remote server')
    parser.add_argument('--atmonu_dest', required=True, help='Local destination directory for AtmoNu logs')
    parser.add_argument('--pdk_dest', required=True, help='Local destination directory for PDK logs')

    args = parser.parse_args()

    try:
        src_ssh = create_client(args.src_ip, 22, args.src_user, args.src_key)
        transf(src_ssh, args.base_dir, 'AtmoNu', args.atmonu_dest)
        transf(src_ssh, args.base_dir, 'PDK', args.pdk_dest)

    except Exception as e:
        print(e)
    finally:
        src_ssh.close()
        print('Connection closed.')

# ==================================#
