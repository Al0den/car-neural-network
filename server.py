import paramiko
import os

# SSH server details
SSH_HOST = '144.21.58.74'
SSH_PORT = 22
SSH_USERNAME = 'ubuntu'
PRIVATE_KEY_PATH = '/Users/alois/.ssh/id_ed25519'  # Specify the path to your private key

# Local directory to copy files from
LOCAL_DIR_TO_COPY = './data/train/'  # Change this to the directory you want to copy

# Remote directory to copy files into
REMOTE_DIR = '/home/ubuntu/car-neural-network/data/train/'  # Change this to the destination directory on the server

# Create an SSH client
ssh_client = paramiko.SSHClient()
ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

# Load the private key for authentication
private_key = paramiko.Ed25519Key(filename=PRIVATE_KEY_PATH)

try:
    # Connect to the SSH server
    ssh_client.connect(SSH_HOST, port=SSH_PORT, username=SSH_USERNAME, pkey=private_key)

    # Create an SFTP client
    sftp = ssh_client.open_sftp()

    # Recursive function to copy a local directory to a remote server
    def copy_dir(local_path, remote_path):
        for root, dirs, files in os.walk(local_path):
            for file in files:
                if not file.startswith('best_agent_'):
                    local_file_path = os.path.join(root, file)
                    remote_file_path = os.path.join(remote_path, local_file_path.replace(local_path, '').lstrip(os.path.sep))
                    sftp.put(local_file_path, remote_file_path)
                    print("Copied file to: " + remote_file_path)

    # Call the function to copy the local directory to the server
    copy_dir(LOCAL_DIR_TO_COPY, REMOTE_DIR)

    # Check if 'car' screen exists
    ssh_client.connect(SSH_HOST, port=SSH_PORT, username=SSH_USERNAME, pkey=private_key)

    # Kill the existing 'car' screen session if it exists
    kill_screen_command = 'screen -X -S car quit\n'
    ssh_client.exec_command(kill_screen_command)

    # Create a new 'car' screen session
    create_screen_command = 'screen -dmS car\n'
    ssh_client.exec_command(create_screen_command)

    # Change directory within the 'car' screen session
    cd_within_screen_command = 'screen -S car -X stuff "cd /home/ubuntu/car-neural-network\n"\n'
    ssh_client.exec_command(cd_within_screen_command)

    # Send a command to execute within the new 'car' screen session
    command_within_screen = 'screen -S car -X stuff "python3 src/main.py\n2\n4\n"\n'
    ssh_client.exec_command(command_within_screen)

except Exception as e:
    print(f"Error: {str(e)}")
finally:
    # Close the SSH connection
    ssh_client.close()