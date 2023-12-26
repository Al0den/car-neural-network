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

def copy_dir(sftp, local_path, remote_path):
    for root, dirs, files in os.walk(local_path):
        for file in files:
            if not file.startswith('best_agent_'):
                local_file_path = os.path.join(root, file)
                remote_file_path = os.path.join(remote_path, local_file_path.replace(local_path, '').lstrip(os.path.sep))
                sftp.put(local_file_path, remote_file_path)
                print("Copied file to: " + remote_file_path)

try:
    # Connect to the SSH server
    ssh_client.connect(SSH_HOST, port=SSH_PORT, username=SSH_USERNAME, pkey=private_key)

    # Create an SFTP client
    sftp = ssh_client.open_sftp()

    sftp = ssh_client.open_sftp()

    # Copy files from the first directory
    LOCAL_DIR_TO_COPY_1 = './data/train/'  # First directory to copy
    REMOTE_DIR_1 = '/home/ubuntu/car-neural-network/data/train/'  # Destination directory on the server for the first directory
    copy_dir(sftp, LOCAL_DIR_TO_COPY_1, REMOTE_DIR_1)

    confirm_copy = input("Do you want to copy the tracks directory? (y/n): ")
    if confirm_copy.lower() == 'y':
        LOCAL_DIR_TO_COPY_2 = './data/tracks/'  # Second directory to copy
        REMOTE_DIR_2 = '/home/ubuntu/car-neural-network/data/tracks/'  # Destination directory on the server for the second directory
        copy_dir(sftp, LOCAL_DIR_TO_COPY_2, REMOTE_DIR_2)
    else:
        print("Files from the track directory were not copied.")
    git_pull = input("Do you want to pull from git? (y/n): ")
    if git_pull.lower() == 'y':
        # Execute in folder home/ubuntu/car-neural-network
        git_pull_command = 'cd /home/ubuntu/car-neural-network && git pull\n'
        ssh_client.exec_command(git_pull_command)

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