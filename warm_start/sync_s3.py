from rllab import config
import os
import argparse
import ast

def sync_s3(folder=None, all=False, dry=False, local_dir=None):

    remote_dir = config.AWS_S3_PATH

    if not local_dir:
        local_dir = os.path.join(config.LOG_DIR, "s3")
    else:
        local_dir = os.path.join(config.PROJECT_PATH, local_dir)  # everything after rllab/ has to be added (ie 'data/')

    if folder:
        remote_dir = os.path.join(remote_dir, folder)
        local_dir = os.path.join(local_dir, folder)
        print (remote_dir)
    if all:
        command = ("""aws s3 sync {remote_dir} {local_dir} --content-type "UTF-8" """.format(local_dir=local_dir, remote_dir=remote_dir))
    else:
        command = ("""aws s3 sync {remote_dir} {local_dir} --exclude '*debug.log' --exclude '*stdout.log' --exclude '*stdouterr.log'  --content-type "UTF-8" """.format(local_dir=local_dir, remote_dir=remote_dir))
    if dry:
        print(command)
    else:
        os.system(command)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=str, default=None, nargs='?')
    parser.add_argument('--all', '-a', action='store_true', default=False, help="add flag to downlowd all log files")
    parser.add_argument('--dry', type=ast.literal_eval, default=False)
    parser.add_argument('--local_dir', type=str, default=None)
    args = parser.parse_args()
    sync_s3(folder=args.folder, all=args.all, dry=args.dry, local_dir=args.local_dir)
