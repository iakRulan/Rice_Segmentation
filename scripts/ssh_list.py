"""Connect to SeetaCloud GPU server via SSH/SFTP (paramiko), list remote dirs.
Usage: python scripts/ssh_list.py --path /root/crop_segmentation/weights
Env: SSH_HOST, SSH_PORT, SSH_USER, SSH_PASS
"""
import argparse
import os

import paramiko


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--path', required=True)
    ap.add_argument('--match', default='', help='only show entries whose name contains this')
    args = ap.parse_args()

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=os.environ['SSH_HOST'],
        port=int(os.environ['SSH_PORT']),
        username=os.environ['SSH_USER'],
        password=os.environ['SSH_PASS'],
        timeout=20,
    )
    sftp = paramiko.SFTPClient.from_transport(client.get_transport())
    try:
        entries = sftp.listdir_attr(args.path)
    except FileNotFoundError:
        print(f'PATH NOT FOUND: {args.path}')
        return
    total = 0
    for e in sorted(entries, key=lambda x: x.filename):
        if args.match and args.match not in e.filename:
            continue
        print(f'{e.filename:55s} {e.st_size/1e6:8.1f} MB')
        total += e.st_size
    print(f'-- {len(entries)} entries, {total/1e6:.1f} MB total')
    client.close()


if __name__ == '__main__':
    main()
