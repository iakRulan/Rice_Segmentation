"""Pull weights from SeetaCloud server via SFTP (paramiko) with resume.
Skips files already present with matching size. Prints per-file progress.
Env: SSH_HOST, SSH_PORT, SSH_USER, SSH_PASS
"""
import os
import sys
import time

import paramiko

LOCAL = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'weights'))
REMOTE = '/root/crop_segmentation/weights'


def main():
    host = os.environ['SSH_HOST']
    port = int(os.environ['SSH_PORT'])
    user = os.environ['SSH_USER']
    pw = os.environ['SSH_PASS']

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print(f'connecting {user}@{host}:{port} ...', flush=True)
    client.connect(hostname=host, port=port, username=user, password=pw, timeout=20)
    sftp = paramiko.SFTPClient.from_transport(client.get_transport())
    print('connected', flush=True)

    entries = sorted(sftp.listdir_attr(REMOTE), key=lambda x: x.filename)
    os.makedirs(LOCAL, exist_ok=True)
    total = 0
    done = 0
    skipped = 0
    for e in entries:
        name = e.filename
        if not (name.endswith('.pth') or name.endswith('.json')):
            continue
        dst = os.path.join(LOCAL, name)
        if os.path.exists(dst) and os.path.getsize(dst) == e.st_size:
            skipped += 1
            print(f'skip {name} (exists)', flush=True)
            continue
        total += e.st_size
        t0 = time.time()
        sftp.get(f'{REMOTE}/{name}', dst)
        dt = time.time() - t0
        done += 1
        print(f'ok {name}  {e.st_size/1e6:7.1f} MB  {e.st_size/1e6/max(dt,1e-6):5.1f} MB/s', flush=True)
    print(f'\ndownloaded {done} files ({total/1e6:.0f} MB), skipped {skipped}', flush=True)
    client.close()


if __name__ == '__main__':
    main()
