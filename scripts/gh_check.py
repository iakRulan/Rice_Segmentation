import urllib.request, json
import os
TOKEN = os.environ.get('GH_TOKEN', '')

def api(path):
    req = urllib.request.Request('https://api.github.com' + path,
        headers={'Authorization': 'token ' + TOKEN, 'User-Agent': 'x'})
    with urllib.request.urlopen(req) as r:
        return r

try:
    r = api('/user')
    print('scopes header:', r.headers.get('X-OAuth-Scopes'))
    data = json.loads(r.read().decode())
    print('login:', data.get('login'))
    r2 = api('/repos/iakRulan/Rice_Segmentation')
    data2 = json.loads(r2.read().decode())
    print('repo permissions:', data2.get('permissions'))
    print('repo private:', data2.get('private'))
except Exception as e:
    print('ERROR', e)
