import urllib.request
import urllib.parse
import json
import time
import subprocess
import sys
import os

client_id = '178c6fc778ccc68e1d6a'
device_code = '565f2904f3432ef8994c4ccdcc2721dd3c9f410e'

poll_url = 'https://github.com/login/oauth/access_token'
data = urllib.parse.urlencode({
    'client_id': client_id,
    'device_code': device_code,
    'grant_type': 'urn:ietf:params:oauth:grant-type:device_code'
}).encode()

print("Waiting for user authorization...")

token = None
while True:
    req = urllib.request.Request(poll_url, data=data, headers={'Accept': 'application/json'})
    try:
        with urllib.request.urlopen(req) as response:
            res = json.loads(response.read().decode())
            if 'access_token' in res:
                token = res['access_token']
                print("Token successfully retrieved!")
                break
            elif res.get('error') == 'authorization_pending':
                time.sleep(5)
            elif res.get('error') == 'slow_down':
                time.sleep(res.get('interval', 5) + 2)
            else:
                print("Error:", res)
                sys.exit(1)
    except Exception as e:
        print("Exception during polling:", e)
        time.sleep(5)

if token:
    print("Configuring git remote and pushing...")
    
    # Configure URL
    remote_url = f"https://{token}@github.com/WOOJU-KIM/lumos.git"
    
    git_exe = r"C:\Program Files\Git\cmd\git.exe"
    
    subprocess.run([git_exe, "remote", "set-url", "origin", remote_url], check=True)
    
    print("Pushing to remote...")
    result = subprocess.run([git_exe, "push", "-u", "origin", "main"], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Push completed successfully!")
    else:
        print("❌ Push failed!")
        print(result.stderr)
