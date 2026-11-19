# Kaggle kernel (private): spiral fit + snap of PHerc0490A z 7000-7750 on a T4, with NovaCorpAI's pod lane
# (github.com/NovaCorpAI/pherc0800-band-screen @ 0ddc5f16, MIT; villa spiral fitter 0fb38c45, MIT).
# Changes: the lasagna field comes from SharpMeow/lasagna-fields on Hugging Face (the open data has none for this scan),
# the umbilicus is the one built with the same lane on a CPU server (embedded below), and the steps are shortened.
# The fitter needs CUDA, so this part runs here. Output: results.tar (snapped windings as tifxyz) + logs.
import base64, os, shutil, subprocess, time
EMBEDDED = {}
REPO, COMMIT = 'https://github.com/NovaCorpAI/pherc0800-band-screen.git', '0ddc5f163315f4eb4b2c6ea329bc1b746eada5cf'

def sh(cmd, check=True):
    print('$', cmd[:300], flush=True)
    return subprocess.run(cmd, shell=True, check=check)

for p in ('/kaggle/working', '/kaggle/tmp', '/tmp', '/kaggle/temp'):
    if os.path.isdir(p):
        u = shutil.disk_usage(p); print(p, 'free GB', u.free // 2**30, flush=True)
W = max([p for p in ('/kaggle/tmp', '/tmp') if os.path.isdir(p)], key=lambda p: shutil.disk_usage(p).free) + '/w'
os.makedirs(W, exist_ok=True)
sh('nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; nproc; free -g | head -2')
src = f'{W}/src'
sh(f'git clone -q --filter=blob:none --sparse {REPO} {src} && cd {src} && git sparse-checkout set pod && git checkout -q {COMMIT}')
PKG = f'{W}/lane0800'
shutil.rmtree(PKG, ignore_errors=True); shutil.copytree(f'{src}/pod', PKG)
LANE = f'{W}/lane'; os.makedirs(f'{LANE}/stages', exist_ok=True)
for name, b64 in EMBEDDED.items():
    dst = f'{PKG}/{name}' if name.endswith('.py') else f'{LANE}/{name}'
    open(dst, 'wb').write(base64.b64decode(b64))
open(f'{LANE}/stages/02_umb.ok', 'w').close()       # umbilicus embedded (built by build_umb.py + smooth_umb.py, same lane)
s = open(f'{PKG}/pod_data.sh').read()
a, b = s.index('LP=$('), s.index('echo "lasagna done')
open(f'{PKG}/pod_data.sh', 'w').write(s[:a] + '$SPY $PKG/fetch_lasagna_hf.py $S $Z0 $Z1 $D/lasagna_inputs || exit 1\n' + s[b:])
open(f'{PKG}/lane.env', 'w').write('SCROLL=PHerc0490A\nZ0=7000\nZ1=7750\nTB=b0490\nUM=8.64\nSENSE=AB\nMAXW=20\n'
                                   'SHORT_N=3000\nSTEPS=6000\nNEED_GB=40\n')
env = dict(os.environ, W=W, LANE=LANE)
t0 = time.time()
p = subprocess.Popen(f'bash {PKG}/pod_lane.sh', shell=True, env=env)
while p.poll() is None:
    time.sleep(300)
    sh(f'echo "--- {round((time.time() - t0) / 60)} min"; tail -3 {LANE}/logs/lane.log; '
       f'for f in {LANE}/logs/fit_*.log; do [ -f $f ] && echo $f && tail -c 400 $f | tr "\\r" "\\n" | tail -2; done; '
       f'nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; df -h {W} | tail -1', check=False)
print('lane exit', p.returncode, round((time.time() - t0) / 60), 'min', flush=True)
OUT = '/kaggle/working/out'; os.makedirs(OUT, exist_ok=True)
sh(f'cp {LANE}/results.tar {OUT}/ 2>/dev/null; cp -r {LANE}/logs {OUT}/lane_logs; cp {LANE}/timing.tsv {LANE}/sense.txt {OUT}/ 2>/dev/null; '
   f'cp {LANE}/FAILED {OUT}/ 2>/dev/null; ls -la {OUT}', check=False)
