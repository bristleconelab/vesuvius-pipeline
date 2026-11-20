# Kaggle kernel (private): spiral fit + snap of PHerc0358 z 7000-7750 on 2x T4, with NovaCorpAI's pod lane
# (github.com/NovaCorpAI/pherc0800-band-screen @ 0ddc5f16, MIT; villa spiral fitter 0fb38c45, MIT).
# PHerc0358 is a 9.362 um / 113 keV scan (the protocol the Studio ink model was trained on) with an official manual
# umbilicus and an official lasagna prediction, so no auto-umbilicus and no third-party lasagna are needed.
# Changes to the pod lane: the official umbilicus is used (stage 02 skipped); the CW and ACW short fits run in parallel,
# one per GPU, each with its own cache and its own hard-linked copy of the tracks (fallback: sequential retry);
# 20000 steps. Output: results.tar (snapped windings as tifxyz) + logs.
import base64, os, shutil, subprocess, time
EMBEDDED = {}
REPO, COMMIT = 'https://github.com/NovaCorpAI/pherc0800-band-screen.git', '0ddc5f163315f4eb4b2c6ea329bc1b746eada5cf'
SCROLL, Z0, Z1, TB = 'PHerc0358', 7000, 7750, 'b0358'

def sh(cmd, check=True):
    print('$', cmd[:300], flush=True)
    return subprocess.run(cmd, shell=True, check=check)

def patch(text, old, new):
    assert old in text, f'patch anchor not found: {old[:80]!r}'
    return text.replace(old, new)

for p in ('/kaggle/working', '/kaggle/tmp', '/tmp'):
    if os.path.isdir(p):
        print(p, 'free GB', shutil.disk_usage(p).free // 2**30, flush=True)
W = max([p for p in ('/kaggle/tmp', '/tmp') if os.path.isdir(p)], key=lambda p: shutil.disk_usage(p).free) + '/w'
os.makedirs(W, exist_ok=True)
sh('nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader; nproc; free -g | head -2')
src = f'{W}/src'
sh(f'git clone -q --filter=blob:none --sparse {REPO} {src} && cd {src} && git sparse-checkout set pod && git checkout -q {COMMIT}')
PKG = f'{W}/lane0800'
shutil.rmtree(PKG, ignore_errors=True); shutil.copytree(f'{src}/pod', PKG)
LANE = f'{W}/lane'; os.makedirs(f'{LANE}/stages', exist_ok=True)
umb = base64.b64decode(EMBEDDED['umb_PHerc0358_official.json'])
for n in (f'umb_{SCROLL}_s75.json', f'umb_{SCROLL}_raw.json'):
    open(f'{LANE}/{n}', 'wb').write(umb)
open(f'{LANE}/stages/02_umb.ok', 'w').close()

s = open(f'{PKG}/pod_lane.sh').read()
s = patch(s, '--cache $LANE/cache )', '--cache $LANE/cache_$dsn )')
s = patch(s, 'for f in tracks lasagna_inputs umbilicus.json; do ln -sfn $DS/$S/$f $A/$f; done',
          'for f in lasagna_inputs umbilicus.json; do ln -sfn $DS/$S/$f $A/$f; done; rm -rf $A/tracks; cp -al $DS/$S/tracks $A/tracks')
s = patch(s, '''      runfit $S ${TB}cws FIT_SPIRAL_STOP_AT=$SHORT_N || return 1
      runfit ${S}_ACW ${TB}acws FIT_SPIRAL_STOP_AT=$SHORT_N || return 1''',
'''      runfit $S ${TB}cws CUDA_VISIBLE_DEVICES=0 FIT_SPIRAL_STOP_AT=$SHORT_N & local p1=$!
      sleep 120; runfit ${S}_ACW ${TB}acws CUDA_VISIBLE_DEVICES=1 FIT_SPIRAL_STOP_AT=$SHORT_N & local p2=$!
      local r1=0 r2=0; wait $p1 || r1=$?; wait $p2 || r2=$?
      echo "$(date +%s) parallel short fits rc cw=$r1 acw=$r2" >> $LANE/logs/lane.log
      if [ $r1 != 0 ]; then rm -rf $LANE/fits/*_${TB}cws; runfit $S ${TB}cws FIT_SPIRAL_STOP_AT=$SHORT_N || return 1; fi
      if [ $r2 != 0 ]; then rm -rf $LANE/fits/*_${TB}acws; runfit ${S}_ACW ${TB}acws FIT_SPIRAL_STOP_AT=$SHORT_N || return 1; fi''')
open(f'{PKG}/pod_lane.sh', 'w').write(s)
open(f'{PKG}/lane.env', 'w').write(f'SCROLL={SCROLL}\nZ0={Z0}\nZ1={Z1}\nTB={TB}\nUM=9.362\nSENSE=AB\nMAXW=24\n'
                                   'SHORT_N=4000\nSTEPS=20000\nNEED_GB=60\n')
env = dict(os.environ, W=W, LANE=LANE)
t0 = time.time()
p = subprocess.Popen(f'bash {PKG}/pod_lane.sh', shell=True, env=env)
while p.poll() is None:
    time.sleep(300)
    sh(f'echo "--- {round((time.time() - t0) / 60)} min"; tail -3 {LANE}/logs/lane.log; '
       f'for f in {LANE}/logs/fit_*.log; do [ -f $f ] && echo $f && tail -c 400 $f | tr "\\r" "\\n" | tail -2; done; '
       f'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; free -g | sed -n 2p; df -h {W} | tail -1',
       check=False)
print('lane exit', p.returncode, round((time.time() - t0) / 60), 'min', flush=True)
OUT = '/kaggle/working/out'; os.makedirs(OUT, exist_ok=True)
sh(f'cp {LANE}/results.tar {OUT}/ 2>/dev/null; cp -r {LANE}/logs {OUT}/lane_logs; cp {LANE}/timing.tsv {LANE}/sense.txt {OUT}/ 2>/dev/null; '
   f'cp {LANE}/FAILED {OUT}/ 2>/dev/null; ls -la {OUT}', check=False)
