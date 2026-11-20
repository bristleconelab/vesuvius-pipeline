# Kaggle kernel (private): spiral fit + snap of PHerc0358 z 7000-7750 on 2x T4, with NovaCorpAI's pod lane
# (github.com/NovaCorpAI/pherc0800-band-screen @ 0ddc5f16, MIT; villa spiral fitter 0fb38c45, MIT).
# PHerc0358 is a 9.362 um / 113 keV scan (the protocol the Studio ink model was trained on) with an official manual
# umbilicus and an official lasagna prediction, so no auto-umbilicus and no third-party lasagna are needed.
# Changes to the pod lane: the official umbilicus is used (stage 02 skipped); the CW and ACW short fits run in parallel,
# one per GPU, each with its own cache and its own hard-linked copy of the tracks (fallback: sequential retry);
# 20000 steps. Output: results.tar (snapped windings as tifxyz) + logs.
import base64, os, shutil, subprocess, time
EMBEDDED = {'umb_PHerc0358_official.json': 'ewoiY29udHJvbF9wb2ludHMiOiBbCnsKIngiOiAzNzMxLjAsCiJ5IjogMzcxMy4wLAoieiI6IDEyMTYuMAp9LAp7CiJ4IjogMzcwOS4wLAoieSI6IDM2ODkuMCwKInoiOiAxMzI4LjAKfSwKewoieCI6IDM3OTIuMCwKInkiOiAzNTk1LjAsCiJ6IjogMTQ0MC4wCn0sCnsKIngiOiAzODAwLjAsCiJ5IjogMzU4NC4wLAoieiI6IDE1NTIuMAp9LAp7CiJ4IjogMzc5NS4wLAoieSI6IDM1NTYuMCwKInoiOiAxNjY0LjAKfSwKewoieCI6IDM3NTQuMCwKInkiOiAzNTcwLjAsCiJ6IjogMTc2OC4wCn0sCnsKIngiOiAzNjQ2LjAsCiJ5IjogMzc2My4wLAoieiI6IDE5NTIuMAp9LAp7CiJ4IjogMzU4Ni4wLAoieSI6IDM4NzguMCwKInoiOiAyMTM2LjAKfSwKewoieCI6IDM0OTUuMCwKInkiOiAzOTY1LjAsCiJ6IjogMjMyMC4wCn0sCnsKIngiOiAzMzc4LjAsCiJ5IjogNDA1Mi4wLAoieiI6IDI1MTIuMAp9LAp7CiJ4IjogMzMyNC4wLAoieSI6IDQxMjUuMCwKInoiOiAyNjk2LjAKfSwKewoieCI6IDMyNjguMCwKInkiOiA0MTY3LjAsCiJ6IjogMjg4MC4wCn0sCnsKIngiOiAzMzE4LjAsCiJ5IjogNDE5NC4wLAoieiI6IDMwNzIuMAp9LAp7CiJ4IjogMzM0MC4wLAoieSI6IDQyMTMuMCwKInoiOiAzMjU2LjAKfSwKewoieCI6IDMzNzMuMCwKInkiOiA0MjI3LjAsCiJ6IjogMzQ0MC4wCn0sCnsKIngiOiAzMzc2LjAsCiJ5IjogNDIyNC4wLAoieiI6IDM2MjQuMAp9LAp7CiJ4IjogMzM5OC4wLAoieSI6IDQyMjcuMCwKInoiOiAzODA4LjAKfSwKewoieCI6IDM0NjMuMCwKInkiOiA0MjIyLjAsCiJ6IjogNDAwMC4wCn0sCnsKIngiOiAzNDg2LjAsCiJ5IjogNDIwNC4wLAoieiI6IDQxODQuMAp9LAp7CiJ4IjogMzU1OS4wLAoieSI6IDQxMDMuMCwKInoiOiA0MzY4LjAKfSwKewoieCI6IDM2NjIuMCwKInkiOiA0MDE2LjAsCiJ6IjogNDU2MC4wCn0sCnsKIngiOiAzNzkxLjAsCiJ5IjogMzk2Ni4wLAoieiI6IDQ3NDQuMAp9LAp7CiJ4IjogMzg5OS4wLAoieSI6IDM5MjUuMCwKInoiOiA0OTI4LjAKfSwKewoieCI6IDM5OTguMCwKInkiOiAzODA0LjAsCiJ6IjogNTExMi4wCn0sCnsKIngiOiA0MTY4LjAsCiJ5IjogMzczNS4wLAoieiI6IDUyOTYuMAp9LAp7CiJ4IjogNDI2Ny4wLAoieSI6IDM2ODcuMCwKInoiOiA1NDg4LjAKfSwKewoieCI6IDQyNTkuMCwKInkiOiAzNjExLjAsCiJ6IjogNTY3Mi4wCn0sCnsKIngiOiA0MzAyLjAsCiJ5IjogMzU2Ny4wLAoieiI6IDU4NTYuMAp9LAp7CiJ4IjogNDM1Ny4wLAoieSI6IDM1NzQuMCwKInoiOiA2MDQ4LjAKfSwKewoieCI6IDQyNjYuMCwKInkiOiAzNjczLjAsCiJ6IjogNjIzMi4wCn0sCnsKIngiOiA0MzAwLjAsCiJ5IjogMzY5My4wLAoieiI6IDY0MTYuMAp9LAp7CiJ4IjogNDM3Ni4wLAoieSI6IDM2NzAuMCwKInoiOiA2NjAwLjAKfSwKewoieCI6IDQ0MzcuMCwKInkiOiAzNjcyLjAsCiJ6IjogNjc4NC4wCn0sCnsKIngiOiA0NDk0LjAsCiJ5IjogMzYzMS4wLAoieiI6IDY5NzYuMAp9LAp7CiJ4IjogNDU0OS4wLAoieSI6IDM2MDMuMCwKInoiOiA3MTYwLjAKfSwKewoieCI6IDQ2NDguMCwKInkiOiAzNjQ0LjAsCiJ6IjogNzM0NC4wCn0sCnsKIngiOiA0Njk4LjAsCiJ5IjogMzYyOS4wLAoieiI6IDc1MjguMAp9LAp7CiJ4IjogNDY4Ny4wLAoieSI6IDM2NDAuMCwKInoiOiA3NzEyLjAKfSwKewoieCI6IDQ3NDYuMCwKInkiOiAzNjQwLjAsCiJ6IjogNzg5Ni4wCn0sCnsKIngiOiA0Nzc4LjAsCiJ5IjogMzY2OS4wLAoieiI6IDgwODAuMAp9LAp7CiJ4IjogNDYyMy4wLAoieSI6IDM4MTYuMCwKInoiOiA4MjcyLjAKfSwKewoieCI6IDQ1OTcuMCwKInkiOiAzOTA1LjAsCiJ6IjogODQ1Ni4wCn0sCnsKIngiOiA0NTgyLjAsCiJ5IjogMzkwNC4wLAoieiI6IDg2NDAuMAp9LAp7CiJ4IjogNDU0Mi4wLAoieSI6IDM5NDcuMCwKInoiOiA4ODI0LjAKfSwKewoieCI6IDQ1NDkuMCwKInkiOiAzOTg3LjAsCiJ6IjogOTAwOC4wCn0sCnsKIngiOiA0NTQ3LjAsCiJ5IjogMzk4Ny4wLAoieiI6IDkyMDAuMAp9LAp7CiJ4IjogNDQ3NC4wLAoieSI6IDQxMTYuMCwKInoiOiA5Mzg0LjAKfSwKewoieCI6IDQwOTAuMCwKInkiOiA0MTY2LjAsCiJ6IjogOTU2OC4wCn0sCnsKIngiOiAzODM0LjAsCiJ5IjogMzk4MC4wLAoieiI6IDk3NTIuMAp9LAp7CiJ4IjogMzI5MS4wLAoieSI6IDM4MzQuMCwKInoiOiA5OTQ0LjAKfSwKewoieCI6IDMwMTUuMCwKInkiOiAzNzQ5LjAsCiJ6IjogMTAxMjguMAp9LAp7CiJ4IjogMjc5NC4wLAoieSI6IDM5MzAuMCwKInoiOiAxMDMxMi4wCn0sCnsKIngiOiAyNTU0LjAsCiJ5IjogNDA1Ni4wLAoieiI6IDEwNDk2LjAKfSwKewoieCI6IDI0NDIuMCwKInkiOiA0Mjc3LjAsCiJ6IjogMTA2ODguMAp9LAp7CiJ4IjogMjQwNC4wLAoieSI6IDQ0MzIuMCwKInoiOiAxMDg3Mi4wCn0sCnsKIngiOiAyMjExLjAsCiJ5IjogNDQyMy4wLAoieiI6IDExMDU2LjAKfSwKewoieCI6IDIxMzIuMCwKInkiOiA0MzcwLjAsCiJ6IjogMTEyNDguMAp9LAp7CiJ4IjogMjA2MC4wLAoieSI6IDQ0MTQuMCwKInoiOiAxMTQzMi4wCn0sCnsKIngiOiAxOTcwLjAsCiJ5IjogNDQ4MC4wLAoieiI6IDExNjE2LjAKfSwKewoieCI6IDE5MzkuMCwKInkiOiA0NTU5LjAsCiJ6IjogMTE4MDAuMAp9LAp7CiJ4IjogMTk4NC4wLAoieSI6IDQ2MzguMCwKInoiOiAxMTk4NC4wCn0sCnsKIngiOiAyMDA4LjAsCiJ5IjogNDU5OS4wLAoieiI6IDEyMTc2LjAKfSwKewoieCI6IDIwMjEuMCwKInkiOiA0NjMwLjAsCiJ6IjogMTIzNjAuMAp9LAp7CiJ4IjogMjA4My4wLAoieSI6IDQ2MTcuMCwKInoiOiAxMjU0NC4wCn0sCnsKIngiOiAyMTE0LjAsCiJ5IjogNDY2MS4wLAoieiI6IDEyNzM2LjAKfSwKewoieCI6IDIxMDAuMCwKInkiOiA0NzIzLjAsCiJ6IjogMTI5MjAuMAp9LAp7CiJ4IjogMjA3MC4wLAoieSI6IDQ3MjcuMCwKInoiOiAxMzA0MC4wCn0sCnsKIngiOiAyMDg3LjAsCiJ5IjogNDc0NC4wLAoieiI6IDEzMTUyLjAKfSwKewoieCI6IDIxMDAuMCwKInkiOiA0NzcyLjAsCiJ6IjogMTMyNjQuMAp9LAp7CiJ4IjogMjA5Ny4wLAoieSI6IDQ3NzUuMCwKInoiOiAxMzM3Ni4wCn0sCnsKIngiOiAyMDk3LjAsCiJ5IjogNDc3NS4wLAoieiI6IDEzNDg4LjAKfSwKewoieCI6IDIxNTYuMCwKInkiOiA0ODU0LjAsCiJ6IjogMTM2MDAuMAp9LAp7CiJ4IjogMjE0OS4wLAoieSI6IDQ4MzAuMCwKInoiOiAxMzcxMi4wCn0sCnsKIngiOiAyMTQyLjAsCiJ5IjogNDg0OC4wLAoieiI6IDEzODMyLjAKfSwKewoieCI6IDIyODMuMCwKInkiOiA0NzM4LjAsCiJ6IjogMTM5NTIuMAp9LAp7CiJ4IjogMjI4My4wLAoieSI6IDQ3MzguMCwKInoiOiAxNDA2NC4wCn0KXSwKIm1ldGFkYXRhIjogewoic291cmNlIjogIlBIZXJjMDM1OC9yZXByZXNlbnRhdGlvbnMvdW1iaWxpY3VzLzIwMjUwODIxMTUxNzM3LXVtYmlsaWN1cy0yMDI2MDkyMjIzNTYzMS5qc29uIChvZmZpY2lhbCwgbWFudWFsIHBvaW50cykiCn0KfQ=='}
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
