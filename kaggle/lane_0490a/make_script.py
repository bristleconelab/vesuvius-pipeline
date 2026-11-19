"""Build script.py (a single-file Kaggle kernel) with the small lane inputs embedded."""
import base64, pathlib
here = pathlib.Path(__file__).parent
files = {n: base64.b64encode((here / n).read_bytes()).decode() for n in
         ('fetch_lasagna_hf.py', 'umb_PHerc0490A_s75.json', 'umb_PHerc0490A_raw.json')}
body = (here / 'lane.py').read_text()
(here / 'script.py').write_text(body.replace('EMBEDDED = {}', 'EMBEDDED = ' + repr(files)))
print('script.py written')
