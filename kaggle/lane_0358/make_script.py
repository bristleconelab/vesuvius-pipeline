"""Build script.py (a single-file Kaggle kernel) with the official PHerc0358 umbilicus embedded."""
import base64, pathlib
here = pathlib.Path(__file__).parent
files = {n: base64.b64encode((here / n).read_bytes()).decode() for n in ('umb_PHerc0358_official.json',)}
body = (here / 'lane.py').read_text()
(here / 'script.py').write_text(body.replace('EMBEDDED = {}', 'EMBEDDED = ' + repr(files)))
print('script.py written')
