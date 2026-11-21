"""Write one single-file Kaggle script per target from lane.py (the only source to edit)."""
import pathlib

here = pathlib.Path(__file__).parent
body = (here / 'lane.py').read_text()
anchor = "TARGET = '306b'\n"
assert body.count(anchor) == 1, 'TARGET anchor not found once'
for target in ('306b', '0358'):
    (here / target / 'script.py').write_text(body.replace(anchor, f"TARGET = '{target}'\n"))
    print(f'{target}/script.py written')
