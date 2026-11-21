"""Write one single-file Kaggle script per target from lane.py (the only source to edit; hx0358top adds hx0358top.py)."""
import pathlib

here = pathlib.Path(__file__).parent
ANCHOR = "TARGET = '306b'\n"
MAIN = "\n\nif __name__ == '__main__':\n    main()\n"


def build(target):
    body = (here / 'lane.py').read_text()
    assert body.count(ANCHOR) == 1, 'TARGET anchor not found once'
    body = body.replace(ANCHOR, f"TARGET = '{target}'\n")
    if target == 'hx0358top':
        assert body.endswith(MAIN), 'lane.py must end with its main guard'
        body = body[:-len(MAIN)] + '\n' + (here / 'hx0358top.py').read_text()
    return body


if __name__ == '__main__':
    for target in ('306b', '0358', 'hx0358top'):
        (here / target / 'script.py').write_text(build(target))
        print(f'{target}/script.py written')
