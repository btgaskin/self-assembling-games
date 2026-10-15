"""Stable bibliographic identities, protocol locators and implementation scope."""
import json
from importlib.resources import files


def bibliography():
    return json.loads(files(__package__).joinpath('data/sources.json').read_text())


def references(identifiers):
    requested = set(identifiers)
    return [entry for entry in bibliography() if requested.intersection(
        {entry['id'], entry.get('doi', ''), entry.get('url', ''),
         'https://doi.org/' + entry.get('doi', '')})]
