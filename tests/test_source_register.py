from self_assembling_games import model
from self_assembling_games.sources import bibliography, references


def test_register_has_unique_stable_ids_and_resolves_doi():
    entries = bibliography()
    assert len({entry['id'] for entry in entries}) == len(entries)
    for entry in entries:
        assert entry['title'] and entry['authors'] and entry['protocol']
        assert entry in references([entry['doi']])


def test_implemented_source_families_carry_references():
    for preset, identifier in [
        ('oconnor.generalization.binary', 'oconnor2017'),
        ('lacroix.logic.stipulated', 'lacroix2022'),
    ]:
        source = model(preset).to_dict()['source']
        assert identifier in {entry['id'] for entry in source['references']}
