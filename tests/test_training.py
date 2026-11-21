"""Twin fine-tuning details (chesstwin.training)."""
import numpy as np

from chesstwin.training import _split


def test_validation_games_are_spread_over_the_whole_history():
    games = np.repeat(np.arange(100), 30)               # 100 games, 30 moves each, game 0 = newest
    tr, va = _split({"game": games})
    val_games = np.unique(games[va])
    assert len(val_games) == 10 and not np.intersect1d(val_games, games[tr]).size
    assert val_games.min() < 10 and val_games.max() >= 90   # both new and old games are checked


def test_few_games_means_no_validation():
    tr, va = _split({"game": np.repeat(np.arange(5), 10)})
    assert tr.all() and not va.any()
