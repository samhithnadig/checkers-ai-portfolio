from checkers_ai.engine import GameState
from checkers_ai.features import board_features, move_features

def test_feature_shapes():
    s=GameState.initial(); moves=s.legal_moves()
    assert board_features(s.board,s.turn).shape==(4,8,8)
    assert move_features(moves[0],s.board,s.turn).shape==(10,)
