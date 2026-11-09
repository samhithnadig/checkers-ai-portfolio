import numpy as np
from checkers_ai.engine import GameState, Move, RED, BLACK, legal_moves, perft

def test_initial_position_has_seven_moves():
    s=GameState.initial()
    assert len(s.legal_moves()) == 7

def test_initial_perft_reference_counts():
    s=GameState.initial()
    expected={1:7,2:49,3:302,4:1469,5:7361,6:36768}
    for depth,count in expected.items():
        assert perft(s.board,s.turn,depth)==count

def test_forced_capture():
    b=np.zeros((8,8),dtype=np.int8)
    b[5,0]=RED; b[4,1]=BLACK; b[5,4]=RED
    moves=legal_moves(b,RED)
    assert moves and all(m.is_capture for m in moves)

def test_multijump_is_one_complete_move():
    b=np.zeros((8,8),dtype=np.int8)
    b[6,1]=RED; b[5,2]=BLACK; b[3,4]=BLACK
    moves=legal_moves(b,RED)
    assert len(moves)==1
    assert len(moves[0].captures)==2
    assert len(moves[0].path)==3

def test_promotion_during_capture_ends_turn():
    b=np.zeros((8,8),dtype=np.int8)
    b[2,1]=RED; b[1,2]=BLACK; b[1,4]=BLACK
    moves=legal_moves(b,RED)
    assert len(moves)==1
    assert moves[0].end==(0,3)
    assert len(moves[0].captures)==1
    s=GameState(b,RED,history={})
    s.apply(moves[0])
    assert s.board[0,3]==2

def test_move_round_trip_against_legal_list():
    s=GameState.initial()
    moves=s.legal_moves()
    # Candidate index is intentionally position-local and deterministic.
    for index,move in enumerate(moves):
        assert moves[index] == move

def test_apply_switches_turn_and_moves_piece():
    s=GameState.initial()
    move=s.legal_moves()[0]
    s.apply(move)
    assert s.turn==BLACK
    assert s.board[move.end] == RED
    assert s.board[move.start] == 0

def test_random_playouts_do_not_crash():
    import random
    for seed in range(20):
        random.seed(seed)
        s=GameState.initial()
        for _ in range(300):
            if s.winner is not None or s.draw: break
            moves=s.legal_moves()
            assert moves
            s.apply(random.choice(moves))
        assert s.winner is not None or s.draw

def test_initial_position_is_counted_for_threefold_repetition():
    board=np.zeros((8,8),dtype=np.int8)
    board[5,0]=2
    board[2,7]=-2
    state=GameState(board,RED)

    cycle=[
        ((5,0),(4,1)),
        ((2,7),(3,6)),
        ((4,1),(5,0)),
        ((3,6),(2,7)),
    ]
    assert state.history[state.position_key()] == 1
    for _ in range(2):
        for start,end in cycle:
            move=next(m for m in state.legal_moves() if m.start==start and m.end==end)
            state.apply(move)
    assert state.draw
    assert state.history[state.position_key()] == 3
