from checkers_ai.mcts import MCTS, Node

def test_parent_prefers_child_that_is_bad_for_child_to_move():
    # A negative child Q means the opponent is losing; from the parent's
    # perspective this is a good move and must score above a positive child Q.
    losing_for_child = Node(state=None, prior=0.5)
    losing_for_child.visits = 4
    losing_for_child.value_sum = -3.2

    winning_for_child = Node(state=None, prior=0.5)
    winning_for_child.visits = 4
    winning_for_child.value_sum = 3.2

    score_good_for_parent = MCTS._selection_score(losing_for_child, 20, 1.4)
    score_bad_for_parent = MCTS._selection_score(winning_for_child, 20, 1.4)
    assert score_good_for_parent > score_bad_for_parent
