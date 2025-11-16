"""
Quick test of MCTS implementation
"""

import chess
from src.utils import CHESS_MODEL, MCTS

def test_mcts():
    """Test MCTS with current model"""
    print("\n" + "="*60)
    print("TESTING MCTS")
    print("="*60 + "\n")

    if CHESS_MODEL is None:
        print("Error: No model loaded!")
        return

    # Create MCTS instance with small number of simulations for testing
    mcts = MCTS(CHESS_MODEL, num_simulations=10, temperature=1.0)

    # Test position
    board = chess.Board()
    print("Testing from starting position:")
    print(board)
    print()

    # Run MCTS search
    print("Running MCTS with 10 simulations...")
    action_probs = mcts.search(board)

    print("\nTop 5 moves by MCTS probability:")
    sorted_moves = sorted(action_probs.items(), key=lambda x: x[1], reverse=True)
    for i, (move, prob) in enumerate(sorted_moves[:5]):
        print(f"  {i+1}. {move.uci()}: {prob:.4f}")

    # Get best move
    best_move = mcts.get_best_move(board)
    print(f"\nBest move: {best_move.uci()}")

    print("\n" + "="*60)
    print("MCTS TEST COMPLETE!")
    print("="*60 + "\n")

if __name__ == "__main__":
    test_mcts()
