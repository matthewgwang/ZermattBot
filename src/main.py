from .utils import chess_manager, GameContext
from .utils.chess_bot import (
    CHESS_MODEL, 
    get_best_move, 
    calculate_search_depth,
    scores_to_probabilities
)
import chess
import random

# Model loads automatically when chess_bot module is imported


@chess_manager.entrypoint
def test_func(ctx: GameContext):
    """
    Main bot function - called every time bot needs to make a move
    Returns a python-chess Move object (converted to UCI automatically)
    """
    board = ctx.board
    move_num = len(board.move_stack) // 2 + 1
    
    print(f"\n{'='*50}")
    print(f"Move {move_num} ({'White' if board.turn else 'Black'})")
    print(f"{'='*50}")
    
    # Get legal moves
    legal_moves = list(board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available")
    
    # Determine search depth (adaptive based on game phase)
    depth = calculate_search_depth(board)
    print(f"Search depth: {depth}")
    
    # Find best move using minimax search
    best_move, move_scores = get_best_move(board, CHESS_MODEL, depth=depth)
    
    # Convert scores to probabilities and log
    move_probs = scores_to_probabilities(move_scores)
    ctx.logProbabilities(move_probs)
    
    # Fallback to random if search failed
    if best_move is None:
        print("⚠ Search failed, using random move")
        best_move = random.choice(legal_moves)
    else:
        eval_score = move_scores.get(best_move, 0)
        print(f"Selected: {best_move.uci()}")
        print(f"Evaluation: {eval_score:.3f}")
    
    return best_move


@chess_manager.reset
def reset_func(ctx: GameContext):
    """Called when a new game begins"""
    print("\n" + "="*50)
    print("NEW GAME STARTED")
    print("="*50 + "\n")