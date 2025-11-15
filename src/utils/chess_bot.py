# ============================================================================
# CHESS BOT IMPLEMENTATION
# ============================================================================

import torch
import torch.nn as nn
import numpy as np
import chess
import os

# ===== NEURAL NETWORK =====

class ChessNet1500(nn.Module):
    """Lightweight chess network for ~1500 ELO"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(15, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)
        
        self.policy = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4096)
        )
        
        self.value = nn.Sequential(
            nn.Conv2d(128, 32, 1),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        return self.policy(x), self.value(x)


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """Convert chess board to 15-plane tensor"""
    planes = np.zeros((15, 8, 8), dtype=np.float32)
    
    piece_idx = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            rank, file = divmod(square, 8)
            idx = piece_idx[piece.piece_type]
            if piece.color == chess.BLACK:
                idx += 6
            planes[idx, rank, file] = 1.0
    
    # Castling
    if board.has_kingside_castling_rights(chess.WHITE):
        planes[12, 0, 4:8] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        planes[12, 0, 0:4] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        planes[12, 7, 4:8] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        planes[12, 7, 0:4] = 1.0
    
    # En passant
    if board.ep_square is not None:
        rank, file = divmod(board.ep_square, 8)
        planes[13, rank, file] = 1.0
    
    # Turn
    if board.turn == chess.WHITE:
        planes[14, :, :] = 1.0
    
    return torch.FloatTensor(planes).unsqueeze(0)


def move_to_index(move: chess.Move) -> int:
    """Convert move to index"""
    return move.from_square * 64 + move.to_square


# ===== EVALUATION =====

def evaluate_material(board: chess.Board) -> float:
    """Material evaluation with positional bonuses"""
    piece_values = {
        chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
        chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0
    }
    
    score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = piece_values[piece.piece_type]
            score += value if piece.color == board.turn else -value
    
    # Center control
    for sq in [chess.E4, chess.D4, chess.E5, chess.D5]:
        piece = board.piece_at(sq)
        if piece and piece.color == board.turn:
            score += 10
    
    # Mobility
    score += len(list(board.legal_moves)) * 2
    
    return score / 1000.0


def evaluate_position(board: chess.Board, model) -> float:
    """Evaluate using neural network or material"""
    if board.is_game_over():
        result = board.result()
        if result == "1-0":
            return 10.0 if board.turn == chess.WHITE else -10.0
        elif result == "0-1":
            return -10.0 if board.turn == chess.WHITE else 10.0
        return 0.0
    
    if model is not None:
        try:
            board_tensor = board_to_tensor(board)
            with torch.no_grad():
                _, value = model(board_tensor)
                return value.item()
        except:
            pass
    
    return evaluate_material(board)


# ===== SEARCH =====

def minimax(board: chess.Board, depth: int, alpha: float, beta: float, 
            maximizing: bool, model) -> float:
    """Minimax with alpha-beta pruning"""
    if depth == 0 or board.is_game_over():
        return evaluate_position(board, model)
    
    legal_moves = list(board.legal_moves)
    
    # Move ordering
    def move_priority(move):
        score = 0
        if board.is_capture(move):
            victim = board.piece_at(move.to_square)
            attacker = board.piece_at(move.from_square)
            if victim:
                score += 100 * victim.piece_type
            if attacker:
                score -= attacker.piece_type
        if move.promotion:
            score += 800
        board.push(move)
        if board.is_check():
            score += 50
        board.pop()
        return score
    
    legal_moves.sort(key=move_priority, reverse=True)
    
    if maximizing:
        max_eval = float('-inf')
        for move in legal_moves:
            board.push(move)
            eval_score = minimax(board, depth - 1, alpha, beta, False, model)
            board.pop()
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in legal_moves:
            board.push(move)
            eval_score = minimax(board, depth - 1, alpha, beta, True, model)
            board.pop()
            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return min_eval


def get_best_move(board: chess.Board, model, depth: int = 3):
    """Find best move"""
    best_move = None
    best_value = float('-inf')
    alpha = float('-inf')
    beta = float('inf')
    move_scores = {}
    
    legal_moves = list(board.legal_moves)
    legal_moves.sort(key=lambda m: (board.is_capture(m), m.promotion), reverse=True)
    
    for move in legal_moves:
        board.push(move)
        move_value = minimax(board, depth - 1, alpha, beta, False, model)
        board.pop()
        
        move_scores[move] = move_value
        if move_value > best_value:
            best_value = move_value
            best_move = move
            alpha = max(alpha, move_value)
    
    return best_move, move_scores


# ===== UTILITIES =====

def calculate_search_depth(board: chess.Board) -> int:
    """Adaptive depth"""
    move_count = len(board.move_stack)
    num_pieces = len(board.piece_map())
    
    if move_count < 12:
        return 2
    elif num_pieces <= 10:
        return 4
    else:
        return 3


def scores_to_probabilities(move_scores: dict) -> dict:
    """Convert scores to probabilities"""
    if not move_scores:
        return {}
    
    temperature = 0.5
    max_score = max(move_scores.values())
    exp_scores = {m: np.exp((s - max_score) / temperature) for m, s in move_scores.items()}
    total = sum(exp_scores.values())
    return {m: s / total for m, s in exp_scores.items()}


def load_chess_model():
    """Load model if exists"""
    model_path = os.path.join(os.path.dirname(__file__), "..", "chess_model.pth")
    
    if os.path.exists(model_path):
        try:
            model = ChessNet1500()
            model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
            model.eval()
            print(f"Model loaded from {model_path}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    else:
        print(f"No model found - using material evaluation only")
        return None


# Load model at import time
CHESS_MODEL = load_chess_model()