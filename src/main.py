from .utils import chess_manager, GameContext
import os
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
import chess.pgn
from chess import Move
import random

# ---------- 1. MODEL DEFINITION ----------
# This MUST be identical to the model class in your train.py
# so that the model.pth weights can be loaded correctly.

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class ChessModel(nn.Module):
    def __init__(self, in_planes=13, blocks=5, channels=128):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(channels)
        self.layers = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.conv_policy = nn.Conv2d(channels, 64, kernel_size=1) 
        self.bn_policy = nn.BatchNorm2d(64)
        self.conv_value = nn.Conv2d(channels, 1, kernel_size=1)
        self.bn_value = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(8*8*1, 256)
        self.fc2 = nn.Linear(256, 1)
    
    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.layers(x)
        p = F.relu(self.bn_policy(self.conv_policy(x)))
        p = p.view(-1, 4096) 
        v = F.relu(self.bn_value(self.conv_value(x)))
        v = v.view(-1, 8*8)
        v = F.relu(self.fc1(v))
        v = torch.tanh(self.fc2(v))
        return p, v

# ---------- 2. HELPER FUNCTIONS ----------

def board_to_tensor(board: chess.Board) -> torch.Tensor:
    tensor = torch.zeros(13, 8, 8)
    piece_map = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            row, col = divmod(sq, 8)
            plane = piece_map[piece.symbol()]
            tensor[plane, 7 - row, col] = 1
    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1
    return tensor

def move_to_index(move: chess.Move) -> int:
    """Map a chess.Move object to an integer index (0-4095)."""
    return move.from_square * 64 + move.to_square



print("Loading model...")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MODEL_PATH = os.path.join(os.path.dirname(__file__), "weights", "model.pth")

MODEL = ChessModel().to(DEVICE)
MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
MODEL.eval() # Set model to evaluation mode
print(f"Model loaded successfully from {MODEL_PATH}")



def get_move(pgn: str) -> str:

    
    board = chess.Board() # Start with a new board
    
    if pgn: # Check if the string is not empty
        moves = pgn.split()
        for move_san in moves:
            if '.' in move_san:
                # This is a move number (like "1." or "2..."), skip it
                continue
            
            try:
                # This is the standard way to play a move in SAN
                board.push_san(move_san)
            except chess.IllegalMoveError:
                # This is a fallback in case the move is UCI (e.g., "e2e4")
                try:
                    board.push_uci(move_san)
                except Exception:
                    print(f"Error: Could not parse move {move_san}")
                    pass # Ignore unparseable moves
            except Exception as e:
                print(f"Error parsing move {move_san}: {e}")
                break
    
    #  Get all legal moves
    legal_moves = list(board.generate_legal_moves())
    
    if not legal_moves:
        # (checkmate or stalemate)
        print("No legal moves available. Returning null move.")
        return "0000" # Return a null move

    #  Convert board to tensor
    input_tensor = board_to_tensor(board).unsqueeze(0).to(DEVICE)

    #  Get model prediction
    with torch.no_grad():
        policy_logits, value_pred = MODEL(input_tensor)

    # Convert logits to probabilities
    all_move_probs = F.softmax(policy_logits, dim=1)[0]
    
    # Filter: Get the model's probability *only for legal moves*
    legal_move_probs = {}
    total_legal_prob = 0.0
    
    for move in legal_moves:
        idx = move_to_index(move)
        
        
        if 0 <= idx < 4096:
            prob = all_move_probs[idx].item()
            legal_move_probs[move] = prob
            total_legal_prob += prob

    #  Renormalize probabilities 
    if total_legal_prob == 0.0:
        print("Warning: Model assigned 0 prob to all legal moves. Picking random.")
        best_move = random.choice(legal_moves)
    else:
        # Renormalize
        final_probs = {
            move: prob / total_legal_prob
            for move, prob in legal_move_probs.items()
        }
        
        #  Choose the best move
        best_move = max(final_probs, key=final_probs.get)

    #  Return the move in UCI format
    print(f"Model chose: {best_move.uci()}")
    return best_move.uci()

