from __future__ import annotations

from .utils import chess_manager, GameContext
import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
import traceback
from pathlib import Path

# ==========================
# MODEL + DEVICE SETUP
# ==========================

HERE = Path(__file__).parent
MODEL_PATH = HERE / "chess_policy.pt"

# Detect best available device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"[main] Using device: {DEVICE}")

# Model definition must match the one used in train_chess_policy.py
class ChessPolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(12, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 128, 3, padding=1)
        self.fc    = nn.Linear(128 * 8 * 8, 4096)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """
    Encode a board into a (12, 8, 8) tensor:
    6 piece types * 2 colors, one plane per (type, color).
    Must match the encoding used during training.
    """
    planes = torch.zeros((12, 8, 8), dtype=torch.float32)

    piece_to_idx = {
        (chess.PAWN,   True): 0,
        (chess.KNIGHT, True): 1,
        (chess.BISHOP, True): 2,
        (chess.ROOK,   True): 3,
        (chess.QUEEN,  True): 4,
        (chess.KING,   True): 5,
        (chess.PAWN,   False): 6,
        (chess.KNIGHT, False): 7,
        (chess.BISHOP, False): 8,
        (chess.ROOK,   False): 9,
        (chess.QUEEN,  False):10,
        (chess.KING,   False):11,
    }

    for square, piece in board.piece_map().items():
        idx = piece_to_idx[(piece.piece_type, piece.color)]
        rank = 7 - chess.square_rank(square)   # rank 0 at top
        file = chess.square_file(square)
        planes[idx, rank, file] = 1.0

    return planes


def move_to_index(move: chess.Move) -> int:
    """
    Encode a move into an integer [0, 4095]:
    index = from_square * 64 + to_square
    """
    return move.from_square * 64 + move.to_square


def index_to_move(index: int) -> chess.Move:
    """
    Inverse of move_to_index, just for completeness.
    """
    from_sq = index // 64
    to_sq = index % 64
    return chess.Move(from_sq, to_sq)


# Load the trained model once at startup
_policy_model: ChessPolicyNet | None = None

def load_policy_model() -> ChessPolicyNet:
    global _policy_model
    if _policy_model is not None:
        return _policy_model

    # Start with a fresh model on the current device
    model = ChessPolicyNet().to(DEVICE)
    model.eval()

    if not MODEL_PATH.exists():
        print(f"[main] WARNING: model file {MODEL_PATH} not found; using untrained model")
        _policy_model = model
        return _policy_model

    try:
        print(f"[main] Loading model from: {MODEL_PATH}")
        # PyTorch 2.6+ defaults weights_only=True which breaks older checkpoints.
        try:
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        except TypeError:
            # Older torch that doesn't support weights_only argument
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

        model.load_state_dict(state_dict)
        model.eval()
        _policy_model = model
        print("[main] Loaded chess_policy.pt")
    except Exception as e:
        print("[main] ERROR while loading chess_policy.pt:", repr(e))
        print(traceback.format_exc())
        # Fall back to randomly initialized (untrained) model instead of crashing
        _policy_model = model

    return _policy_model


def choose_model_move(board: chess.Board) -> chess.Move:
    """
    Use the trained policy network to select a move.
    - Compute logits over all 4096 moves
    - Softmax to probabilities
    - Mask illegal moves
    - Sample according to masked probs (or argmax if you prefer)
    """
    model = load_policy_model()

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        raise ValueError("No legal moves available")

    with torch.no_grad():
        x = board_to_tensor(board).unsqueeze(0).to(DEVICE)  # (1, 12, 8, 8)
        logits = model(x)[0]                                # (4096,)
        probs = torch.softmax(logits, dim=0)                # (4096,)

    # Mask illegal moves
    mask = torch.zeros_like(probs)
    legal_indices = []
    for mv in legal_moves:
        idx = move_to_index(mv)
        legal_indices.append(idx)
    mask[legal_indices] = 1.0

    masked_probs = probs * mask
    total = masked_probs.sum().item()

    if total <= 0:
        # Fallback: uniform random over legal moves
        import random
        return random.choice(legal_moves)

    masked_probs = masked_probs / total

    # Move to CPU for sampling to avoid MPS multinomial issues
    masked_probs_cpu = masked_probs.detach().cpu()

    # Sample a move index according to probabilities
    idx_tensor = torch.multinomial(masked_probs_cpu, 1)
    chosen_index = int(idx_tensor.item())
    chosen_move = index_to_move(chosen_index)

    # Very unlikely, but double-check legality
    if chosen_move not in legal_moves:
        # fallback to highest-prob legal move using CPU probs
        best_move = max(
            legal_moves,
            key=lambda m: float(masked_probs_cpu[move_to_index(m)].item())
        )
        return best_move

    return chosen_move


# ==========================
# ENTRYPOINTS
# ==========================

@chess_manager.entrypoint
def test_func(ctx: GameContext):
    """
    This gets called every time the model needs to make a move.
    We:
      - Use the policy net to get probabilities over all moves
      - Mask to legal moves
      - Log those probabilities
      - Return the chosen move
    """
    try:
        print("Cooking move...")
        print(ctx.board.move_stack)

        # Pick move with the trained model
        move = choose_model_move(ctx.board)

        # Build probability dictionary for logging
        model = load_policy_model()
        with torch.no_grad():
            x = board_to_tensor(ctx.board).unsqueeze(0).to(DEVICE)
            logits = model(x)[0]
            probs = torch.softmax(logits, dim=0).detach().cpu()

        legal_moves = list(ctx.board.legal_moves)
        move_probs = {}
        total_prob = 0.0
        for mv in legal_moves:
            idx = move_to_index(mv)
            p = float(probs[idx].item())
            move_probs[mv] = p
            total_prob += p

        # Normalize just in case of numerical drift
        if total_prob > 0:
            move_probs = {mv: p / total_prob for mv, p in move_probs.items()}

        ctx.logProbabilities(move_probs)

        return move

    except Exception as e:
        # Log full traceback so we can see what went wrong in the devtools UI
        print("[test_func] ERROR while choosing/logging move:", repr(e))
        print(traceback.format_exc())
        # Re-raise so the server still returns a 500 and the error is visible upstream
        raise


@chess_manager.reset
def reset_func(ctx: GameContext):
    # This gets called when a new game begins
    # You could clear or re-init any game-specific caches here if needed.
    # For now, nothing is required since the model is stateless.
    pass
