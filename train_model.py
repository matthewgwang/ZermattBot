"""Training script"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import chess
import chess.pgn
from src.utils.chess_bot  import ChessNet1500, board_to_tensor, move_to_index
import os

class ChessDataset(Dataset):
    def __init__(self, pgn_file, max_games=5000, min_elo=1800):
        print(f"Loading games from {pgn_file}...")
        self.data = []
        
        if not os.path.exists(pgn_file):
            raise FileNotFoundError(f"PGN file not found: {pgn_file}")
        
        with open(pgn_file) as f:
            game_count = 0
            while game_count < max_games:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                
                try:
                    white_elo = int(game.headers.get("WhiteElo", 0))
                    black_elo = int(game.headers.get("BlackElo", 0))
                    if white_elo < min_elo or black_elo < min_elo:
                        continue
                except:
                    continue
                
                result = game.headers.get("Result", "*")
                outcome = 1.0 if result == "1-0" else (-1.0 if result == "0-1" else 0.0)
                
                board = game.board()
                for i, move in enumerate(game.mainline_moves()):
                    self.data.append({
                        'board': board.copy(),
                        'move': move,
                        'outcome': outcome if i % 2 == 0 else -outcome
                    })
                    board.push(move)
                
                game_count += 1
                if game_count % 100 == 0:
                    print(f"  {game_count} games, {len(self.data)} positions")
        
        print(f"✓ Dataset ready: {len(self.data)} positions")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return (
            board_to_tensor(item['board']).squeeze(0),
            move_to_index(item['move']),
            torch.FloatTensor([item['outcome']])
        )

def train():
    print("\n" + "="*60)
    print("CHESS BOT TRAINING")
    print("="*60 + "\n")
    
    model = ChessNet1500()
    
    if os.path.exists('src/chess_model.pth'):
        response = input("Resume training? (y/n): ")
        if response.lower() == 'y':
            model.load_state_dict(torch.load('src/chess_model.pth', weights_only=True))
    
    dataset = ChessDataset('games.pgn', max_games=5000)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
    
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(10):
        model.train()
        total_p, total_v, num = 0, 0, 0
        
        for batch_idx, (boards, moves, outcomes) in enumerate(dataloader):
            optimizer.zero_grad()
            policy_pred, value_pred = model(boards)
            
            p_loss = policy_criterion(policy_pred, moves)
            v_loss = value_criterion(value_pred, outcomes)
            loss = p_loss + 0.5 * v_loss
            
            loss.backward()
            optimizer.step()
            
            total_p += p_loss.item()
            total_v += v_loss.item()
            num += 1
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}: P={p_loss.item():.4f}, V={v_loss.item():.4f}")
        
        print(f"\nEpoch {epoch+1}: P={total_p/num:.4f}, V={total_v/num:.4f}\n")
        torch.save(model.state_dict(), f'chess_model_epoch{epoch+1}.pth')
    
    torch.save(model.state_dict(), 'src/chess_model.pth')
    print("\n✓ Training complete! Model saved to src/chess_model.pth")

if __name__ == '__main__':
    train()