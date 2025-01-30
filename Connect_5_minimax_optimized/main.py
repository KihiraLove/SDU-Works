from board import Board
from bot import Bot

board = Board()
bot = Bot(board)
win = False
is_player_x = True
enlarged = False
last_move = ()
while not win:
    board.print_board()
    try:
        if is_player_x:
            move = input("Enter your move (x y): ").split(' ')
            x = int(move[0])
            y = int(move[1])
            last_move = (x, y)
            win, enlarged = board.move(x, y, is_player_x)
        else:
            move = bot.smart_move(last_move, enlarged)
            win, enlarged = board.move(move[0], move[1], is_player_x)
        is_player_x = not is_player_x
    except:
        print("Invalid move")

board.print_board()
