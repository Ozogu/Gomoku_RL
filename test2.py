import numpy as np
np.set_printoptions(precision=3, suppress=True, threshold=np.nan, linewidth=np.nan)

def update_attack_board(attack_boards, token, coordinates):
    offsets = ( (1,0), (1,1), (0,1), (1,-1) )
    x = coordinates['x']
    y = coordinates['y']
    for i in range(4):
        dx = offsets[i][0]
        dy = offsets[i][1]
        direction_values = [(None,None,1),(None,None,1)]

        for direction in range(2):
            sign = 1 if direction == 0 else -1
            for n in range(1, size+1):
                # Offsetted x,y
                ox = x+sign*n*dx
                oy = y+sign*n*dy
                if oy < 0 or oy >= size or ox < 0 or ox >= size: break
                if attack_boards[token-1][i][oy][ox] != 0:
                    direction_values[direction] = (ox,oy,n)
                    break

        # Direction index
        for di in range(2):
            ox = direction_values[di][0]
            oy = direction_values[di][1]
            # Other directions n
            on = direction_values[di-1][2]
            if ox is not None and oy is not None:
                attack_boards[token-1][i][oy][ox] += on

        attack_boards[0][i][y][x] = 0
        attack_boards[1][i][y][x] = 0

def place_token(board, attack_boards, token, coordinates):
    board[coordinates['y']][coordinates['x']] = token
    print(board)
    print()
    update_attack_board(attack_boards, token, coordinates)
    print(attack_boards[0][0])
    print('--------')
    print(attack_boards[1][0])
    print('-'*100)

size = 10
board = np.zeros((size,size), np.uint8)
attack_boards = np.ones((2, 4, size, size), dtype=np.int8)

place_token(board, attack_boards, 2, { 'x': 1, 'y': 1 })
place_token(board, attack_boards, 1, { 'x': 0, 'y': 1 })

# place_token(board, attack_boards, 1, { 'x': 6, 'y': 3 })

# place_token(board, attack_boards, 1, { 'x': 5, 'y': 5 })
# place_token(board, attack_boards, 1, { 'x': 3, 'y': 5 })
# place_token(board, attack_boards, 1, { 'x': 4, 'y': 5 })
# place_token(board, attack_boards, 1, { 'x': 1, 'y': 5 })
# place_token(board, attack_boards, 1, { 'x': 2, 'y': 5 })

# place_token(board, attack_boards, 1, { 'x': 5, 'y': 4 })
# place_token(board, attack_boards, 1, { 'x': 5, 'y': 3 })
# place_token(board, attack_boards, 1, { 'x': 5, 'y': 2 })
# place_token(board, attack_boards, 1, { 'x': 5, 'y': 1 })

# place_token(board, attack_boards, 1, { 'x': 6, 'y': 6 })
# place_token(board, attack_boards, 1, { 'x': 7, 'y': 7 })
# place_token(board, attack_boards, 1, { 'x': 8, 'y': 8 })
# place_token(board, attack_boards, 1, { 'x': 9, 'y': 9 })

# place_token(board, attack_boards, 1, { 'x': 6, 'y': 4 })
# place_token(board, attack_boards, 1, { 'x': 7, 'y': 3 })
# place_token(board, attack_boards, 1, { 'x': 8, 'y': 2 })
# place_token(board, attack_boards, 1, { 'x': 9, 'y': 1 })