from tkinter import Tk, Button, Label, N, W, E, S, DISABLED, NORMAL
from tkinter.messagebox import showerror
import numpy as np
import re

from Nn import Nn
from Ai import Ai
from shared import WIN, LOSE, DRAW, PLAY, TRAIN

class Gomoku():
    def __init__(self, size, state=PLAY, AiDefault=None, AiTrainPartner=None, gui=True):
        # variables
        self._board_size = size
        self._gui = gui
        self._turn_counter = 0
        self._player = 1
        self._state = state
        self._winner = 0
        self._game_number = 1
        self._player_names = ['Player', str(AiDefault)] if state==PLAY else [str(AiDefault), str(AiTrainPartner)]
        self._AiDefault = AiDefault
        self._AiTrainPartner = AiTrainPartner
        self._board_snapshot = np.zeros((size,size), dtype=np.int8)
        # Longest streak for each direction for each token type
        self._attack_boards = np.ones((2,4,size,size), dtype=np.int8)

        if (self._gui): self.__render_mainwindow()

        if (self.__check_ai_turn()):
            self.__play_against_self()
        else:
            self.__mainwindow.mainloop()

    def __reset_game(self):
        print(f"Game {self._game_number}, winner {self._player_names[self._player-2]}")
        rewards = {"default": 0, "partner": 0}
        if (self._winner == 1):
            rewards["default"] = LOSE
            rewards["partner"] = WIN
        elif (self._winner == 2):
            rewards["default"] = WIN
            rewards["partner"] = LOSE
        elif (self._winner == 3):
            rewards["default"] = DRAW
            rewards["partner"] = DRAW

        self._AiDefault.reward(rewards["default"])
        self._AiDefault.new_game()

        if self._state == TRAIN:
            self._AiTrainPartner.reward(rewards["partner"])
            self._AiTrainPartner.new_game()

        if (self._game_number % self._AiDefault.batch_size == 0):
            print("Learning....")
            self._AiDefault.train()
            if self._state == TRAIN:
                self._AiTrainPartner.train()

        self._winner = 0
        self._turn_counter = 0
        self._player = 1
        self._board_snapshot = np.zeros((self._board_size, self._board_size), dtype=np.int8)
        self._attack_boards = np.ones((2, 4, self._board_size, self._board_size), dtype=np.int8)
        self._game_number += 1

        if (self._gui):
            self.__reset_mainwindow()

    def __play_against_self(self):
        while True:
            self.__ai_turn()
            if (self._winner):
                self.__reset_game()

    def __check_ai_turn(self):
        return not re.match(r"player", self._player_names[self._player-1].lower())

    def __reset_mainwindow(self):
        size = self._board_size
        for i in range(0,size**2):
            x = i % size
            y =int(i / size)
            self.__board[x][y].config(text="",background="SystemButtonFace", state=NORMAL)

    def __render_mainwindow(self):
        self.__mainwindow = Tk()
        self.__mainwindow.title("Gomoku")

        # Creating labels
        self.__status = Label(self.__mainwindow,text=self._player_names[self._player-1])
        self.__turn_counter_label = Label(self.__mainwindow,text="{}. Turns taken".format(self._turn_counter))

        # Create board
        self.__board = []
        for size_x in range(self._board_size):
            board_x = []
            for size_y in range(self._board_size):
                # Creating button
                button = Button(
                    self.__mainwindow,
                    width=2,
                    height=1,
                    command=(lambda y=size_y,x=size_x: self.__take_turn(x,y))
                    )
                # Place button to board
                button.grid(row=size_y+1,column=size_x,sticky=N+W+E+S)
                # Place button to to list
                board_x.append(button)
            self.__board.append(board_x)

        # Placing label
        self.__status.grid(row=0,column=0,columnspan=4)
        self.__turn_counter_label.grid(row=0,column=4,columnspan=4)


    def __take_turn(self,x,y):
        """
        Places x or o, locks the button and passes the information
        :param x: X coordinate
        :param y: y coordinate
        :return: None
        """
        # Take player 1 turn
        if self._player == 1:
            # Return if winner found
            if self.__place_token("lightblue", "x", x, y):
                return
        # Take player 2 turn
        elif self._player == 2:
            # Return if winner found
            if self.__place_token("lightcoral", "o", x, y):
                return
        # Update board
        if (self._gui): self.__board[x][y].update()
        self.__end_turn()

    def __ai_turn(self):
        ai = None
        if self._player == 2:
            ai = self._AiDefault
        else:
            ai = self._AiTrainPartner
        data =  \
        {
            "board": self._board_snapshot,
            "player": self._player,
            "attack_boards": self._attack_boards[self._player-1],
            "defense_boards": self._attack_boards[self._player-2]
        }
        coordinates = ai.predict(data)
        assert(coordinates.keys() >= {'x', 'y'})
        self.__take_turn(coordinates["x"], coordinates["y"])

    def __place_token(self, token, symbol, x, y):
        if (self._gui):
            self.__board[x][y].config(state=DISABLED, background=token, text=symbol)
        # Snapshot is a matrix, hence coordinates are the wrong way.
        assert self._board_snapshot[y][x] == 0
        self._board_snapshot[y][x] = self._player
        self.__update_attack_boards(x,y)
        # Return if win condition met
        return self.__check_win_condition(token, [x, y])

    def __update_attack_boards(self, x, y):
        offsets = ((1, 0), (1, 1), (0, 1), (1, -1))
        for i in range(4):
            dx = offsets[i][0]
            dy = offsets[i][1]
            direction_values = [(None, None, 1), (None, None, 1)]

            for direction in range(2):
                sign = 1 if direction == 0 else -1
                for n in range(1, self._board_size+1):
                    # Offsetted x,y
                    ox = x+sign*n*dx
                    oy = y+sign*n*dy
                    if oy < 0 or oy >= self._board_size or ox < 0 or ox >= self._board_size:
                        break
                    if self._attack_boards[self._player-1][i][oy][ox] > 0:
                        direction_values[direction] = (ox, oy, n)
                        break
                    elif self._attack_boards[self._player-1][i][oy][ox] != -self._player:
                        direction_values[direction] = (None, None, n)
                        break

            # Direction index
            for di in range(2):
                ox = direction_values[di][0]
                oy = direction_values[di][1]
                # Other directions n
                on = direction_values[di-1][2]
                if ox is not None and oy is not None:
                    self._attack_boards[self._player-1][i][oy][ox] += on

            self._attack_boards[0][i][y][x] = -self._player
            self._attack_boards[1][i][y][x] = -self._player

    def __end_turn(self):
        """
        Update playing player and turn label
        """
        self._turn_counter += 1
        # päivittää pelaajan
        self._player %=2
        self._player += 1
        if (self._gui):
            self.__status["text"] = self._player_names[self._player-1]
            self.__turn_counter_label["text"]= "{}. Turns taken".format\
                (self._turn_counter)
            self.__turn_counter_label.update()

        # This will cause stackoverflow in training.
        if (self._state == PLAY and self.__check_ai_turn()):
             self.__ai_turn()


    def __check_win_condition(self,token,coordinate):
        """
        Check if win condition met.
        Check horizontal, vertical, and diagonal vectors for win conditions
        :param token: Player token
        :param coordinate: Coordinate of the button
        :return: True if win condition met
        """
        # Check if it's even possible to win yet
        if self._turn_counter >= 8 and self._turn_counter+1 < self._board_size**2:
            # Disable win
            # return False

            # Up and up right vectors
            vec1=(1,0)
            vec2=(1,1)

            # Loop both directions of vector
            for _ in range(2):
                if self.__check_direction(vec1,coordinate):
                    self.__declare_winner()
                    return True
                if self.__check_direction(vec2,coordinate):
                    self.__declare_winner()
                    return True

                # Turn vector directions
                vec1 = -vec1[1], vec1[0]
                vec2 = -vec2[1], vec2[0]

        # Check for draw
        elif self._turn_counter+1 >= self._board_size**2:
            if (self._gui):
                self.__turn_counter_label["text"] = "Draw!"
                self.__status["text"] = ""
                self.__status.update()
                self.__turn_counter_label.update()
            self._winner = 3
            if (self._state == PLAY): showerror("Draw","Draw!")
            return True

    def __check_direction(self, vector, coordinate):
        """
        Takes a vector, creates inverse vector and gives both to direction function.
        :param token: Player token
        :param vector: direction vector which will be passed forward
        :param coordinate: Coordinate of the button
        :return: True if 5-straight found.
        """
        inverse_vector = -vector[0], -vector[1]
        # Calculate hits to direction
        hits = self.__direction(vector,1,coordinate)
        if hits == 5:
            return True
        # After reaching the end, add hits towards the opposite direction
        hits = self.__direction(inverse_vector,hits,coordinate)
        if hits == 5:
            return True

    def __direction(self, vector, hits, coordinate):
        """
        Calculate hits towards this direction
        :param vector: direction vector
        :param hits:  Hits so far
        :param coordinate: Coordinate of the button
        :return:
        """
        try:
            assert hits is not None
            # Button at the end to the vector
            next_x = coordinate[0]+vector[0]
            next_y = coordinate[1]+vector[1]
            next_coordinate = [next_x,next_y]
            # Check token and save it as new
            if self._board_snapshot[next_y][next_x] == self._player:
                # Add hit and continue if next token is of the players
                return self.__direction(vector, hits+1 ,next_coordinate)
            else:
                return hits
        # Out of bounds
        except IndexError:
            return hits

    def __declare_winner(self):
        """
        Pop up winner
        """
        self._winner = self._player

        # Update label
        if (self._gui):
            self.__turn_counter_label["text"] = self._player_names[self._player-1] + " won!"
            self.__status["text"] = ""
            self.__status.update()

            # Disable buttons
            for x_buttons in range(len(self.__board)):
                for y_buttons in range(len(self.__board)):
                    self.__board[x_buttons][y_buttons].config(state=DISABLED)
            self.__turn_counter_label.update()

        # Declare winner
        if (self._state == PLAY):
            showerror("Winner", self._player_names[self._player-1] + " won!")
            self.__reset_game()

if __name__ == "__main__":
    Ai1=Nn(keep_prob=.9, greedy=False, epsilon=.2, batch_size=64, epochs=10)
    Ai2=Nn(keep_prob=.9, greedy=False, epsilon=.2, batch_size=64, epochs=10)
    Gomoku(size=25, state=TRAIN, AiDefault=Ai1, AiTrainPartner=Ai2, gui=False)
