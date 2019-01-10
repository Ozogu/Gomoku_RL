from tkinter import Tk, Button, Label, N, W, E, S, DISABLED, NORMAL
from tkinter.messagebox import showerror
import numpy as np
import re

from Ai import Ai
from shared import WIN, LOSE, DRAW, PLAY, TRAIN

class Gomoku():
    def __init__(self, size, state=PLAY, AiDefault=None, AiTrainPartner=None, gui=True):
        # variables
        self.__board_size = size
        self.__gui = gui
        self.__turn_counter = 0
        self.__player = 1
        self.__state = state
        self.__winner = 0
        self.__game_number = 1
        self.__player_names = ['Player', "Ai"] if state==PLAY else ["Ai1", "Ai2"]
        self.__board_snapshot = np.zeros((size,size), dtype=np.int8)
        self.__AiDefault = AiDefault
        self.__AiTrainPartner = AiTrainPartner

        if (self.__gui): self.__render_mainwindow()

        if (self.__check_ai_turn()):
            self.__play_against_self()
        else:
            self.__mainwindow.mainloop()

    def __reset_game(self):
        print(f"Game {self.__game_number}, winner {self.__winner}")
        rewards = {"default": 0, "partner": 0}
        if (self.__winner == 1):
            rewards["default"] = LOSE
            rewards["partner"] = WIN
        elif (self.__winner == 2):
            rewards["default"] = WIN
            rewards["partner"] = LOSE
        elif (self.__winner == 3):
            rewards["default"] = DRAW
            rewards["partner"] = DRAW

        self.__AiDefault.reward(rewards["default"])
        self.__AiDefault.new_game()

        if self.__state == TRAIN:
            self.__AiTrainPartner.reward(rewards["partner"])
            self.__AiTrainPartner.new_game()

        if (self.__game_number % self.__AiDefault.batch_size == 0):
            print("Learning....")
            self.__AiDefault.train()
            if self.__state == TRAIN:
                self.__AiTrainPartner.train()

        self.__winner = 0
        self.__turn_counter = 0
        self.__player = 1
        self.__board_snapshot = np.zeros(
            (self.__board_size, self.__board_size), dtype=np.int8)
        self.__game_number += 1

        if (self.__gui):
            self.__reset_mainwindow()

    def __play_against_self(self):
        while True:
            self.__ai_turn()
            if (self.__winner):
                self.__reset_game()

    def __check_ai_turn(self):
        return re.match(r"ai(\d)?", self.__player_names[self.__player-1].lower())

    def __reset_mainwindow(self):
        size = self.__board_size
        for i in range(0,size**2):
            x = i % size
            y =int(i / size)
            self.__board[x][y].config(text="",background="SystemButtonFace", state=NORMAL)

    def __render_mainwindow(self):
        self.__mainwindow = Tk()
        self.__mainwindow.title("Gomoku")

        # Creating labels
        self.__status = Label(self.__mainwindow,text=self.__player_names[self.__player-1])
        self.__turn_counter_label = Label(self.__mainwindow,text="{}. Turns taken".format(self.__turn_counter))

        # Create board
        self.__board = []
        for size_x in range(self.__board_size):
            board_x = []
            for size_y in range(self.__board_size):
                # Creating button
                button = Button(
                    self.__mainwindow,
                    width=2,
                    height=1,
                    command=(lambda y=size_y,x=size_x: self.__take_turn(x,y))
                    )
                # Place button to board
                button.grid(row=size_y+1,column=size_x,sticky =N+W+E+S)
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
        if self.__player == 1:
            # Return if winner found
            if self.__place_token("lightblue", "x", x, y):
                return
        # Take player 2 turn
        elif self.__player == 2:
            # Return if winner found
            if self.__place_token("lightcoral", "o", x, y):
                return
        # Update board
        if (self.__gui): self.__board[x][y].update()
        self.__end_turn()

    def __ai_turn(self):
        ai = None
        if self.__player == 2:
            ai = self.__AiDefault
        else:
            ai = self.__AiTrainPartner
        data = { "board": self.__board_snapshot, "player": self.__player }
        coordinates = ai.predict(data)
        self.__take_turn(coordinates["x"], coordinates["y"])

    def __place_token(self, token, symbol, x, y):
        if (self.__gui):
            self.__board[x][y].config(state=DISABLED, background=token, text=symbol)
        # Snapshot is a matrix, hence coordinates are the wrong way.
        assert self.__board_snapshot[y][x] == 0
        self.__board_snapshot[y][x] = self.__player
        # Return if win condition met
        return self.__check_win_condition(token, [x, y])

    def __end_turn(self):
        """
        Update playing player and turn label
        """
        self.__turn_counter += 1
        # päivittää pelaajan
        self.__player %=2
        self.__player += 1
        if (self.__gui):
            self.__status["text"] = self.__player_names[self.__player-1]
            self.__turn_counter_label["text"]= "{}. Turns taken".format\
                (self.__turn_counter)
            self.__turn_counter_label.update()

        # This will cause stackoverflow in training.
        if (self.__state == PLAY and self.__check_ai_turn()):
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
        if self.__turn_counter >= 8:
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
        if self.__turn_counter+1 == self.__board_size**2:
            if (self.__gui):
                self.__turn_counter_label["text"] = "Draw!"
                self.__status["text"] = ""
                self.__status.update()
                self.__turn_counter_label.update()
            self.__winner = 3
            if (self.__state == PLAY): showerror("Draw","Draw!")
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
            if self.__board_snapshot[next_y][next_x] == self.__player:
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
        self.__winner = self.__player

        # Update label
        if (self.__gui):
            self.__turn_counter_label["text"] = self.__player_names[self.__player-1] + " won!"
            self.__status["text"] = ""
            self.__status.update()

            # Disable buttons
            for x_buttons in range(len(self.__board)):
                for y_buttons in range(len(self.__board)):
                    self.__board[x_buttons][y_buttons].config(state=DISABLED)
            self.__turn_counter_label.update()

        # Declare winner
        if (self.__state == PLAY):
            showerror("Winner", self.__player_names[self.__player-1] + " won!")
            self.__reset_game()

if __name__ == "__main__":
    Ai1=Ai(keep_prob=.6, greedy=False, epsilon=.2)
    Ai2=Ai(keep_prob=.6, greedy=False, epsilon=.2)
    Gomoku(size=25, state=TRAIN, AiDefault=Ai1, AiTrainPartner=Ai2, gui=False)
