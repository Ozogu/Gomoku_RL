from tkinter import *
from tkinter.messagebox import showerror
import numpy as np
import re

import os, sys
sys.path.insert(0, os.path.abspath(''))
from Ai import Ai

class Gomoku():
    def __init__(self, size, state="play", Ai=None, gui=True):
        self.__mainwindow = Tk()
        self.__mainwindow.title("Gomoku")

        # variables
        self.__gui = gui or state!="train"
        self.__board_size = size
        self.__turn_counter = 0
        self.__player = 1
        self.__player_names = ['Player', "Ai"] if state=="play" else ["Ai1", "Ai2"]
        self.__board_snapshot = np.zeros((size,size))
        self.__Ai = Ai

        # Creating labels
        self.__status = Label(self.__mainwindow,text=self.__player_names[self.__player-1])
        self.__turn_counter_label = Label(self.__mainwindow,text=
                            "{}. Turns taken".format(self.__turn_counter))

        # Placing label
        self.__status.grid(row=0,column=0,columnspan=4)
        self.__turn_counter_label.grid(row=0,column=4,columnspan=4)

        # Creating board
        self.__board = []
        for size_x in range(self.__board_size):
            board_x = []
            for size_y in range(self.__board_size):
                # Creating button
                button = Button(self.__mainwindow,width=2,height=1,
                                      command =(lambda y=size_y,x=size_x:
                                                self.take_turn(x,y)))
                # Place button to board
                button.grid(row=size_y+1,column=size_x,sticky =N+W+E+S)
                # Place button to to list
                board_x.append(button)
            self.__board.append(board_x)

        if (gui):
            if (re.match(self.__player_names[self.__player-1].lower(), 'ai')):
                self.ai_turn()
            else:
                self.__mainwindow.mainloop()

    def take_turn(self,x,y):
        """
        Places x or o, locks the button and passes the information
        :param x: X coordinate
        :param y: y coordinate
        :return: None
        """
        # Take player 1 turn
        if self.__player == 1:
            # Return if winner found
            if self.place_token("lightblue", "x", x, y):
                return
        # Take player 2 turn
        elif self.__player == 2:
            # Return if winner found
            if self.place_token("lightcoral", "o", x, y):
                return
        # Update board
        self.__board[x][y].update()
        self.__board_snapshot[x][y] = self.__player
        self.end_turn()

    def ai_turn(self):
        data = { "snapshot": self.__board_snapshot, "player": self.__player }
        coordinates = self.__Ai.predict()
        self.take_turn(coordinates["x"], coordinates["y"])

    def place_token(self, token, symbol, x, y):
        self.__board[x][y].config(state=DISABLED, background=token, text=symbol)
        self.__board_snapshot[x][y] = self.__player-1
        # Return if win condition met
        return self.check_win_condition(token, [x, y])

    def end_turn(self):
        """
        Update playing player and turn label
        """
        self.__turn_counter += 1
        # päivittää pelaajan
        self.__player %=2
        self.__player += 1
        self.__status["text"] = self.__player_names[self.__player-1]
        self.__turn_counter_label["text"]= "{}. Turns taken".format\
            (self.__turn_counter)
        self.__turn_counter_label.update()

        if (re.match(self.__player_names[self.__player-1].lower(), 'ai')):
            self.ai_turn()

    def check_win_condition(self,token,coordinate):
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
                if True == self.check_direction(token,vec1,coordinate):
                    self.declare_winner()
                    return True
                if True == self.check_direction(token,vec2,coordinate):
                    self.declare_winner()
                    return True

                # Turn vector directions
                vec1 = -vec1[1], vec1[0]
                vec2 = -vec2[1], vec2[0]

        # Check for draw
        if self.__turn_counter+1 == self.__board_size**2:
            self.__turn_counter_label["text"] = "Draw!"
            self.__status["text"] = ""
            self.__status.update()
            self.__turn_counter_label.update()
            showerror("Tasapeli","tasapeli!")
            return True

    def check_direction(self, token, vector, coordinate):
        """
        Takes a vector, creates inverse vector and gives both to direction function.
        :param token: Player token
        :param vector: direction vector which will be passed forward
        :param coordinate: Coordinate of the button
        :return: True if 5-straight found.
        """
        inverse_vector = -vector[0], -vector[1]
        # Calculate hits to direction
        hits = self.direction(token,vector,1,coordinate)
        if hits == 5:
            return True
        # After reaching the end, add hits towards the opposite direction
        hits = self.direction(token,inverse_vector,hits,coordinate)
        if hits == 5:
            return True

    def direction(self, token, vector, hits, coordinate):
        """
        Calculate hits towards this direction
        :param token: Player token
        :param vector: direction vector
        :param hits:  Hits so far
        :param coordinate: Coordinate of the button
        :return:
        """
        try:
            # Button at the end to the vector
            next_x = coordinate[0]+vector[0]
            next_y = coordinate[1]+vector[1]
            next_coordinate = [next_x,next_y]
            # tarkistetaan token ja tallennetaan se uutena merkkinä
            if self.__board[next_x][next_y]["background"] == "lightcoral":
                next_token = "lightcoral"
            elif self.__board[next_x][next_y]["background"] == "lightblue":
                next_token = "lightblue"
            else:
                next_token = None

            # Add hit and continue if next token is of the players
            if next_token == token:
                hits = self.direction(next_token, vector, hits+1 ,next_coordinate)
            # Else return hits
            return hits
        # Out of bounds
        except IndexError:
            return hits

    def declare_winner(self):
        """
        Pop up winner
        """
        # Update label
        self.__turn_counter_label["text"] = self.__player_names[self.__player-1] + " won!"
        self.__status["text"] = ""
        self.__status.update()

        # Disable buttons
        for x_buttons in range(len(self.__board)):
            for y_buttons in range(len(self.__board)):
                self.__board[x_buttons][y_buttons].config(state=DISABLED)
        self.__turn_counter_label.update()

        # Declare winner
        showerror("Winner", self.__player_names[self.__player-1] + " won!")

def main():
    # ai = [Ai(), Ai()]
    ai = Ai()
    Gomoku(size=25, state="play", Ai=ai)

main()
