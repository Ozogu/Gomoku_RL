from tkinter import Tk, Label, Entry, Button, Checkbutton, IntVar, Radiobutton
from tkinter.messagebox import showerror

from shared import TRAIN, PLAY
from Gomoku import Gomoku
from Ai import Ai

class Setup():
    def __init__(self):
        self.success = False
        # Init window
        self.__window = Tk()

        self.__radio_var = IntVar()
        self.__checkbox_var = IntVar()
        # Init buttons
        self.__window.title("Game settings")
        self.__label_model = Label(self.__window, text="Model name")
        self.__model_name = Entry(self.__window)
        self.__label_render = Label(self.__window, text="Render game every")
        self.__render_number = Entry(self.__window)
        self.__label_gui = Label(self.__window, text="Use Gui")
        self.__train__radio = Radiobutton(self.__window, text="train", variable=self.__radio_var, value=TRAIN)
        self.__play__radio = Radiobutton(self.__window, text="play", variable=self.__radio_var, value=PLAY)
        self.__state = Entry(self.__window)
        self.__label_gui = Label(self.__window, text="Use Gui")
        self.__gui_checkbox = Checkbutton(self.__window, variable=self.__checkbox_var)
        self.__ok_button = Button(self.__window, text="Ok", command=self.confirm)

        # Defaults
        self.__model_name.insert(0, "model")
        self.__state.insert(0, "play")
        self.__gui_checkbox.select()
        self.__play__radio.select()

        # Places
        self.__label_model.grid(row=0, column=0)
        self.__model_name.grid(row=0, column=1)
        self.__label_gui.grid(row=2, column=0)
        self.__gui_checkbox.grid(row=2, column=1)
        self.__label_render.grid(row=3, column=0)
        self.__render_number.grid(row=3, column=1)
        self.__train__radio.grid(row=4, column=0)
        self.__play__radio.grid(row=4, column=1)
        self.__ok_button.grid(row=5, column=0)

        # Hotkeys
        self.__window.bind("<Return>", self.confirm)

        self.__window.mainloop()

    def confirm(self, event=""):
        """
        Take and return board size from entry
        :param event: For hotkey
        """
        self.gui = self.__checkbox_var.get()
        self.model = self.__model_name.get()
        self.state = self.__radio_var.get()
        self.render = self.__render_number.get()
        self.success = True

        self.__window.destroy()

if __name__ == "__main__":
    setup = Setup()
    Gomoku(size=25, state=setup.state, AiDefault=Ai(), AiTrainPartner=Ai(), gui=setup.gui)