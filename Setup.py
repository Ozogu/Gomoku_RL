from tkinter import Tk, Label, Entry, Button, Checkbutton, IntVar, Radiobutton, Frame, LEFT, RIGHT, X

from tkinter.messagebox import showerror

from shared import TRAIN, PLAY, GREEDY, BOLTZMANN
from Gomoku import Gomoku
from Ai import Ai

class Setup():
    def __init__(self):
        self.success = False
        # Init window
        self.__window = Tk()

        # Vars
        self.__state_radio_var = IntVar()
        self.__approach_radio_var = IntVar()
        self.__gui_checkbox_var = IntVar()
        self.__verbose_checkbox_var = IntVar()

        # Init elements
        self.__window.title("Game settings")

        self.__approach_frame = Frame(self.__window)
        self.__greedy_radio = Radiobutton(self.__approach_frame, text="Greedy", variable=self.__approach_radio_var, value=GREEDY)
        self.__boltzman_radio = Radiobutton(self.__approach_frame, text="Boltzman, e-greedy", variable=self.__approach_radio_var, value=BOLTZMANN)

        self.__epsilon_frame = Frame(self.__window)
        self.__epsilon_label = Label(self.__epsilon_frame, text="Epsilon")
        self.__epsilon = Entry(self.__epsilon_frame)

        self.__keep_prob_frame = Frame(self.__window)
        self.__keep_prob_label = Label(self.__keep_prob_frame, text="Keep probability")
        self.__keep_prob = Entry(self.__keep_prob_frame)

        self.__gui_frame = Frame(self.__window)
        self.__label_gui = Label(self.__gui_frame, text="Use Gui")
        self.__gui_checkbox = Checkbutton(self.__gui_frame, variable=self.__gui_checkbox_var)

        self.__verbose_frame = Frame(self.__window)
        self.__verbose_label = Label(self.__verbose_frame, text="Verbose")
        self.__verbose_checkbox = Checkbutton(self.__verbose_frame, variable=self.__verbose_checkbox_var)

        self.__state_frame = Frame(self.__window)
        self.__train__radio = Radiobutton(self.__state_frame, text="train", variable=self.__state_radio_var, value=TRAIN)
        self.__play__radio = Radiobutton(self.__state_frame, text="play", variable=self.__state_radio_var, value=PLAY)

        self.__ok_button = Button(self.__window, text="Ok", command=self.confirm)

        # Defaults
        self.__gui_checkbox.select()
        self.__play__radio.select()
        self.__greedy_radio.select()
        self.__epsilon.insert(0,'0')
        self.__keep_prob.insert(0,'1')

        # Places
        self.__approach_frame.pack(fill=X)
        self.__greedy_radio.pack(side=LEFT)
        self.__boltzman_radio.pack(side=RIGHT)

        self.__epsilon_frame.pack(fill=X)
        self.__epsilon_label.pack(side=LEFT)
        self.__epsilon.pack(side=RIGHT)

        self.__keep_prob_frame.pack(fill=X)
        self.__keep_prob_label.pack(side=LEFT)
        self.__keep_prob.pack(side=RIGHT)

        self.__gui_frame.pack(fill=X)
        self.__label_gui.pack(side=LEFT)
        self.__gui_checkbox.pack(side=RIGHT)

        self.__state_frame.pack(fill=X)
        self.__train__radio.pack(side=LEFT)
        self.__play__radio.pack(side=RIGHT)
        self.__ok_button.pack()

        # Hotkeys
        self.__window.bind("<Return>", self.confirm)

        self.__window.mainloop()

    def confirm(self, event=""):
        self.gui = self.__gui_checkbox_var.get()
        self.keep_prob = float(self.__keep_prob.get())
        self.approach = self.__approach_radio_var.get()
        self.state = self.__state_radio_var.get()
        self.epsilon = float(self.__epsilon.get())
        self.verbose = self.__verbose_checkbox_var.get()
        self.success = True

        self.__window.destroy()

if __name__ == "__main__":
    setup = Setup()
    Ai1 = Ai(keep_prob=setup.epsilon, greedy=setup.approach, verbose=setup.verbose)
    Ai2 = Ai(keep_prob=setup.epsilon, greedy=setup.approach, verbose=setup.verbose)
    Gomoku(size=25, state=setup.state, AiDefault=Ai1, AiTrainPartner=Ai2, gui=setup.gui)
