from tkinter import *
from tkinter.messagebox import showerror

class BoardSize():
    def __init__(self):
        self.success = False
        # Init window
        self.__window = Tk()

        # Init buttons
        self.__window.title("Board settings")
        self.__label = Label(self.__window,
                             text="Size of board?")
        self.__size = Entry(self.__window)
        self.__ok_button = Button(self.__window,
                                  text="Ok", command=self.take_size)

        # Default size
        self.__size.insert(0, "20")

        # Places
        self.__label.grid(row=0, column=0)
        self.__size.grid(row=1, column=0)
        self.__ok_button.grid(row=2, column=0)

        # Hotkeys
        self.__size.bind("<Return>", self.take_size)

        self.__window.mainloop()

    def take_size(self, event=""):
        """
        Take and return board size from entry
        :param event: For hotkey
        """
        try:
            # size that is returned
            self.size = int(self.__size.get())

            if self.size in range(10, 36):
                self.__window.destroy()
                self.success = True
            else:
                raise Exception("Size out of bounds")
        except:
             showerror("Error!", "Size must be integer between 10-35")
