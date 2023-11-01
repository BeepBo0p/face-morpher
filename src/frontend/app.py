"""
This file specifies the UI including the buttons and the layout of the UI.
"""
import tkinter as tk
import os

# Get width and height of the screen
app = tk.Tk()

sw, sh = app.winfo_screenwidth(), app.winfo_screenheight()

print("Screen Width: ", sw)
print("Screen Height: ", sh)

screen_coverage = 0.6

w, h = screen_coverage*sw, screen_coverage*sh


print("Width: ", w)
print("Height: ", h)

menu = tk.Frame(master=app, width=0.05*w, height=h, bg="#FFFFFF")
menu.pack(side=tk.LEFT, fill=tk.Y)

# the right canvas for displaying the image
canvas = tk.Canvas(master=app, width=0.95*w, height=h, bg="#808080")
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

app.mainloop()