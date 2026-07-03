import tkinter as tk
from tkinter import font
import math

class Calculator:
    def __init__(self, root):
        self.root = root
        self.root.title("Complex Calculator")
        self.root.geometry("500x600")
        self.root.resizable(False, False)
        
        self.expression = ""
        self.create_widgets()
    
    def create_widgets(self):
        # Display
        display_font = font.Font(family="Arial", size=16, weight="bold")
        self.display = tk.Entry(
            self.root,
            font=display_font,
            borderwidth=2,
            relief="solid",
            justify="right"
        )
        self.display.pack(fill="both", padx=10, pady=10, ipady=10)
        
        # Button frame
        button_frame = tk.Frame(self.root)
        button_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Button layout
        buttons = [
            ["C", "←", "(", ")", "√"],
            ["7", "8", "9", "/", "^"],
            ["4", "5", "6", "*", "sin"],
            ["1", "2", "3", "-", "cos"],
            ["0", ".", "=", "+", "tan"],
            ["log", "ln", "π", "e", "deg"]
        ]
        
        button_font = font.Font(family="Arial", size=12, weight="bold")
        
        for row_idx, row in enumerate(buttons):
            for col_idx, btn_text in enumerate(row):
                btn = tk.Button(
                    button_frame,
                    text=btn_text,
                    font=button_font,
                    command=lambda text=btn_text: self.on_button_click(text)
                )
                btn.grid(row=row_idx, column=col_idx, sticky="nsew", padx=5, pady=5)
        
        # Configure grid weights
        for i in range(len(buttons)):
            button_frame.grid_rowconfigure(i, weight=1)
        for j in range(5):
            button_frame.grid_columnconfigure(j, weight=1)
    
    def on_button_click(self, char):
        if char == "C":
            self.expression = ""
            self.display.delete(0, tk.END)
        elif char == "←":
            self.expression = self.expression[:-1]
            self.display.delete(0, tk.END)
            self.display.insert(0, self.expression)
        elif char == "=":
            try:
                result = eval(self.expression)
                self.expression = str(result)
                self.display.delete(0, tk.END)
                self.display.insert(0, result)
            except:
                self.display.delete(0, tk.END)
                self.display.insert(0, "Error")
                self.expression = ""
        elif char == "√":
            try:
                result = math.sqrt(float(self.expression))
                self.expression = str(result)
                self.display.delete(0, tk.END)
                self.display.insert(0, result)
            except:
                self.display.delete(0, tk.END)
                self.display.insert(0, "Error")
                self.expression = ""
        elif char == "^":
            self.expression += "**"
            self.display.delete(0, tk.END)
            self.display.insert(0, self.expression)
        elif char == "sin":
            try:
                result = math.sin(math.radians(float(self.expression)))
                self.expression = str(result)
                self.display.delete(0, tk.END)
                self.display.insert(0, result)
            except:
                self.display.delete(0, tk.END)
                self.display.insert(0, "Error")
                self.expression = ""
        elif char == "cos":
            try:
                result = math.cos(math.radians(float(self.expression)))
                self.expression = str(result)
                self.display.delete(0, tk.END)
                self.display.insert(0, result)
            except:
                self.display.delete(0, tk.END)
                self.display.insert(0, "Error")
                self.expression = ""
        elif char == "tan":
            try:
                result = math.tan(math.radians(float(self.expression)))
                self.expression = str(result)
                self.display.delete(0, tk.END)
                self.display.insert(0, result)
            except:
                self.display.delete(0, tk.END)
                self.display.insert(0, "Error")
                self.expression = ""
        elif char == "log":
            try:
                result = math.log10(float(self.expression))
                self.expression = str(result)
                self.display.delete(0, tk.END)
                self.display.insert(0, result)
            except:
                self.display.delete(0, tk.END)
                self.display.insert(0, "Error")
                self.expression = ""
        elif char == "ln":
            try:
                result = math.log(float(self.expression))
                self.expression = str(result)
                self.display.delete(0, tk.END)
                self.display.insert(0, result)
            except:
                self.display.delete(0, tk.END)
                self.display.insert(0, "Error")
                self.expression = ""
        elif char == "π":
            self.expression += str(math.pi)
            self.display.delete(0, tk.END)
            self.display.insert(0, self.expression)
        elif char == "e":
            self.expression += str(math.e)
            self.display.delete(0, tk.END)
            self.display.insert(0, self.expression)
        elif char == "deg":
            self.display.delete(0, tk.END)
            self.display.insert(0, "Degree Mode")
        else:
            self.expression += char
            self.display.delete(0, tk.END)
            self.display.insert(0, self.expression)

if __name__ == "__main__":
    root = tk.Tk()
    calculator = Calculator(root)
    root.mainloop()
