import tkinter as tk
from idlelib.tooltip import Hovertip

class RopeHovertip(Hovertip):
    def __init__(self, widget, text, bg_color='white', font_size=14, delay=1, x_offset=0, y_offset=0, *args, **kwargs):
        super().__init__(widget, text, *args, **kwargs)

        self.widget = widget
        self.bg_color = bg_color
        self.font_size = font_size
        self.delay = delay
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.tooltip_active = False

        # Customize tooltip appearance after it's created
        self.tooltip = self._create_tooltip(text)

        # Bind events
        widget.bind("<Enter>", self.show_tooltip)
        widget.bind("<Leave>", self.hide_tooltip)

    def _create_tooltip(self, text):
        tooltip = tk.Toplevel(self.widget)
        tooltip.wm_overrideredirect(True)
        tooltip.withdraw()  # Hide it initially
        label = tk.Label(tooltip, text=text, justify='left', bg=self.bg_color, font=("Helvetica", self.font_size))
        label.pack()
        return tooltip

    def show_tooltip(self, event):
        if not self.tooltip_active:
            self.tooltip_active = True
            # Schedule the tooltip to appear after the specified delay
            self.tooltip.after(self.delay, self._display_tooltip)

    def _display_tooltip(self):
        if self.tooltip_active:
            x = self.widget.winfo_rootx() + self.x_offset
            y = self.widget.winfo_rooty() + self.y_offset
            self.tooltip.geometry(f"+{x}+{y}")
            self.tooltip.deiconify()

    def hide_tooltip(self, event):
        self.tooltip_active = False
        self.tooltip.withdraw()
