import tkinter as tk
from tkinter import font
from PIL import Image, ImageTk
from  rope.Dicts import DEFAULT_DATA
import rope.Styles as style
import customtkinter as ctk

#import inspect print(inspect.currentframe().f_back.f_code.co_name, 'resize_image')

icon_off = None
icon_on = None

def load_switch_icons():
    global icon_off, icon_on
    icon_off_img = Image.open(style.icon['IconOff']).resize((40, 40), Image.ANTIALIAS)
    icon_on_img = Image.open(style.icon['IconOn']).resize((40, 40), Image.ANTIALIAS)
    icon_off = ImageTk.PhotoImage(icon_off_img)
    icon_on = ImageTk.PhotoImage(icon_on_img)

class CTkScrollableFrame(ctk.CTkFrame):
    def __init__(self, parent, allow_drag_and_drop=True, min_row_for_drag=0, allowed_widget_type=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.is_resizing = False

        self.canvas = ctk.CTkCanvas(self, bg=style.main, highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        self.scrollbar = ctk.CTkScrollbar(self, orientation="vertical", command=self.canvas.yview)
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Definisco lo scrollable_frame come un sottoclasse di CTkFrame per includere i metodi necessari
        if allow_drag_and_drop:
            self.scrollable_frame = ScrollableFrameContent(self.canvas, canvas=self.canvas, allowed_widget_type=allowed_widget_type, **kwargs)
        else:
            self.scrollable_frame = ctk.CTkFrame(self.canvas, **kwargs)

        self.scrollable_frame.bind("<Configure>", self.on_frame_configure)

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        parent.bind("<Configure>", self.on_resize)

    def on_frame_configure(self, event=None):
        if not self.is_resizing:
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def on_resize(self, event=None):
        self.is_resizing = True
        self.canvas.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.is_resizing = False

class ScrollableFrameContent(ctk.CTkFrame):
    def __init__(self, parent, canvas, allowed_widget_type=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.canvas = canvas  # Referenza al canvas per controllare lo scrolling
        self.allowed_widget_type = allowed_widget_type  # Tipo di widget che può essere trascinato
        self.drag_data = {"widget": None, "start_row": None, "start_y": None, "placeholder": None, "floating_label": None}

    def start_drag(self, event):
        if event.num != 1:  # Assicuriamoci di gestire solo il pulsante sinistro
            return

        # rimuovi "floating_label" se rimasta appesa
        self.cancel_drag()

        widget = event.widget
        while not isinstance(widget, tk.Frame) and not isinstance(widget, tk.Label):
            widget = widget.master

        # Controlla se il tipo di widget è consentito
        if self.allowed_widget_type:
            # Verifica che il widget abbia l'attributo `draggable_object_instance`
            if not hasattr(widget.master, 'draggable_object_instance'):
                return
            # Verifica che il tipo del controllo corrisponda a quello consentito
            if not isinstance(widget.master.draggable_object_instance, self.allowed_widget_type):
                return

        self.drag_data["widget"] = widget.master  # Assume che il widget sia un Label o altro figlio di frame
        self.drag_data["start_row"] = widget.master.grid_info()["row"]
        self.drag_data["start_y"] = event.y_root

        # Crea un placeholder per mantenere lo spazio
        self.drag_data["placeholder"] = ctk.CTkFrame(self, width=widget.master.winfo_width(), height=widget.master.winfo_height(), fg_color="transparent")
        self.drag_data["placeholder"].grid(row=self.drag_data["start_row"], column=0, padx=widget.master.grid_info()["padx"], pady=widget.master.grid_info()["pady"])

        widget.master.lift()

        # Crea la floating label
        self.drag_data["floating_label"] = ctk.CTkLabel(self, text=widget.cget("text"), fg_color="yellow", text_color="black", corner_radius=5)
        self.drag_data["floating_label"].place(x=event.x_root - self.winfo_rootx(), y=event.y_root - self.winfo_rooty())

    def on_drag(self, event):
        if not self.drag_data["widget"]:  # Se non c'è un widget in drag, esci
            return

        widget = self.drag_data["widget"]
        y = event.y_root - self.winfo_rooty()
        new_row = max(0, min(self.grid_size()[1] - 1, y // widget.winfo_height()))

        if new_row != self.drag_data["start_row"]:
            self.drag_data["placeholder"].grid(row=new_row, column=0)

        # Aggiorna la posizione della floating label
        self.drag_data["floating_label"].place(x=event.x_root - self.winfo_rootx(), y=event.y_root - self.winfo_rooty())

        # Implementazione dell'autoscroll
        self.autoscroll(event)

    def autoscroll(self, event):
        # Calcola la posizione relativa del mouse rispetto al canvas
        canvas_y = event.y_root - self.canvas.winfo_rooty()

        # Determina se il mouse è vicino ai bordi del canvas
        if canvas_y < 20:  # Vicino alla parte superiore
            self.scroll_slowly(-0.005)  # Scroll lento verso l'alto
        elif canvas_y > self.canvas.winfo_height() - 20:  # Vicino alla parte inferiore
            self.scroll_slowly(0.005)  # Scroll lento verso il basso

    def scroll_slowly(self, delta):
        current_view = self.canvas.yview()
        new_view = current_view[0] + delta
        new_view = max(0, min(new_view, 1))  # Limita la vista tra 0 e 1
        self.canvas.yview_moveto(new_view)

    def end_drag(self, event):
        if event.num != 1:  # Assicuriamoci di gestire solo il pulsante sinistro
            return

        widget = self.drag_data["widget"]
        if widget:
            y = event.y_root - self.winfo_rooty()
            new_row = max(0, min(self.grid_size()[1] - 1, y // widget.winfo_height()))

            # Verifica che il widget nella nuova riga sia del tipo consentito
            target_widgets = self.grid_slaves(row=new_row)
            found_valid_target = False

            for target_widget in target_widgets:
                # Se il widget non è del tipo che ci interessa, lo ignoriamo e cerchiamo di trovare un altro widget nella stessa riga
                if not hasattr(target_widget, 'draggable_object_instance'):
                    continue  # Ignora e continua la ricerca

                # Verifica se il widget è del tipo consentito
                if isinstance(target_widget.draggable_object_instance, self.allowed_widget_type):
                    # Trovato un widget valido, esci dal ciclo
                    found_valid_target = True
                    break

            # Se nessun widget valido è stato trovato nella riga, annulla il drag
            if not found_valid_target:
                #print("Drop non consentito: nessun widget valido trovato nella riga di destinazione.")
                self.cancel_drag()
                return

            if new_row != self.drag_data["start_row"]:
                self.rearrange_rows(self.drag_data["start_row"], new_row)

        self.cleanup_drag()

    def cancel_drag(self):
        """Annulla il drag riportando il widget alla posizione originale."""
        if self.drag_data["widget"]:
            self.drag_data["widget"].grid(row=self.drag_data["start_row"], column=0)
        self.cleanup_drag()

    def cleanup_drag(self):
        """Pulisce i dati di drag e ripristina lo stato."""
        if self.drag_data["floating_label"]:
            self.drag_data["floating_label"].destroy()
        if self.drag_data["placeholder"]:
            self.drag_data["placeholder"].destroy()

        self.drag_data = {"widget": None, "start_row": None, "start_y": None, "placeholder": None, "floating_label": None}

    def rearrange_rows(self, start_row, new_row):
        widgets = []
        for row in range(min(start_row, new_row), max(start_row, new_row) + 1):
            for widget in self.grid_slaves(row=row):
                widgets.append(widget)

        if start_row < new_row:
            for widget in widgets:
                current_row = widget.grid_info()["row"]
                widget.grid(row=current_row - 1)

        elif start_row > new_row:
            for widget in reversed(widgets):
                current_row = widget.grid_info()["row"]
                widget.grid(row=current_row + 1)

        self.drag_data["widget"].grid(row=new_row, column=0)

class Separator_x():
    def __init__(self, parent, x, y):
        self.parent = parent
        self.x = x
        self.y = y
        self.parent.update()
        self.blank = tk.PhotoImage()
        self.sep = tk.Label(self.parent, bg='#090909', image=self.blank, compound='c', border=0, width=self.parent.winfo_width(), height=1)
        self.sep.place(x=self.x, y=self.y)
        self.is_resizing = False
        # self.parent.bind('<Configure>', self.update_sep_after_window_resize)

    # def update_sep_after_window_resize(self, event):
        # self.parent.update()
        # self.sep.configure(width=self.parent.winfo_width())

    def hide(self):
        if not self.is_resizing:
            self.sep.place_forget()

    def unhide(self):
        if not self.is_resizing:
            self.parent.update()
            self.sep.place(x=self.x, y=self.y)
            self.sep.configure(width=self.parent.winfo_width())

class Separator_y():
    def __init__(self, parent, x, y):
        self.parent = parent
        self.x = x
        self.y = y
        self.parent.update()
        self.blank = tk.PhotoImage()
        self.sep = tk.Label(self.parent, bg='#090909', image=self.blank, compound='c', border=0, width=1, height=self.parent.winfo_height())
        self.sep.place(x=self.x, y=self.y)
        self.is_resizing = False
        # self.parent.bind('<Configure>', self.update_sep_after_window_resize)

    # def update_sep_after_window_resize(self, event):
        # self.parent.update()
        # self.sep.configure(height=self.parent.winfo_height())

    def hide(self):
        if not self.is_resizing:
            self.sep.place_forget()

    def unhide(self):
        if not self.is_resizing:
            self.parent.update()
            self.sep.place(x=self.x, y=self.y)
            self.sep.configure(height=self.parent.winfo_height())

class Text():
    def __init__(self, parent, text, style_level, x, y, width, height):
        self.blank = tk.PhotoImage()
        self.is_resizing = False

        if style_level == 1:
            self.style = style.text_1
        elif style_level == 2:
            self.style = style.text_2
        elif style_level == 3:
            self.style = style.text_3

        self.label = tk.Label(parent, self.style, image=self.blank, compound='c', text=text, anchor='w', width=width, height=height)
        self.label.place(x=x, y=y)

    def configure(self, text):
        self.label.configure(text=text)

class Scrollbar_x():
    def __init__(self, parent, child):

        self.child = child

        self.trough_short_dim = 15
        self.trough_long_dim = []
        self.handle_short_dim = self.trough_short_dim * 0.5

        self.left_of_handle = []
        self.middle_of_handle = []
        self.right_of_handle = []

        self.old_coord = 0
        self.is_resizing = False

        # Child data
        self.child.bind('<Configure>', self.resize_scrollbar)

        # Set the canvas
        self.scrollbar_canvas = parent
        self.scrollbar_canvas.configure(height=self.trough_short_dim)
        self.scrollbar_canvas.bind("<MouseWheel>", self.scroll)
        self.scrollbar_canvas.bind("<ButtonPress-1>", self.scroll)
        self.scrollbar_canvas.bind("<B1-Motion>", self.scroll)

        # Draw handle
        self.resize_scrollbar(None)

    def resize_scrollbar(self, event):  # on window updates
        self.child.update()
        self.child.configure(scrollregion=self.child.bbox("all"))

        # Reconfigure data
        self.trough_long_dim = self.child.winfo_width()
        self.scrollbar_canvas.delete('all')
        self.scrollbar_canvas.configure(width=self.trough_long_dim)

        # Redraw the scrollbar
        y1 = (self.trough_short_dim - self.handle_short_dim) / 2
        y2 = self.trough_short_dim - y1
        x1 = self.child.xview()[0] * self.trough_long_dim
        x2 = self.child.xview()[1] * self.trough_long_dim

        self.middle_of_handle = self.scrollbar_canvas.create_rectangle(x1, y1, x2, y2, fill='grey25', outline='')

    def scroll(self, event):
        delta = 0

        # Get handle dimensions
        handle_x1 = self.scrollbar_canvas.coords(self.middle_of_handle)[0]
        handle_x2 = self.scrollbar_canvas.coords(self.middle_of_handle)[2]
        handle_center = (handle_x2 - handle_x1) / 2 + handle_x1
        handle_length = handle_x2 - handle_x1

        if event.type == '38':  # mousewheel
            delta = -int(event.delta / 20.0)

        elif event.type == '4':  # l-button press
            # If the mouse coord is within the handle don't jump the handle
            if event.x > handle_x1 and event.x < handle_x2:
                self.old_coord = event.x
            else:
                self.old_coord = handle_center

            delta = event.x - self.old_coord

        elif event.type == '6':  # l-button drag
            delta = event.x - self.old_coord

        # Do some bounding
        if handle_x1 + delta < 0:
            delta = -handle_x1
        elif handle_x2 + delta > self.trough_long_dim:
            delta = self.trough_long_dim - handle_x2

        # Update the scrollbar
        self.scrollbar_canvas.move(self.middle_of_handle, delta, 0)

        # Get the new handle position to calculate the change for the child
        handle_x1 = self.scrollbar_canvas.coords(self.middle_of_handle)[0]

        # Move the child
        self.child.xview_moveto(handle_x1 / self.trough_long_dim)

        self.old_coord = event.x

    def set(self, value):
        handle_x1 = self.scrollbar_canvas.coords(self.middle_of_handle)[0]
        handle_x2 = self.scrollbar_canvas.coords(self.middle_of_handle)[2]
        handle_center = (handle_x2 - handle_x1) / 2 + handle_x1

        coord_del = self.scrollbar_canvas.winfo_width() * value - handle_center
        self.old_coord = self.scrollbar_canvas.winfo_width() * value

        self.scrollbar_canvas.move(self.middle_of_handle, coord_del, 0)

    def hide(self):
        pass

    def unhide(self):
        pass

class Scrollbar_y():
    def __init__(self, parent, child):

        self.child = child

        self.trough_short_dim = 15
        self.trough_long_dim = []
        self.handle_short_dim = self.trough_short_dim*0.5

        self.top_of_handle = []
        self.middle_of_handle = []
        self.bottom_of_handle = []

        self.old_coord = 0
        self.is_resizing = False

        # Child data
        self.child.bind('<Configure>', self.resize_scrollbar)

        # Set the canvas
        self.scrollbar_canvas = parent
        self.scrollbar_canvas.configure(width=self.trough_short_dim)
        self.scrollbar_canvas.bind("<MouseWheel>", self.scroll)
        self.scrollbar_canvas.bind("<ButtonPress-1>", self.scroll)
        self.scrollbar_canvas.bind("<B1-Motion>", self.scroll)

        # Draw handle
        self.resize_scrollbar(None)

    def resize_scrollbar(self, event): # on window updates
        self.child.update()
        self.child.configure(scrollregion=self.child.bbox("all"))

        # Reconfigure data
        self.trough_long_dim = self.child.winfo_height()
        self.scrollbar_canvas.delete('all')
        self.scrollbar_canvas.configure(height=self.trough_long_dim)

        # Redraw the scrollbar
        x1 = (self.trough_short_dim-self.handle_short_dim)/2
        x2 = self.trough_short_dim-x1
        y1 = self.child.yview()[0]*self.trough_long_dim
        y2 = self.child.yview()[1]*self.trough_long_dim

        self.middle_of_handle = self.scrollbar_canvas.create_rectangle(x1, y1, x2, y2, fill='grey25', outline='')

    def scroll(self, event):
        delta = 0

        # Get handle dimensions
        handle_y1 = self.scrollbar_canvas.coords(self.middle_of_handle)[1]
        handle_y2 = self.scrollbar_canvas.coords(self.middle_of_handle)[3]
        handle_center = (handle_y2-handle_y1)/2 + handle_y1
        handle_length = handle_y2-handle_y1

        if event.type == '38': # mousewheel
            delta = -int(event.delta/20.0)

        elif event.type == '4': # l-button press
            # If the mouse coord is within the handle dont jump the handle
            if event.y > handle_y1 and event.y<handle_y2:
                self.old_coord = event.y
            else:
                self.old_coord = handle_center

            delta = event.y-self.old_coord

        elif event.type == '6': # l-button drag
            delta = event.y-self.old_coord

        # Do some bounding
        if handle_y1+delta<0:
            delta = -handle_y1
        elif handle_y2+delta>self.trough_long_dim:
            delta = self.trough_long_dim-handle_y2

        # update the scrollbar
        self.scrollbar_canvas.move(self.middle_of_handle, 0, delta)

        # Get the new handle postition to calculate the change for the child
        handle_y1 = self.scrollbar_canvas.coords(self.middle_of_handle)[1]

        # Move the child
        self.child.yview_moveto(handle_y1/self.trough_long_dim)

        self.old_coord = event.y

    def set(self, value):
        handle_y1 = self.scrollbar_canvas.coords(self.middle_of_handle)[1]
        handle_y2 = self.scrollbar_canvas.coords(self.middle_of_handle)[3]
        handle_center = (handle_y2-handle_y1)/2 + handle_y1

        coord_del = self.scrollbar_canvas.winfo_height()*value-handle_center
        self.old_coord = self.scrollbar_canvas.winfo_height()*value

        self.scrollbar_canvas.move(self.middle_of_handle, 0, coord_del)

    def hide(self):
        pass

    def unhide(self):
        pass

class Timeline():
    def __init__(self, parent, widget, temp_toggle_swapper, temp_toggle_enhancer, temp_toggle_faces_editor, add_action):
        self.parent = parent
        self.add_action = add_action
        self.temp_toggle_swapper = temp_toggle_swapper
        self.temp_toggle_enhancer = temp_toggle_enhancer
        self.temp_toggle_faces_editor = temp_toggle_faces_editor
        self.frame_length = 0
        self.height = 20
        self.counter_width = 40

        self.entry_string = tk.StringVar()
        self.entry_string.set(0)

        self.last_position = 0

        # Widget variables
        self.max_ = 100#video_length

        self.handle = []
        self.slider_left = []
        self.slider_right = []

        self.fps = 0
        self.time_elapsed_string = tk.StringVar()
        self.time_elapsed_string.set("00:00:00")

        # Event trigget for window resize
        self.parent.bind('<Configure>', self.window_resize)

        # Add the Slider Canvas to the frame
        self.slider = tk.Canvas(self.parent, style.timeline_canvas, height=self.height)
        self.slider.place(x=0, y=0)
        self.slider.bind('<B1-Motion>', lambda e: self.update_timeline_handle(e, True))
        self.slider.bind('<ButtonPress-1>', lambda e: self.update_timeline_handle(e, True))
        self.slider.bind('<ButtonRelease-1>', lambda e: self.update_timeline_handle(e, True))
        self.slider.bind('<MouseWheel>', lambda e: self.update_timeline_handle(e, True))
        self.slider.bind('<Button-4>', lambda e: self.update_timeline_handle(e, True))
        self.slider.bind('<Button-5>', lambda e: self.update_timeline_handle(e, True))

        # Add the Entry to the frame
        self.entry_width = 40
        self.entry = tk.Entry(self.parent, style.entry_3, textvariable=self.entry_string)
        self.entry.bind('<Return>', lambda event: self.entry_input(event))

        # Add the Time Entry to the frame
        self.time_width = 40
        self.time_entry = tk.Entry(self.parent, style.entry_3, textvariable=self.time_elapsed_string, width=8)
        self.time_entry.bind('<Return>', lambda event: self.time_entry_input(event))

    def draw_timeline(self):
        self.slider.delete('all')

        # Configure widths and placements
        self.slider.configure(width=self.frame_length)
        self.entry.place(x=self.parent.winfo_width()-self.counter_width, y=0)
        self.time_entry.place(x=(self.parent.winfo_width()-self.counter_width-self.time_width-40) / 2, y=25)

        # Draw the slider
        slider_pad = 20
        entry_pad = 20
        self.slider_left = slider_pad
        self.slider_right = self.frame_length-entry_pad-self.entry_width
        slider_center = (self.height)/2

        line_loc = self.pos2coord(self.last_position)

        line_height = 8
        line_width = 1.5
        line_x1 = line_loc-line_width
        line_y1 = slider_center -line_height
        line_x2 = line_loc+line_width
        line_y2 = slider_center +line_height

        trough_x1 = self.slider_left
        trough_y1 = slider_center-1
        trough_x2 = self.slider_right
        trough_y2 = slider_center+1

        self.slider.create_rectangle(trough_x1, trough_y1, trough_x2, trough_y2, fill='#43474D', outline='')
        self.handle = self.slider.create_rectangle(line_x1, line_y1, line_x2, line_y2, fill='#FFFFFF', outline='')

    def coord2pos(self, coord):
        return float(coord-self.slider_left)*self.max_/(self.slider_right-self.slider_left)

    def pos2coord(self, pos):
        return float(float(pos)*(self.slider_right-self.slider_left)/self.max_ + self.slider_left)

    def update_timeline_handle(self, event, also_update_entry=False):
        requested = True

        if isinstance(event, float):
            position = event
            requested = False
        else:
            if event.type == '38': # windows mousewheel
                self.add_action("play_video", "stop")
                position = self.last_position+int(event.delta/120.0)

            elif event.type == '4': # l-button press or Unix mousewheel

                if event.num == 4:  # Mouse wheel up (Unix)
                    self.add_action("play_video", "stop")
                    position = self.last_position + self.fps
                elif event.num == 5:  # Mouse wheel down (Unix)
                    self.add_action("play_video", "stop")
                    position = self.last_position - self.fps
                else:
                    x_coord = float(event.x)
                    position = self.coord2pos(x_coord)

                    # Turn off swapping, enhancer, face editor
                    self.temp_toggle_swapper('off')
                    self.temp_toggle_enhancer('off')
                    self.temp_toggle_faces_editor('off')
                    self.add_action("play_video", "stop")

            elif event.type == '5': # l-button release
                x_coord = float(event.x)
                position = self.coord2pos(x_coord)

                # Turn on swapping, if it was already on and request new frame
                self.temp_toggle_swapper('on')
                self.temp_toggle_enhancer('on')
                self.temp_toggle_faces_editor('on')

            elif event.type == '6': # l-button drag
                x_coord = float(event.x)
                position = self.coord2pos(x_coord)

        # constrain mousewheel movement
        if position < 0: position = 0
        elif position > self.max_: position = self.max_

        # Find closest position increment
        position = round(position)

        # moving sends many events, so only update when the next frame is reached
        if position != self.last_position:
            # Move handle to coordinate based on position
            self.slider.move(self.handle, self.pos2coord(position) - self.pos2coord(self.last_position), 0)

            if requested:
                self.add_action("get_requested_video_frame", position)

            # Save for next time
            self.last_position = position

            if also_update_entry:
                self.entry_string.set(str(position))
                self.update_time_elapsed(position)

    def entry_input(self, event):
    # event.char
        self.entry.update()
        try:
            input_num = float(self.entry_string.get())
            self.update_timeline_handle(input_num, False)
        except:
            return

    def time_entry_input(self, event):
    # event.char
        self.time_entry.update()
        try:
            time_string = self.time_elapsed_string.get()

            # Divide la stringa in ore, minuti e secondi
            hours, minutes, seconds = map(int, time_string.split(':'))

            # Verifica che le ore, i minuti e i secondi siano nei range corretti
            if not (0 <= hours <= 23) or not (0 <= minutes <= 59) or not (0 <= seconds <= 59):
                return

            # get total seconds
            total_seconds = hours * 3600 + minutes * 60 + seconds

            # get the current frame
            current_frame = float(total_seconds * self.fps)

            self.update_timeline_handle(current_frame, False)

            # sync with entry_string
            self.entry_string.set(str(self.last_position))
        except:
            return

    def update_time_elapsed(self, position):
        try:
            time_elapsed = position / self.fps  # time elapsed

            # Converti il tempo in ore, minuti e secondi
            hours, remainder = divmod(time_elapsed, 3600)
            minutes, seconds = divmod(remainder, 60)

            # Formatta il tempo come HH:MM:SS
            time_elapsed_string = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

            # Imposta la stringa formattata nella variabile desiderata
            self.time_elapsed_string.set(time_elapsed_string)
        except:
            return

    def set(self, value):
        self.update_timeline_handle(float(value), also_update_entry=True)

    def get(self):
        return int(self.last_position)

    def set_length(self, value):
        self.max_ = value
        self.update_timeline_handle(float(self.last_position), also_update_entry=True)

    def get_length(self):
        return int(self.max_)

    def set_fps(self, value):
        self.fps = value

    # Event when the window is resized
    def window_resize(self, event):
        self.parent.update()
        self.frame_length = self.parent.winfo_width()
        self.draw_timeline()

class Button():
    def __init__(self, parent, name, style_level, function, args, data_type, x, y, width=125, height=20):
        self.parent = parent
        self.default_data = DEFAULT_DATA
        self.name = name
        self.function = function
        self.args = args
        self.info = []
        self.state = []
        self.hold_state = []
        self.error = []
        self.data_type = data_type
        self.visible = True
        self.is_resizing = False

        # Save botton position for unhiding
        self.x = x
        self.y = y
        self.width = width
        self.height = height

        if style_level == 1:
            self.button_style = style.button_1
        elif style_level == 2:
            self.button_style = style.button_2
        elif style_level == 3:
            self.button_style = style.button_3

        # Add Icon
        if self.default_data[self.name+'Display'] == 'both':
            img = Image.open(self.default_data[self.name+'IconOn'])
            resized_image= img.resize((20,20), Image.ANTIALIAS)
            self.icon_on = ImageTk.PhotoImage(resized_image)
            img = Image.open(self.default_data[self.name+'IconOff'])
            resized_image= img.resize((20,20), Image.ANTIALIAS)
            self.icon_off = ImageTk.PhotoImage(resized_image)
            img = Image.open(self.default_data[self.name+'IconHover'])
            resized_image= img.resize((20,20), Image.ANTIALIAS)
            self.icon_hover = ImageTk.PhotoImage(resized_image)

            text = ' '+self.default_data[self.name+'Text']

        elif self.default_data[self.name+'Display'] == 'icon':
            img = Image.open(self.default_data[self.name+'IconOn'])
            resized_image= img.resize((20,20), Image.ANTIALIAS)
            self.icon_on = ImageTk.PhotoImage(resized_image)
            img = Image.open(self.default_data[self.name+'IconOff'])
            resized_image= img.resize((20,20), Image.ANTIALIAS)
            self.icon_off = ImageTk.PhotoImage(resized_image)
            img = Image.open(self.default_data[self.name+'IconHover'])
            resized_image= img.resize((20,20), Image.ANTIALIAS)
            self.icon_hover = ImageTk.PhotoImage(resized_image)

            text = ''

        elif self.default_data[self.name+'Display'] == 'text':
            self.icon_on = tk.PhotoImage()
            self.icon_off = tk.PhotoImage()
            self.icon_hover = tk.PhotoImage()

            text = ' '+self.default_data[self.name+'Text']

        # Create Button and place
        self.button = tk.Button(parent, self.button_style, compound='left', text=text, anchor='w')
        self.button.configure(width=width, height=height)
        self.button.place(x=x, y=y)

        self.button.bind("<Enter>", lambda event: self.on_enter())
        self.button.bind("<Leave>", lambda event: self.on_leave())

        if self.function != None:
            if self.args != None:
                self.button.configure(command=lambda: self.function(self.args))
            else:
                self.button.configure(command=lambda: self.function())

        # Set inital state
        self.button.configure(image=self.icon_on)

        if self.default_data[self.name+'State']:
            self.enable_button()

        else:
            self.disable_button()

    def add_info_frame(self, info):
        self.info = info

    def on_enter(self):
        if self.info:
            self.info.configure(text=self.default_data[self.name+'InfoText'])

        if not self.state and not self.error:
            self.button.configure(image=self.icon_hover)
            self.button.configure(fg='#B1B1B2')

    def on_leave(self):
        if not self.state and not self.error:

            self.button.configure(image=self.icon_off)
            self.button.configure(fg='#828282')

    def enable_button(self):

        self.button.configure(image=self.icon_on)
        self.button.configure(fg='#FFFFFF')
        self.state = True
        self.error = False

    def disable_button(self):

        self.button.configure(image=self.icon_off)
        self.button.configure(fg='#828282')
        self.state = False
        self.error = False

    def toggle_button(self):
        self.state = not self.state

        if self.state:
            self.button.configure(image=self.icon_on)
            self.button.configure(fg='#FFFFFF')
        else:
            self.button.configure(image=self.icon_off)
            self.button.configure(fg='#828282')

    def temp_disable_button(self):
        self.hold_state = self.state
        self.state = False

    def temp_enable_button(self):
        self.state = self.hold_state

    def error_button(self):

        self.button.configure(image=self.icon_off)
        self.button.configure(fg='light goldenrod')
        self.state = False
        self.error = True

    def get(self):
        return self.state

    def set(self, value, request_frame=True):
        if value:
            self.enable_button()

        elif not value:
            self.disable_button()
        if request_frame:
            if self.function != None:
                if self.args != None:
                    self.function(self.args)
                else:
                    self.function()

    def hide(self):
        if not self.is_resizing:
            self.button.place_forget()
            self.visible = False

    def unhide(self):
        if not self.is_resizing:
            self.button.place(x=self.x, y=self.y, width=self.width, height=self.height)
            self.visible = True

    def get_data_type(self):
        return self.data_type

    def load_default(self):
        self.set(self.default_data[self.name+'State'])

class TextSelection():
    def __init__(self, parent, name, display_text, style_level, function, argument, data_type, width, height, row, column, padx, pady, text_percent):
        self.blank = tk.PhotoImage()

        self.default_data = DEFAULT_DATA
        # Capture inputs as instance variables
        self.parent = parent
        self.name = name
        self.function = function
        self.argument = argument
        self.data_type = data_type
        self.width = width
        self.height = height
        self.style = []
        self.info = []
        self.row = row
        self.column = column
        self.visible = True
        self.is_resizing = False

        if style_level == 3:
            self.frame_style = style.canvas_frame_label_3
            self.text_style = style.text_3
            self.sel_off_style = style.text_selection_off_3
            self.sel_on_style = style.text_selection_on_3

        if style_level == 2:
            self.frame_style = style.canvas_frame_label_2
            self.text_style = style.text_2
            self.sel_off_style = style.text_selection_off_2
            self.sel_on_style = style.text_selection_on_2

        self.display_text = display_text+' '

        self.textselect_label = {}

        # Initial data
        self.selection = self.default_data[self.name+'Mode']

        # Frame to hold everything
        self.frame = tk.Frame(self.parent, self.frame_style, width=self.width, height=self.height)
        self.frame.grid(row=row, column=column, sticky='NEWS', padx=padx, pady=pady)
        self.frame.bind("<Enter>", lambda event: self.on_enter())

        self.text_width = int(width*(1.0-text_percent))

        # Create the text on the left
        self.text_label = tk.Label(self.frame, self.text_style, image=self.blank, compound='c', text=self.display_text, anchor='e', width=self.text_width, height=height)
        self.text_label.place(x=0, y=0)

        # Loop through the parameter modes, create a label
        # Gotta find the size of the buttons according to the font
        self.font = tk.font.Font(family="Segoe UI", size=10, weight="normal")
        x_spacing = self.text_width + 10

        for mode in self.default_data[self.name+'Modes']:
            # Get size of text in pixels
            m_len = self.font.measure(mode)

            # Create a label with the text
            self.textselect_label[mode] = tk.Label(self.frame, self.sel_off_style, text=mode, image=self.blank, compound='c', anchor='c', width=m_len, height=height)
            self.textselect_label[mode].place(x=x_spacing, y=0)
            self.textselect_label[mode].bind("<ButtonRelease-1>", lambda event, mode=mode: self.select_ui_text_selection(mode))

            # Initial value
            if mode==self.selection:
                self.textselect_label[mode].configure(self.sel_on_style)

            x_spacing = x_spacing + m_len+10

    def select_ui_text_selection(self, selection, request_frame=True):
        # Loop over all of the Modes
        for mode in self.default_data[self.name+'Modes']:

            # If the Mode has been selected
            if mode==selection:
                # Set state to true
                self.textselect_label[mode].configure(self.sel_on_style)
                self.selection = mode
                if request_frame:
                    self.function(self.argument, self.name)

            else:
                self.textselect_label[mode].configure(self.sel_off_style)

    def add_info_frame(self, info):
        self.info = info

    def on_enter(self):
        if self.info:
            self.info.configure(text=self.default_data[self.name+'InfoText'])

    def get(self):
        return self.selection

    def set(self, value, request_frame=True):
        self.select_ui_text_selection(value, request_frame)

    def hide(self):
        if not self.is_resizing:
            self.frame.grid_remove()
            self.visible = False

    def unhide(self):
        if not self.is_resizing:
            self.frame.grid()
            self.visible = True

    def get_data_type(self):
        return self.data_type

    def load_default(self):
        self.set(self.default_data[self.name+'Mode'])

class TextSelectionComboBox:
    def __init__(self, parent, name, display_text, style_level, function, argument, data_type, width, height, row, column, padx, pady, text_percent, combobox_width):
        self.blank = tk.PhotoImage()

        self.default_data = DEFAULT_DATA
        self.parent = parent
        self.name = name
        self.function = function
        self.argument = argument
        self.data_type = data_type
        self.width = width
        self.height = height
        self.style = []
        self.info = []
        self.display_text = display_text + ' '
        self.selection = self.default_data[self.name + 'Mode']
        self.row = row
        self.column = column
        self.visible = True
        self.is_resizing = False

        self.styles = {
            3: (style.canvas_frame_label_3, style.text_3, style.text_selection_off_3, style.text_selection_on_3),
            2: (style.canvas_frame_label_2, style.text_2, style.text_selection_off_2, style.text_selection_on_2)
        }

        self.frame_style, self.text_style, self.sel_off_style, self.sel_on_style = self.styles.get(style_level)

        self.frame = tk.Frame(self.parent, self.frame_style, width=self.width, height=self.height)
        self.frame.grid(row=row, column=column, sticky='NEWS', padx=padx, pady=pady)
        self.frame.bind("<Enter>", lambda event: self.on_enter())

        self.text_width = int(width * (1.0 - text_percent))
        self.combobox_width = combobox_width

        self.text_label = tk.Label(self.frame, self.text_style, image=self.blank, compound='c', text=self.display_text, anchor='e', width=self.text_width, height=height)
        self.text_label.place(x=0, y=0)

        modes = self.default_data[self.name + 'Modes']

        self.font = ctk.CTkFont(family="Segoe UI", size=10, weight="normal")
        self.combo_box = ctk.CTkComboBox(self.frame, values=modes, command=self.select_ui_text_selection, font=self.font, dropdown_font=self.font, state="readonly", width=self.combobox_width, height=height, border_width=1, fg_color=style.main, dropdown_fg_color=style.main)
        self.combo_box.place(x=self.text_width + 10, y=0)
        self.combo_box.set(self.selection)

    def select_ui_text_selection(self, selection):
        self.selection = selection
        self.function(self.argument, self.name)

    def add_info_frame(self, info):
        self.info = info

    def on_enter(self):
        if self.info:
            self.info.configure(text=self.default_data[self.name + 'InfoText'])

    def get(self):
        return self.selection

    def set(self, value, request_frame=True):
        self.combo_box.set(value)
        if request_frame:
            self.function(self.argument, self.name)

    def hide(self):
        if not self.is_resizing:
            self.frame.grid_remove()
            self.visible = False

    def unhide(self):
        if not self.is_resizing:
            self.frame.grid()
            self.visible = True

    def get_data_type(self):
        return self.data_type

    def load_default(self):
        self.set(self.default_data[self.name + 'Mode'])

class Switch2():
    def __init__(self, parent, name, display_text, style_level, function, argument, width, height, row, column, padx, pady, toggle_x=0, toggle_width=40):
        self.blank = tk.PhotoImage()
        self.default_data = DEFAULT_DATA
        # Capture inputs as instance variables
        self.parent = parent
        self.name = name
        self.function = function
        self.argument = argument
        self.data_type = argument
        self.width = width
        self.height = height
        self.toggle_x = toggle_x
        self.toggle_width = toggle_width
        self.style = []
        self.info = []
        self.row = row
        self.column = column
        self.visible = True
        self.is_resizing = False

        # Initial Value
        self.state = self.default_data[name+'State']

        if style_level == 3:
            self.frame_style = style.canvas_frame_label_3
            self.text_style = style.text_3
            self.entry_style = style.entry_3

        self.display_text = display_text
        # Load Icons
        if icon_on == None or icon_off == None:
            load_switch_icons()

        # Frame to hold everything
        self.frame = tk.Frame(self.parent, self.frame_style, width=self.width, height=self.height)
        self.frame.grid(row=row, column=column, sticky='NEWS', padx=padx, pady=pady)
        self.frame.bind("<Enter>", lambda event: self.on_enter())

        text_width = self.width-self.toggle_width

        # Toggle Switch
        self.switch = tk.Label(self.frame, style.parameter_switch_3, image=icon_off, width=toggle_width, height=self.height)
        if self.state:
            self.switch.configure(image=icon_on)

        self.switch.place(x=self.toggle_x, y=2)
        self.switch.bind("<ButtonRelease-1>", lambda event: self.toggle_switch(event))

        # Text
        self.switch_text = tk.Label(self.frame, style.parameter_switch_3, image=self.blank, compound='right', text=self.display_text, anchor='w', width=text_width, height=height-2)
        self.switch_text.place(x=self.toggle_x + self.toggle_width + 10, y=0)

    def toggle_switch(self, event, set_value=None, request_frame=True):
        # flip state
        if set_value==None:
            self.state = not self.state
        else:
            self.state = set_value

        if self.state:
            self.switch.configure(image=icon_on)

        else:
            self.switch.configure(image=icon_off)

        if request_frame:
            self.function(self.argument, self.name, use_markers=False)

    def add_info_frame(self, info):
        self.info = info

    def on_enter(self):
        if self.info:
            self.info.configure(text=self.default_data[self.name+'InfoText'])

    def hide(self):
        if not self.is_resizing:
            self.frame.grid_remove()
            self.visible = False

    def unhide(self):
        if not self.is_resizing:
            self.frame.grid()
            self.visible = True

    def set(self, value, request_frame=True):
        self.toggle_switch(None, value, request_frame)

    def get(self):
        return self.state

    def get_data_type(self):
        return self.data_type

    def load_default(self):
        self.set(self.default_data[self.name+'State'])

class ParamSwitch():
    def __init__(self, parent, name, display_text, style_level, function, value, width, height, row, column, padx, pady, allow_drag_and_drop=True, toggle_x=0, toggle_width=40):
        self.blank = tk.PhotoImage()
        self.parent = parent
        self.name = name
        self.function = function
        self.width = width
        self.height = height
        self.toggle_x = toggle_x
        self.toggle_width = toggle_width
        self.style = []
        self.row = row
        self.column = column
        self.is_resizing = False

        self.visible = value

        if style_level == 3:
            self.frame_style = style.canvas_frame_label_3
            self.text_style = style.text_3
            self.entry_style = style.entry_3

        self.display_text = display_text

        # Load Icons
        if icon_on is None or icon_off is None:
            load_switch_icons()

        # Frame to hold everything
        self.frame = tk.Frame(self.parent, self.frame_style, width=self.width, height=self.height)
        self.frame.grid(row=row, column=column, sticky='NEWS', padx=padx, pady=pady)
        self.frame.bind("<Enter>", lambda event: self.on_enter())

        text_width = self.width - self.toggle_width

        # Toggle Switch
        self.switch = tk.Label(self.frame, style.parameter_switch_3, image=icon_off, width=toggle_width, height=self.height)
        if self.visible:
            self.switch.configure(image=icon_on)
        self.switch.place(x=self.toggle_x, y=2)
        self.switch.bind("<ButtonRelease-1>", lambda event: self.toggle_switch(event))

        # Text
        self.switch_text = tk.Label(self.frame, style.parameter_switch_3, image=self.blank, compound='right', text=self.display_text, anchor='w', width=text_width, height=height - 2)
        self.switch_text.place(x=self.toggle_x + self.toggle_width + 10, y=0)

        # Aggiungi un riferimento a se stesso come attributo del frame
        self.frame.draggable_object_instance = self

        # Bind per drag and drop sul testo
        if allow_drag_and_drop:
            self.switch_text.bind("<ButtonPress-1>", self.parent.start_drag)
            self.switch_text.bind("<B1-Motion>", self.parent.on_drag)
            self.switch_text.bind("<ButtonRelease-1>", self.parent.end_drag)

    def toggle_switch(self, event, set_value=None, request_frame=True):
        if set_value is None:
            self.visible = not self.visible
        else:
            self.visible = set_value

        if self.visible:
            self.switch.configure(image=icon_on)
        else:
            self.switch.configure(image=icon_off)

        if request_frame:
            self.function(self.name, self.visible)

    def on_enter(self):
        pass

    def hide(self):
        pass

    def unhide(self):
        pass

    def set(self, value, request_frame=True):
        self.toggle_switch(None, value, request_frame)

    def get(self):
        return self.visible

class Slider2():
    def __init__(self, parent, name, display_text, style_level, function, argument, width, height, row, column, padx, pady, slider_percent, entry_width=60):
        # self.constants = CONSTANTS
        self.default_data = DEFAULT_DATA
        self.blank = tk.PhotoImage()

        # Capture inputs as instance variables
        self.parent = parent
        self.name = name
        self.function = function
        self.data_type = argument
        self.slider_percent = slider_percent
        self.width = width
        self.height = height
        self.info = []
        self.row = row
        self.column = column
        self.visible = True
        self.is_resizing = False

        # Initial Value
        self.amount = self.default_data[name+'Amount']

        if style_level == 1:
            self.frame_style = style.canvas_frame_label_1
            self.text_style = style.text_1
            self.entry_style = style.entry_3

        elif style_level == 3:
            self.frame_style = style.canvas_frame_label_3
            self.text_style = style.text_3
            self.entry_style = style.entry_3

        # UI-controlled variables
        self.entry_string = tk.StringVar()
        self.entry_string.set(self.amount)

        # Widget variables
        self.min_ = self.default_data[name+'Min']
        self.max_ = self.default_data[name+'Max']
        self.inc_ = self.default_data[name+'Inc']
        self.decimal_places_ = 0
        if str(self.inc_).find('.') != -1:
            self.decimal_places_ = len(str(self.inc_)[str(self.inc_).find('.') + 1:])

        self.display_text = display_text+' '

        # Set up spacing
        # |----------------------|slider_pad|-slider-|entry_pad|-|
        # |---1-slider_percent---|---slider_percent---|
        # |--------------------width------------------|

        # Create a frame to hold it all
        self.frame_width = width
        self.frame_height = height

        self.frame = tk.Frame(self.parent, self.frame_style, width=self.frame_width, height=self.frame_height)
        self.frame.grid(row=row, column=column, sticky='NEWS', padx=padx, pady=pady)
        self.frame.bind("<Enter>", lambda event: self.on_enter())

        # Add the slider Label text to the frame
        self.txt_label_x = 0
        self.txt_label_y = 0
        self.txt_label_width = int(width*(1.0-slider_percent))

        self.label = tk.Label(self.frame, self.text_style, image=self.blank, compound='c', text=self.display_text, anchor='e', width=self.txt_label_width, height=self.height)
        self.label.place(x=self.txt_label_x, y=self.txt_label_y)

        # Add the Slider Canvas to the frame
        self.slider_canvas_x = self.txt_label_width
        self.slider_canvas_y = 0
        self.slider_canvas_width = width-self.txt_label_width

        self.slider = tk.Canvas(self.frame, self.frame_style, width=self.slider_canvas_width, height=self.height)
        self.slider.place(x=self.slider_canvas_x, y=self.slider_canvas_y)
        self.slider.bind('<B1-Motion>', lambda e: self.update_handle(e, True))
        self.slider.bind('<MouseWheel>', lambda e: self.update_handle(e, True))

        # Add the Entry to the frame
        self.entry_width = entry_width
        self.entry_x = self.frame_width-self.entry_width
        self.entry_y = 0

        self.entry = tk.Entry(self.frame, self.entry_style, textvariable=self.entry_string)
        self.entry.place(x=self.entry_x, y=self.entry_y)
        self.entry.bind('<Return>', lambda event: self.entry_input(event))
        self.entry.bind('<Tab>', lambda event: self.entry_input(event))

        # Draw the slider
        self.slider_pad = 20
        self.entry_pad = 20
        self.slider_left = self.slider_pad
        self.slider_right = self.slider_canvas_width-self.entry_pad-self.entry_width
        self.slider_center = (self.height+1)/2

        self.oval_loc = self.pos2coord(self.amount)
        self.oval_radius = 5
        self.oval_x1 = self.oval_loc-self.oval_radius
        self.oval_y1 = self.slider_center-self.oval_radius
        self.oval_x2 = self.oval_loc+self.oval_radius
        self.oval_y2 = self.slider_center+self.oval_radius

        self.trough_x1 = self.slider_left
        self.trough_y1 = self.slider_center-2
        self.trough_x2 = self.slider_right
        self.trough_y2 = self.slider_center+2

        self.slider.create_rectangle(self.trough_x1, self.trough_y1, self.trough_x2, self.trough_y2, fill='#1F1F1F', outline='')
        self.handle = self.slider.create_oval(self.oval_x1, self.oval_y1, self.oval_x2, self.oval_y2, fill='#919191', outline='')

    def coord2pos(self, coord):
        return float((coord-self.slider_left)*(self.max_-self.min_)/(self.slider_right-self.slider_left) + self.min_)

    def pos2coord(self, pos):
        return float((float(pos)-self.min_)*(self.slider_right-self.slider_left)/(self.max_-self.min_) + self.slider_left)

    def update_handle(self, event, also_update_entry=False, request_frame=True):
        if isinstance(event, float):
            position = event

        elif event.type == '38':
            position = self.amount+self.inc_*int(event.delta/120.0)

        elif event.type == '6':
            x_coord = float(event.x)
            position = self.coord2pos(x_coord)

        # constrain mousewheel movement
        if position < self.min_: position = self.min_
        elif position > self.max_: position = self.max_

        # Find closest position increment
        position_inc = round((position-self.min_) / self.inc_)
        position = (position_inc * self.inc_)+self.min_

        # moving sends many events, so only update when the next frame is reached
        if position != self.amount:
            # Move handle to coordinate based on position
            self.slider.move(self.handle, self.pos2coord(position) - self.pos2coord(self.amount), 0)

            # Save for next time
            self.amount = position

            if also_update_entry:
                self.entry_string.set(str(format(position, '.' + str(self.decimal_places_) + 'f')))

            if request_frame:
                self.function(self.data_type, self.name, use_markers=False)

            # return True
        # return False

    def add_info_frame(self, info):
        self.info = info

    def on_enter(self):
        if self.info:
            self.info.configure(text=self.default_data[self.name+'InfoText'])

    def entry_input(self, event):
    # event.char
        self.entry.update()
        try:
            input_num = float(self.entry_string.get())
            self.update_handle(input_num, False)
        except:
            return

    def set(self, value, request_frame=True):
        self.update_handle(float(value), True)

    def set_max(self, value, request_frame=True):
        if value < self.min_:
            value = self.min_

        self.max_ = value
        if self.amount > value:
            self.update_handle(float(value), True)
            return True

        return False

    def get(self):
        return self.amount

    def hide(self):
        if not self.is_resizing:
            self.frame.grid_remove()
            self.visible = False

    def unhide(self):
        if not self.is_resizing:
            self.frame.grid()
            self.visible = True

    # def save_to_file(self, filename, data):
        # with open(filename, 'w') as outfile:
            # json.dump(data, outfile)

    def get_data_type(self):
        return self.data_type

    def load_default(self):
        self.set(self.default_data[self.name+'Amount'])

class Slider3():
    def __init__(self, parent, name, display_text, style_level, function, argument, width, height, row, column, padx, pady, slider_percent):
        self.blank = tk.PhotoImage()

        # Capture inputs as instance variables
        self.parent = parent
        self.name = name
        self.function = function
        self.data_type = argument
        self.slider_percent = slider_percent
        self.width = width
        self.height = height
        self.info = []
        self.row = row
        self.column = column
        self.visible = True
        self.is_resizing = False

        # Initial Value
        self.amount = 0

        if style_level == 1:
            self.frame_style = style.canvas_frame_label_1
            self.text_style = style.text_1
            self.entry_style = style.entry_3

        elif style_level == 3:
            self.frame_style = style.canvas_frame_label_3
            self.text_style = style.text_3
            self.entry_style = style.entry_3

            # UI-controlled variables
        self.entry_string = tk.StringVar()
        self.entry_string.set(self.amount)

        # Widget variables
        self.min_ = -2
        self.max_ = 2
        self.inc_ = 0.001
        self.display_text = display_text + ' '

        # Set up spacing
        # |----------------------|slider_pad|-slider-|entry_pad|-|
        # |---1-slider_percent---|---slider_percent---|
        # |--------------------width------------------|

        # Create a frame to hold it all
        self.frame_width = width
        self.frame_height = height

        self.frame = tk.Frame(self.parent, self.frame_style, width=self.frame_width, height=self.frame_height)
        self.frame.grid(row=row, column=column, sticky='NEWS', padx=padx, pady=pady)
        # self.frame.bind("<Enter>", lambda event: self.on_enter())

        # Add the slider Label text to the frame
        self.txt_label_x = 0
        self.txt_label_y = 0
        self.txt_label_width = int(width * (1.0 - slider_percent))

        self.label = tk.Label(self.frame, self.text_style, image=self.blank, compound='c', text=self.display_text, anchor='e', width=self.txt_label_width, height=self.height)
        self.label.place(x=self.txt_label_x, y=self.txt_label_y)

        # Add the Slider Canvas to the frame
        self.slider_canvas_x = self.txt_label_width
        self.slider_canvas_y = 0
        self.slider_canvas_width = width - self.txt_label_width

        self.slider = tk.Canvas(self.frame, self.frame_style, width=self.slider_canvas_width, height=self.height)
        self.slider.place(x=self.slider_canvas_x, y=self.slider_canvas_y)
        self.slider.bind('<B1-Motion>', lambda e: self.update_handle(e, True))
        self.slider.bind('<MouseWheel>', lambda e: self.update_handle(e, True))

        # Add the Entry to the frame
        self.entry_width = 60
        self.entry_x = self.frame_width - self.entry_width
        self.entry_y = 0

        self.entry = tk.Entry(self.frame, self.entry_style, textvariable=self.entry_string)
        self.entry.place(x=self.entry_x, y=self.entry_y)
        self.entry.bind('<Return>', lambda event: self.entry_input(event))

        # Draw the slider
        self.slider_pad = 20
        self.entry_pad = 20
        self.slider_left = self.slider_pad
        self.slider_right = self.slider_canvas_width - self.entry_pad - self.entry_width
        self.slider_center = (self.height + 1) / 2

        self.oval_loc = self.pos2coord(self.amount)
        self.oval_radius = 5
        self.oval_x1 = self.oval_loc - self.oval_radius
        self.oval_y1 = self.slider_center - self.oval_radius
        self.oval_x2 = self.oval_loc + self.oval_radius
        self.oval_y2 = self.slider_center + self.oval_radius

        self.trough_x1 = self.slider_left
        self.trough_y1 = self.slider_center - 2
        self.trough_x2 = self.slider_right
        self.trough_y2 = self.slider_center + 2

        self.slider.create_rectangle(self.trough_x1, self.trough_y1, self.trough_x2, self.trough_y2, fill='#1F1F1F', outline='')
        self.handle = self.slider.create_oval(self.oval_x1, self.oval_y1, self.oval_x2, self.oval_y2, fill='#919191', outline='')

    def coord2pos(self, coord):
        return float((coord - self.slider_left) * (self.max_ - self.min_) / (self.slider_right - self.slider_left) + self.min_)

    def pos2coord(self, pos):
        return float((float(pos) - self.min_) * (self.slider_right - self.slider_left) / (self.max_ - self.min_) + self.slider_left)

    def update_handle(self, event, also_update_entry=False, request_frame=True):
        if isinstance(event, float):
            position = event

        elif event.type == '38':
            position = self.amount + self.inc_ * int(event.delta / 120.0)

        elif event.type == '6':
            x_coord = float(event.x)
            position = self.coord2pos(x_coord)

        # constrain mousewheel movement
        if position < self.min_:
            position = self.min_
        elif position > self.max_:
            position = self.max_

        # Find closest position increment
        position_inc = round((position - self.min_) / self.inc_)
        position = (position_inc * self.inc_) + self.min_

        # moving sends many events, so only update when the next frame is reached
        if position != self.amount:
            # Move handle to coordinate based on position
            self.slider.move(self.handle, self.pos2coord(position) - self.pos2coord(self.amount), 0)

            # Save for next time
            self.amount = position

            if also_update_entry:
                self.entry_string.set(str(position))

            if request_frame:
                self.function(self.data_type)

            # return True
        # return False

    def add_info_frame(self, info):
        self.info = info

    # def on_enter(self):
    #     if self.info:
    #         self.info.configure(text=self.default_data[self.name + 'InfoText'])

    def entry_input(self, event):
        # event.char
        self.entry.update()
        try:
            input_num = float(self.entry_string.get())
            self.update_handle(input_num, False)
        except:
            return

    def set(self, value, request_frame=True):
        self.update_handle(float(value), True, request_frame)

    def get(self):
        return self.amount

    def hide(self):
        if not self.is_resizing:
            self.frame.grid_remove()
            self.visible = False

    def unhide(self):
        if not self.is_resizing:
            self.frame.grid()
            self.visible = True

        # def save_to_file(self, filename, data):
        # with open(filename, 'w') as outfile:
        # json.dump(data, outfile)

    def get_data_type(self):
        return self.data_type

    def load_default(self):
        self.set(0)

class Text_Entry():
    def __init__(self, parent, name, display_text, style_level, function, data_type, width, height, row, column, padx, pady, text_percent):
        self.blank = tk.PhotoImage()

        self.default_data = DEFAULT_DATA
        # Capture inputs as instance variables
        self.parent = parent
        self.name = name
        self.function = function
        self.data_type = data_type
        self.width = width
        self.height = height
        self.style = []
        self.info = []
        self.row = row
        self.column = column
        self.visible = True
        self.is_resizing = False

        if style_level == 3:
            self.frame_style = style.canvas_frame_label_3
            self.text_style = style.text_3
            self.sel_off_style = style.text_selection_off_3
            self.sel_on_style = style.text_selection_on_3

        if style_level == 2:
            self.frame_style = style.canvas_frame_label_2
            self.text_style = style.text_2
            self.sel_off_style = style.text_selection_off_2
            self.sel_on_style = style.text_selection_on_2

        self.display_text = display_text+' '

        # Initial data
        self.entry_text = tk.StringVar()
        self.entry_text.set(self.default_data[self.name])

        # Frame to hold everything
        self.frame = tk.Frame(self.parent, self.frame_style, width=self.width, height=self.height)
        self.frame.grid(row=row, column=column, sticky='NEWS', padx=padx, pady=pady)
        self.frame.bind("<Enter>", lambda event: self.on_enter())

        self.text_width = int(width*(1.0-text_percent))

        # Create the text on the left
        self.text_label = tk.Label(self.frame, self.text_style, image=self.blank, compound='c', text=self.display_text, anchor='e', width=self.text_width, height=height)
        self.text_label.place(x=0, y=0)

        self.entry = tk.Entry(self.frame, style.entry_2, textvariable=self.entry_text)
        self.entry.place(x=self.text_width+20, y=0, width = self.width-self.text_width-50, height=height)
        self.entry.bind("<Return>", lambda event: self.send_text(self.entry_text.get()))

    def send_text(self, text):
        self.function(self.data_type, self.name, use_markers=False)

    def add_info_frame(self, info):
        self.info = info

    def on_enter(self):
        if self.info:
            self.info.configure(text=self.default_data[self.name+'InfoText'])

    def get(self):
        return self.entry_text.get()

    def set(self, value, request_frame=True):
        pass
        # self.select_ui_text_selection(value, request_frame)

    def hide(self):
        if not self.is_resizing:
            self.frame.grid_remove()
            self.visible = False

    def unhide(self):
        if not self.is_resizing:
            self.frame.grid()
            self.visible = True

    def get_data_type(self):
        return self.data_type

    def load_default(self):
        pass

class VRAM_Indicator:
    def __init__(self, parent, style_level, width, height, x, y):
        self.parent = parent
        self.width = width
        self.height = height
        self.x = x
        self.y = y
        self.blank = tk.PhotoImage()  # Immagine vuota di placeholder

        self.used = 0
        self.total = 1  # Per evitare divisione per zero
        self.is_resizing = False

        # Imposta gli stili basati su `style_level`
        self._set_styles(style_level)

        # Frame principale
        self.frame = tk.Frame(self.parent, self.frame_style, width=self.width, height=self.height)
        self.frame.place(x=self.x, y=self.y)

        # Label del nome VRAM
        self.label_name = tk.Label(
            self.frame, self.frame_style, image=self.blank, compound='c', fg='#b1b1b2',
            font=("Segoe UI", 9), width=50, text='VRAM', height=self.height
        )
        self.label_name.place(x=0, y=0)

        # Canvas per la barra di indicazione VRAM
        self.canvas = tk.Canvas(
            self.frame, self.frame_style, highlightthickness=2, highlightbackground='#b1b1b2',
            width=self.width - 60, height=self.height - 4
        )
        self.canvas.place(x=50, y=0)

    def _set_styles(self, style_level):
        """Imposta gli stili in base al livello di stile fornito."""
        style_map = {
            3: (style.canvas_frame_label_3, style.text_3, style.text_selection_off_3, style.text_selection_on_3),
            2: (style.canvas_frame_label_2, style.text_2, style.text_selection_off_2, style.text_selection_on_2),
            1: (style.canvas_frame_label_1, None, None, None)
        }
        self.frame_style, self.text_style, self.sel_off_style, self.sel_on_style = style_map.get(style_level, (None, None, None, None))

    def update_display(self):
        """Aggiorna il display dell'indicatore VRAM."""
        # Controlla se il canvas esiste ancora
        if not self.canvas.winfo_exists():
            return  # Esci se il canvas non esiste più

        self.canvas.delete('all')  # Ora è sicuro eliminare tutti gli oggetti

        width = self.canvas.winfo_width()

        # Calcolo del rapporto usato/total
        try:
            ratio = self.used / self.total
        except ZeroDivisionError:
            ratio = 1

        # Colore della barra in base all'uso della VRAM
        color = '#d10303' if ratio > 0.9 else '#b1b1b2'
        filled_width = ratio * width

        # Crea il rettangolo indicatore
        self.canvas.create_rectangle(0, 0, filled_width, self.height, fill=color)

    def set(self, used, total):
        """Imposta i valori di VRAM usata e totale, e aggiorna il display."""
        self.used = used
        self.total = total
        self.update_display()

    def hide(self):
        """Nascondi il widget dell'indicatore VRAM."""
        if not self.is_resizing:
            self.frame.place_forget()

    def unhide(self):
        """Mostra il widget dell'indicatore VRAM."""
        if not self.is_resizing:
            self.frame.place(x=self.x, y=self.y)

