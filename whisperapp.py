import tkinter as tk
from ttkbootstrap.constants import *
import ttkbootstrap as ttk
import soundcard as sc
import whisper
import torch
from threading import Thread, Event
from transcriber import start_listen


def prepare():
    # load silero VAD
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=False)
    (get_speech_ts, _, _, _, _) = utils

    whisper_model = whisper.load_model("tiny.en")

    return model, get_speech_ts, whisper_model


class WhisperPy(ttk.Window):
    def __init__(self, title, header_text, size, theme="darkly"):
        super().__init__(title=title, size=size, themename=theme)

        self.model, self.get_speech_ts, self.whisper_model = prepare()

        self.create_header(header_text=header_text)

        self.option_val = tk.StringVar()
        self.progress_val = tk.DoubleVar()
        self.progress_label_val = tk.StringVar()
        self.progress = 0
        self.processing = False

        self.mainFrame = self.create_main()
        self.options = self.create_option()
        self.progressLabel, self.progressbar = self.create_progressbar()
        self.start_button, self.stop_button = self.create_buttons()
        self.start_button["state"] = "normal"
        self.stop_button["state"] = "disabled"
        self.textbox = self.create_textbox()

        self.progress_label_val.set('Ready')

        self.mainloop()

    def callback(self, text, translation):
        self.textbox.insert(END, f'>> {text}\n>> {translation}\n')
        self.textbox.yview(END)

    def start_click(self):
        if self.processing is False:
            self.processing = True
            self.update_progress()
        self.event = Event()
        self.task = Thread(target=start_listen,
                           args=(self.whisper_model,
                                 self.model,
                                 self.get_speech_ts,
                                 self.mics[self.options.current()],
                                 self.event,
                                 self.callback))
        self.task.daemon = True
        self.task.start()
        self.start_button["state"] = "disabled"
        self.stop_button["state"] = "normal"
        self.progress_label_val.set('Listening...')

    def stop_click(self):
        self.processing = False
        self.event.set()
        self.start_button["state"] = "normal"
        self.stop_button["state"] = "disabled"
        self.progress_label_val.set('Ready')

    def update_progress(self):
        self.progress += 1
        if self.progress > 100:
            self.progress = 0
        self.progress_val.set(self.progress)
        if self.processing:
            self.after(100, self.update_progress)
        else:
            self.progress = 0
            self.progress_val.set(self.progress)

    def create_header(self, header_text):
        titleFrame = ttk.Frame(self, height=100)

        label = ttk.Label(titleFrame, text=header_text, font=(
            'TkDefaultFixed', 20), anchor='center', bootstyle="inverse-primary")
        label.pack(fill='x')
        titleFrame.pack(fill='x')

    def create_main(self):
        mainFrame = ttk.Frame(self)
        mainFrame.pack(expand=True, fill='both', padx=4, pady=4)
        return mainFrame

    def getDevices(self):
        self.mics = sc.all_microphones(include_loopback=True)
        devices = []
        for i in range(len(self.mics)):
            try:
                devices.append(self.mics[i].name)
            except Exception as e:
                print(e)
        return devices

    def create_option(self):
        optionFrame = ttk.Frame(self.mainFrame)
        optionFrame.pack(fill='x')

        optionLabel = ttk.Label(optionFrame, text='Select Device')
        optionLabel.pack(side=LEFT, padx=(8, 0), pady=(4, 0))
        # create the combobox in a readonly state
        options = ttk.Combobox(optionFrame, state="readonly",
                               textvariable=self.option_val)
        options['values'] = self.getDevices()
        options.current(0)
        options.pack(side=LEFT, expand=True, fill='x',
                     padx=(8, 8), pady=(4, 0))
        return options

    def create_progressbar(self):
        labelFrame = ttk.Frame(self.mainFrame)
        labelFrame.pack(fill='x')
        progressFrame = ttk.Frame(self.mainFrame)
        progressFrame.pack(fill='x')
        progress_label = ttk.Label(
            labelFrame, textvariable=self.progress_label_val)
        progress_label.pack(side=LEFT, fill='x', padx=(8, 8), pady=(4, 0))
        progressbar = ttk.Progressbar(
            progressFrame, variable=self.progress_val, maximum=100, bootstyle='info-striped')
        progressbar.pack(side=LEFT, expand=True, fill='x',
                         padx=(8, 8), pady=(4, 0))
        return progress_label, progressbar

    def create_buttons(self):
        buttonFrame = ttk.Frame(self.mainFrame)
        buttonFrame.columnconfigure((0, 1, 2, 3), weight=1, uniform='a')
        buttonFrame.pack(fill='x')

        button1 = ttk.Button(buttonFrame, text='Start',
                             command=self.start_click)
        button1.grid(row=0, column=1, pady=4)

        button2 = ttk.Button(buttonFrame, text='Stop',
                             command=self.stop_click)
        button2.grid(row=0, column=2, pady=4)

        return button1, button2

    def create_textbox(self):
        textbox = ttk.ScrolledText(self.mainFrame, font=(
            'TkDefaultFixed', 12))
        textbox.config(spacing1=10, spacing2=10, spacing3=10)
        textbox.pack(expand=True, fill='both', padx=8, pady=(4, 8))
        return textbox


WhisperPy(title='Whisper Python', size=(800, 600),
          header_text='Whisper Translator', theme='vapor')
