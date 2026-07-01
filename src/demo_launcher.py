# src/demo_launcher.py
# FlyPose-SAR — Launcher interattivo con GUI
#
# USO:
#   cd C:\Temp\FlyPose
#   .\venv\Scripts\python.exe src/demo_launcher.py

import tkinter as tk
from tkinter import messagebox, ttk
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIGURAZIONE PATH
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path.cwd()

MODELS = {
    'rgb': {
        'label': 'RGB — Large SAR (Fase 1)',
        'path' : _PROJECT_ROOT / 'runs/fase1/fase1_large/weights/best.pt',
    },
    'thermal': {
        'label': 'Thermal Multi-Palette (Fase 2b)',
        'path' : _PROJECT_ROOT / 'runs/fase2/flypose_multipalette_large/weights/best.pt',
    },
}

GAN_WEIGHTS    = _PROJECT_ROOT / 'runs/fase2/cyclegan_run/G_AB_final.pth'
SEQUENCES_DIR  = _PROJECT_ROOT / 'datasets/dataset_test_official/sequences'
THERMAL_DIR    = _PROJECT_ROOT / 'datasets/dataset_sar_thermal_multipalette'

PALETTES = {
    'White Hot' : 'white_hot',
    'Black Hot' : 'black_hot',
    'Iron Red'  : 'iron_red',
    'Rainbow 1' : 'rainbow1',
    'Hot Iron'  : 'hot_iron',
}

# ---------------------------------------------------------------------------
# COLORI
# ---------------------------------------------------------------------------
BG        = '#0d1117'
CARD      = '#161b22'
CARD2     = '#1c2128'
BORDER    = '#30363d'
BLUE      = '#2196F3'
BLUE_DK   = '#1565C0'
GREEN     = '#4CAF50'
GREEN_DK  = '#2E7D32'
ORANGE    = '#FF9800'
WHITE     = '#e6edf3'
GRAY      = '#8b949e'
LGRAY     = '#30363d'
SEL_BG    = '#1f6feb'
ACCENT    = '#00B4D8'

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def make_card(parent, **kwargs):
    f = tk.Frame(parent, bg=CARD2,
                 highlightbackground=BORDER, highlightthickness=1,
                 **kwargs)
    return f

def section_label(parent, text, pady=(12,4)):
    tk.Label(parent, text=text,
             font=('Segoe UI', 9, 'bold'),
             fg=ACCENT, bg=BG).pack(anchor='w', pady=pady)

# ---------------------------------------------------------------------------
# APP
# ---------------------------------------------------------------------------
class FlyPoseLauncher(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('FlyPose-SAR')
        self.configure(bg=BG)
        self.resizable(True, True)
        self.minsize(720, 560)

        # Stato
        self.mode_var        = tk.StringVar(value='rgb')
        self.palette_var     = tk.StringVar(value='iron_red')
        self.conf_var        = tk.DoubleVar(value=0.25)
        self.sidebyside_var  = tk.BooleanVar(value=True)
        self.use_preconv_var = tk.BooleanVar(value=True)

        self._build_ui()
        self._refresh_sequences()
        self._on_mode_change()

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Topbar ──────────────────────────────────────────────────────────
        topbar = tk.Frame(self, bg=CARD, height=52)
        topbar.pack(fill='x')
        topbar.pack_propagate(False)

        tk.Label(topbar, text='FlyPose', font=('Segoe UI', 18, 'bold'),
                 fg=WHITE, bg=CARD).pack(side='left', padx=(18,0), pady=10)
        tk.Label(topbar, text='-SAR', font=('Segoe UI', 18, 'bold'),
                 fg=BLUE, bg=CARD).pack(side='left', pady=10)
        tk.Label(topbar, text='  Demo Launcher',
                 font=('Segoe UI', 10), fg=GRAY, bg=CARD).pack(side='left', pady=14)

        # Badge stato modelli
        self.badge_rgb = tk.Label(topbar, text='RGB ✓', font=('Segoe UI', 8, 'bold'),
                                   fg=GREEN, bg=CARD, padx=6)
        self.badge_rgb.pack(side='right', padx=(0,8), pady=16)
        self.badge_th = tk.Label(topbar, text='Thermal ✓', font=('Segoe UI', 8, 'bold'),
                                  fg=GREEN, bg=CARD, padx=6)
        self.badge_th.pack(side='right', padx=(0,4), pady=16)

        # ── Body ────────────────────────────────────────────────────────────
        body = tk.Frame(self, bg=BG)
        body.pack(fill='both', expand=True, padx=14, pady=10)
        body.columnconfigure(0, weight=0, minsize=220)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        # ── Pannello sinistro ────────────────────────────────────────────────
        left = tk.Frame(body, bg=BG)
        left.grid(row=0, column=0, sticky='nsew', padx=(0,10))

        # Sezione 1 — Modalità
        section_label(left, '① MODALITÀ')
        card_mode = make_card(left)
        card_mode.pack(fill='x', pady=(0,6))
        for text, value in [('RGB  —  Fase 1 zenitale', 'rgb'),
                             ('Thermal  —  Fase 2b multi-palette', 'thermal')]:
            row = tk.Frame(card_mode, bg=CARD2)
            row.pack(fill='x', padx=8, pady=4)
            rb = tk.Radiobutton(
                row, text=text, variable=self.mode_var, value=value,
                command=self._on_mode_change,
                fg=WHITE, bg=CARD2, selectcolor=SEL_BG,
                activeforeground=WHITE, activebackground=CARD2,
                font=('Segoe UI', 9), indicatoron=True,
                relief='flat', bd=0,
            )
            rb.pack(anchor='w')

        # Sezione 2 — Palette
        section_label(left, '② PALETTE TERMICA')
        self.card_palette = make_card(left)
        self.card_palette.pack(fill='x', pady=(0,6))
        self.palette_widgets = []
        palette_colors = {
            'White Hot': '#f0f0f0', 'Black Hot': '#555555',
            'Iron Red': '#c0392b', 'Rainbow 1': '#2980b9', 'Hot Iron': '#e67e22',
        }
        for label, value in PALETTES.items():
            row = tk.Frame(self.card_palette, bg=CARD2)
            row.pack(fill='x', padx=8, pady=2)
            dot = tk.Label(row, text='●', font=('Segoe UI', 8),
                           fg=palette_colors.get(label, BLUE), bg=CARD2)
            dot.pack(side='left', padx=(0,4))
            rb = tk.Radiobutton(
                row, text=label, variable=self.palette_var, value=value,
                command=self._on_palette_change,
                fg=WHITE, bg=CARD2, selectcolor=SEL_BG,
                activeforeground=WHITE, activebackground=CARD2,
                font=('Segoe UI', 9), relief='flat', bd=0,
            )
            rb.pack(side='left', anchor='w')
            self.palette_widgets.append((dot, rb))

        # Opzioni thermal
        self.card_opts = make_card(left)
        self.card_opts.pack(fill='x', pady=(0,6))
        self.preconv_cb = tk.Checkbutton(
            self.card_opts,
            text='⚡ Usa frame pre-convertiti',
            variable=self.use_preconv_var,
            command=self._on_option_change,
            fg=GREEN, bg=CARD2, selectcolor=LGRAY,
            activeforeground=WHITE, activebackground=CARD2,
            font=('Segoe UI', 9), relief='flat',
        )
        self.preconv_cb.pack(anchor='w', padx=8, pady=(6,2))
        self.sidebyside_cb = tk.Checkbutton(
            self.card_opts,
            text='◫ Vista side-by-side (RGB | Thermal)',
            variable=self.sidebyside_var,
            fg=GRAY, bg=CARD2, selectcolor=LGRAY,
            activeforeground=WHITE, activebackground=CARD2,
            font=('Segoe UI', 9), relief='flat',
        )
        self.sidebyside_cb.pack(anchor='w', padx=8, pady=(0,6))

        # Sezione 4 — Confidenza
        section_label(left, '④ SOGLIA CONFIDENZA')
        card_conf = make_card(left)
        card_conf.pack(fill='x')
        conf_row = tk.Frame(card_conf, bg=CARD2)
        conf_row.pack(fill='x', padx=8, pady=8)
        self.conf_label = tk.Label(conf_row, text='0.25', width=4,
                                    fg=BLUE, bg=CARD2,
                                    font=('Segoe UI', 11, 'bold'))
        self.conf_label.pack(side='right')
        self.conf_slider = tk.Scale(
            conf_row, from_=0.05, to=0.90, resolution=0.05,
            orient='horizontal', variable=self.conf_var,
            bg=CARD2, fg=WHITE, troughcolor=LGRAY,
            highlightthickness=0, sliderrelief='flat',
            command=lambda v: self.conf_label.config(text=f'{float(v):.2f}'),
            showvalue=False, length=160,
        )
        self.conf_slider.pack(side='left', fill='x', expand=True)

        # ── Pannello destro ──────────────────────────────────────────────────
        right = tk.Frame(body, bg=BG)
        right.grid(row=0, column=1, sticky='nsew')
        right.rowconfigure(1, weight=1)

        section_label(right, '③ SEQUENZA DI TEST')

        # Badge disponibilità
        self.avail_label = tk.Label(right, text='', fg=GRAY, bg=BG,
                                     font=('Segoe UI', 8))
        self.avail_label.pack(anchor='w', pady=(0,4))

        # Listbox con scrollbar
        list_frame = make_card(right)
        list_frame.pack(fill='both', expand=True, pady=(0,6))

        scrollbar = tk.Scrollbar(list_frame, bg=CARD2, troughcolor=BG,
                                  relief='flat', bd=0)
        scrollbar.pack(side='right', fill='y', pady=2, padx=(0,2))

        self.seq_listbox = tk.Listbox(
            list_frame, yscrollcommand=scrollbar.set,
            bg=CARD2, fg=WHITE,
            selectbackground=SEL_BG, selectforeground=WHITE,
            font=('Consolas', 9),
            borderwidth=0, highlightthickness=0,
            activestyle='none',
        )
        self.seq_listbox.pack(side='left', fill='both', expand=True,
                               padx=(6,0), pady=4)
        scrollbar.config(command=self.seq_listbox.yview)
        self.seq_listbox.bind('<<ListboxSelect>>', self._on_seq_select)

        tk.Button(
            right, text='↻  Aggiorna',
            command=self._refresh_sequences,
            bg=CARD2, fg=GRAY, relief='flat',
            activebackground=LGRAY, activeforeground=WHITE,
            font=('Segoe UI', 8), cursor='hand2', bd=0,
        ).pack(anchor='e', pady=(0,4))

        # ── Bottom bar ───────────────────────────────────────────────────────
        bottom = tk.Frame(self, bg=CARD, height=64)
        bottom.pack(fill='x', side='bottom')
        bottom.pack_propagate(False)

        self.info_label = tk.Label(bottom, text='', fg=GRAY, bg=CARD,
                                    font=('Segoe UI', 8))
        self.info_label.pack(side='left', padx=14, pady=20)

        self.run_btn = tk.Button(
            bottom, text='▶   AVVIA DEMO',
            command=self._run_demo,
            bg=GREEN, fg='white', relief='flat',
            font=('Segoe UI', 11, 'bold'),
            activebackground=GREEN_DK, activeforeground='white',
            cursor='hand2', padx=28,
        )
        self.run_btn.pack(side='right', padx=14, pady=10, ipady=4)

    # ── LOGICA ────────────────────────────────────────────────────────────────
    def _on_mode_change(self):
        is_thermal = self.mode_var.get() == 'thermal'
        pal_state = 'normal' if is_thermal else 'disabled'
        for dot, rb in self.palette_widgets:
            rb.config(state=pal_state)
        self.preconv_cb.config(state=pal_state)
        self.sidebyside_cb.config(state='normal' if is_thermal else 'disabled')

        # Badge modelli
        for key, badge in [('rgb', self.badge_rgb), ('thermal', self.badge_th)]:
            exists = MODELS[key]['path'].exists()
            badge.config(
                text=f"{'RGB' if key=='rgb' else 'Thermal'} {'✓' if exists else '✗'}",
                fg=GREEN if exists else ORANGE,
            )

        mode = self.mode_var.get()
        model_path = MODELS[mode]['path']
        exists = model_path.exists()
        self.info_label.config(
            text=f"{'●' if exists else '○'}  {model_path.name}  {'trovato' if exists else '— NON TROVATO'}",
            fg=GREEN if exists else ORANGE,
        )
        self._refresh_sequences()

    def _on_palette_change(self):
        self._refresh_sequences()

    def _on_option_change(self):
        self._refresh_sequences()

    def _preconv_available(self, seq_name):
        palette = self.palette_var.get()
        d = THERMAL_DIR / palette / 'images' / 'test' / seq_name
        return d.exists() and any(d.glob('*.jpg'))

    def _refresh_sequences(self):
        self.seq_listbox.delete(0, 'end')
        if not SEQUENCES_DIR.exists():
            self.seq_listbox.insert('end', '  [cartella non trovata]')
            return
        sequences = sorted([d.name for d in SEQUENCES_DIR.iterdir() if d.is_dir()])
        if not sequences:
            self.seq_listbox.insert('end', '  [nessuna sequenza]')
            return

        is_thermal  = self.mode_var.get() == 'thermal'
        use_preconv = self.use_preconv_var.get() and is_thermal
        n_avail = 0

        for s in sequences:
            if use_preconv:
                avail = self._preconv_available(s)
                marker = '✓' if avail else '○'
                color  = WHITE if avail else GRAY
                if avail: n_avail += 1
            else:
                marker = ' '
                color  = WHITE
            self.seq_listbox.insert('end', f'  {marker}  {s}')
            self.seq_listbox.itemconfig(self.seq_listbox.size()-1, fg=color)

        self.seq_listbox.selection_set(0)

        if is_thermal and use_preconv:
            self.avail_label.config(
                text=f'✓ pre-convertiti: {n_avail}/{len(sequences)}  ·  ○ conversione GAN live',
                fg=GRAY,
            )
        else:
            self.avail_label.config(text='', fg=GRAY)

    def _on_seq_select(self, event):
        pass

    def _get_selected_sequence(self):
        sel = self.seq_listbox.curselection()
        if not sel:
            return None
        raw = self.seq_listbox.get(sel[0])
        # rimuovi marker e spazi
        return raw.strip().lstrip('✓○ ').strip()

    def _run_demo(self):
        mode = self.mode_var.get()
        seq  = self._get_selected_sequence()

        if not seq or seq.startswith('['):
            messagebox.showwarning('Attenzione', 'Seleziona una sequenza.')
            return

        model_path = MODELS[mode]['path']
        if not model_path.exists():
            messagebox.showerror('Errore', f'Modello non trovato:\n{model_path}')
            return

        conf    = self.conf_var.get()
        palette = self.palette_var.get()
        python  = sys.executable
        script  = str(_PROJECT_ROOT / 'src/demo_realtime.py')

        if mode == 'rgb':
            seq_path = SEQUENCES_DIR / seq
            if not seq_path.exists():
                messagebox.showerror('Errore', f'Sequenza non trovata:\n{seq_path}')
                return
            cmd = [python, script,
                   '--source', str(seq_path),
                   '--model',  str(model_path),
                   '--conf',   str(conf)]

        else:
            use_preconv   = self.use_preconv_var.get()
            show_sbs      = self.sidebyside_var.get()
            preconv_path  = THERMAL_DIR / palette / 'images' / 'test' / seq
            rgb_seq_path  = SEQUENCES_DIR / seq

            if use_preconv and preconv_path.exists() and any(preconv_path.glob('*.jpg')):
                # Frame pre-convertiti disponibili
                if show_sbs and rgb_seq_path.exists():
                    # Side-by-side: passa entrambi i path separati da "|"
                    # demo_realtime gestirà i due flussi in parallelo
                    cmd = [python, script,
                           '--source',      str(preconv_path),
                           '--source-rgb',  str(rgb_seq_path),
                           '--model',       str(model_path),
                           '--conf',        str(conf),
                           '--preconv-thermal',
                           '--palette',     palette]
                else:
                    # Solo termico, nessun side-by-side
                    cmd = [python, script,
                           '--source', str(preconv_path),
                           '--model',  str(model_path),
                           '--conf',   str(conf),
                           '--no-sidebyside']
            else:
                # Fallback: conversione GAN live
                if not GAN_WEIGHTS.exists():
                    messagebox.showerror('Errore', f'GAN weights non trovati:\n{GAN_WEIGHTS}')
                    return
                cmd = [python, script,
                       '--source',      str(rgb_seq_path),
                       '--model',       str(model_path),
                       '--conf',        str(conf),
                       '--thermal',
                       '--palette',     palette,
                       '--gan-weights', str(GAN_WEIGHTS)]
                if not show_sbs:
                    cmd += ['--no-sidebyside']

        print(f'[Launcher] {mode.upper()}'
              + (f' | palette={palette}' if mode=='thermal' else '')
              + f' | seq={seq}')

        self.run_btn.config(text='⏳  Avvio...', state='disabled', bg='#444')
        self.after(300, lambda: self._launch(cmd))

    def _launch(self, cmd):
        subprocess.Popen(cmd)
        self.run_btn.config(text='▶   AVVIA DEMO', state='normal', bg=GREEN)


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    app = FlyPoseLauncher()
    app.mainloop()