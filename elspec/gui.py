"""
The Qt interface: a matplotlib plot of the data with the current ion's model
(red) and the combined model of the saved ions (blue), plus controls for the
global and per-ion parameters and the saved ion set.
"""
import os.path
import sys
import time
from dataclasses import asdict

import matplotlib

matplotlib.use("QtAgg")
# A user matplotlibrc may enable LaTeX, which is slow and fails without a full
# TeX installation; plain text labels are all that is needed here.
matplotlib.rcParams["text.usetex"] = False
from matplotlib.backends.qt_compat import QtGui, QtWidgets, QtCore
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure

from .constants import (
    LOGA_MIN, LOGA_MAX, LOGA_INIT, LOGA_STEP, LIVE_UPDATE_MAX_TIME,
)
from .linelist import available_ions, ModelError
from .model import IonModel, GlobalParams, combine_models
from .params import GLOBAL_PARAMS, ION_PARAMS, WAVE_OPTIONS
from .session import write_session

Qt = QtCore.Qt
PANEL_WIDTH = 290


def make_spin(spec, value):
    """
    Spin box for a ParamSpec. Updates only on Enter/arrows, not per keystroke.
    """
    if spec.is_int:
        spin = QtWidgets.QSpinBox()
    else:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setDecimals(spec.decimals)
    spin.setRange(spec.minimum, spec.maximum)
    spin.setSingleStep(spec.step)
    spin.setValue(value)
    spin.setKeyboardTracking(False)
    spin.setToolTip(spec.tooltip)
    return spin


class blocked_signals:
    """
    Context manager to change widgets without triggering their callbacks
    """
    def __init__(self, *widgets):
        self.widgets = widgets

    def __enter__(self):
        for widget in self.widgets:
            widget.blockSignals(True)

    def __exit__(self, *exc):
        for widget in self.widgets:
            widget.blockSignals(False)


class AbundanceControl(QtWidgets.QWidget):
    """
    log10 abundance slider with a linked spin box for exact values.
    on_change(logA) is called when the user changes either.
    """
    def __init__(self, logA, on_change):
        super().__init__()
        self.on_change = on_change
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QtWidgets.QLabel("log₁₀(A)"))

        self.slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, round((LOGA_MAX - LOGA_MIN) / LOGA_STEP))
        layout.addWidget(self.slider, stretch=1)

        self.spin = QtWidgets.QDoubleSpinBox()
        self.spin.setRange(LOGA_MIN, LOGA_MAX)
        self.spin.setSingleStep(LOGA_STEP)
        self.spin.setDecimals(2)
        self.spin.setKeyboardTracking(False)
        layout.addWidget(self.spin)

        self.set_value(logA)
        self.slider.valueChanged.connect(
            lambda i: self._changed(LOGA_MIN + i * LOGA_STEP))
        self.spin.valueChanged.connect(self._changed)

    def value(self):
        return self.spin.value()

    def set_value(self, logA):
        """
        Set both widgets, without calling on_change
        """
        with blocked_signals(self.slider, self.spin):
            self.spin.setValue(logA)
            self.slider.setValue(round((logA - LOGA_MIN) / LOGA_STEP))

    def set_live(self, live):
        """
        If not live, the slider only reports changes when released
        """
        self.slider.setTracking(live)

    def _changed(self, logA):
        self.set_value(logA)
        self.on_change(self.value())


class ElementSpecWindow(QtWidgets.QMainWindow):
    """
    Main window. S is the (joined) data spectrum, initial the starting
    parameter values, saved the list of saved IonModels, and session_file
    where they are stored.
    """
    def __init__(self, S, has_data, initial, saved, session_file):
        super().__init__()
        self.setWindowTitle("element_spec")
        self.S = S
        self.has_data = has_data
        self.saved = saved
        self.session_file = session_file
        self.initial = initial

        self.current = self._initial_ion(initial)  # The ion being edited (red)
        self.M = None  # Its model

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)
        layout.addLayout(self._build_plot_area(), stretch=1)
        controls = QtWidgets.QVBoxLayout()
        layout.addLayout(controls)
        controls.addWidget(self._build_global_panel())
        controls.addWidget(self._build_ion_panel())
        controls.addWidget(self._build_saved_panel(), stretch=1)
        self.status = self.statusBar()

        self._fill_saved_list()
        self.update_models()

    def _initial_ion(self, initial):
        """
        The starting ion: its saved version if there is one, otherwise
        defaults, with any per-ion values given explicitly taking precedence.
        """
        key = (initial["ion"], initial["emission"])
        saved = [im for im in self.saved if im.key == key]
        if saved:
            im = saved[0].copy()
        else:
            defaults = {spec.name: spec.default for spec in ION_PARAMS}
            im = IonModel(*key, LOGA_INIT, **defaults)
        for name in ["logA"] + [spec.name for spec in ION_PARAMS]:
            if initial[name] is not None:
                setattr(im, name, initial[name])
        return im

    # .........................................................................
    # Layout

    def _build_plot_area(self):
        area = QtWidgets.QVBoxLayout()
        self.fig = Figure(figsize=(10, 6), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.fig)
        area.addWidget(NavigationToolbar2QT(self.canvas, self))
        area.addWidget(self.canvas, stretch=1)
        self.abundance = AbundanceControl(self.current.logA, self._logA_changed)
        area.addWidget(self.abundance)

        self.ax = self.fig.add_subplot()
        if self.has_data:
            self.ax.plot(self.S.x, self.S.y, c="grey", drawstyle="steps-mid",
                         zorder=1)
        (self.combined_line,) = self.ax.plot([], [], "C0-", zorder=2)
        (self.model_line,) = self.ax.plot([], [], "r-", zorder=3)
        self.ax.set_xlabel(r"Wavelength [Å]")
        self.ax.set_ylabel(r"Normalised flux")
        self.ax.set_xlim(*self.S.x01)
        self.ax.set_ylim(0, 1.2 * self.initial["scale"])
        return area

    def _build_global_panel(self):
        box = QtWidgets.QGroupBox("All ions")
        box.setFixedWidth(PANEL_WIDTH)
        form = QtWidgets.QFormLayout(box)

        self.wave_box = QtWidgets.QComboBox()
        for text, wave in WAVE_OPTIONS:
            self.wave_box.addItem(text, wave)
        self.wave_box.setCurrentIndex(self.wave_box.findData(self.initial["wave"]))
        self.wave_box.setToolTip(
            "Wavelength convention of the models (should match the data)")
        self.wave_box.currentIndexChanged.connect(self._global_changed)
        form.addRow("Wavelengths", self.wave_box)

        self.global_spins = {}
        for spec in GLOBAL_PARAMS:
            spin = make_spin(spec, self.initial[spec.name])
            spin.valueChanged.connect(self._global_changed)
            self.global_spins[spec.name] = spin
            form.addRow(spec.label, spin)
        return box

    def _build_ion_panel(self):
        box = QtWidgets.QGroupBox("Current ion (red)")
        box.setFixedWidth(PANEL_WIDTH)
        form = QtWidgets.QFormLayout(box)

        self.ion_box = QtWidgets.QComboBox()
        self.ion_box.setEditable(True)
        self.ion_box.addItems(available_ions())
        self.ion_box.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
        completer = self.ion_box.completer()
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setCompletionMode(
            QtWidgets.QCompleter.CompletionMode.PopupCompletion)
        self.ion_box.setCurrentText(self.current.ion)
        self.ion_box.lineEdit().editingFinished.connect(self._ion_changed)
        self.ion_box.activated.connect(self._ion_changed)
        form.addRow("Ion", self.ion_box)

        self.emission_box = QtWidgets.QCheckBox("Emission lines")
        self.emission_box.setChecked(self.current.emission)
        self.emission_box.toggled.connect(self._emission_changed)
        form.addRow(self.emission_box)

        self.ion_spins = {}
        for spec in ION_PARAMS:
            spin = make_spin(spec, getattr(self.current, spec.name))
            spin.valueChanged.connect(
                lambda value, name=spec.name: self._ion_param_changed(name, value))
            self.ion_spins[spec.name] = spin
            form.addRow(spec.label, spin)

        self.show_current = QtWidgets.QCheckBox("Show current ion")
        self.show_current.setChecked(True)
        self.show_current.toggled.connect(lambda _: self.update_models())
        form.addRow(self.show_current)

        self.save_button = QtWidgets.QPushButton("Save to ion set")
        self.save_button.setToolTip(
            "Add the current ion to the saved ions (or update it if already saved)")
        self.save_button.clicked.connect(self.save_current)
        form.addRow(self.save_button)
        return box

    def _build_saved_panel(self):
        box = QtWidgets.QGroupBox("Saved ions (blue)")
        box.setFixedWidth(PANEL_WIDTH)
        layout = QtWidgets.QVBoxLayout(box)

        self.saved_list = QtWidgets.QListWidget()
        self.saved_list.setToolTip(
            "Tick to include in the blue model. Double-click to edit.")
        self.saved_list.itemChanged.connect(self._saved_item_changed)
        self.saved_list.itemDoubleClicked.connect(
            lambda item: self._load_into_current(self._saved_from_item(item)))
        layout.addWidget(self.saved_list)

        buttons = QtWidgets.QGridLayout()
        layout.addLayout(buttons)
        for i, (text, slot) in enumerate([
            ("All", lambda: self._set_all_saved(True)),
            ("None", lambda: self._set_all_saved(False)),
            ("Edit", self._edit_selected),
            ("Remove", self._remove_selected),
        ]):
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(slot)
            buttons.addWidget(button, i // 2, i % 2)
        return box

    # .........................................................................
    # Parameters

    def global_params(self):
        """
        Returns (GlobalParams, display scale)
        """
        values = {name: spin.value() for name, spin in self.global_spins.items()}
        scale = values.pop("scale")
        return GlobalParams(wave=self.wave_box.currentData(), **values), scale

    def _logA_changed(self, logA):
        self.current.logA = logA
        self.update_models()

    def _global_changed(self, *_):
        _, scale = self.global_params()
        self.ax.set_ylim(0, 1.2 * scale)
        self.update_models()

    def _ion_param_changed(self, name, value):
        setattr(self.current, name, value)
        self.update_models()

    def _ion_changed(self, *_):
        ion = self.ion_box.currentText().strip()
        if ion == self.current.ion:
            return
        if ion not in available_ions():
            self.status.showMessage(f"No atomic data for '{ion}'")
            self.ion_box.setCurrentText(self.current.ion)
            return
        self._switch_current(ion, self.current.emission)

    def _emission_changed(self, emission):
        self._switch_current(self.current.ion, emission)

    def _switch_current(self, ion, emission):
        """
        Change the current ion. If it is already saved, its saved parameters
        are loaded; otherwise the per-ion parameters carry over.
        """
        for im in self.saved:
            if im.key == (ion, emission):
                self._load_into_current(im)
                return
        self.current.ion = ion
        self.current.emission = emission
        self._fill_saved_list()
        self.update_models()

    def _load_into_current(self, im):
        """
        Make a saved ion the current one (a copy, so the saved version only
        changes when saved again)
        """
        self.current = im.copy()
        with blocked_signals(self.ion_box, self.emission_box,
                             *self.ion_spins.values()):
            self.ion_box.setCurrentText(im.ion)
            self.emission_box.setChecked(im.emission)
            for name, spin in self.ion_spins.items():
                spin.setValue(getattr(im, name))
        self.abundance.set_value(im.logA)
        self._fill_saved_list()
        self.update_models()

    # .........................................................................
    # Models

    def update_models(self):
        """
        Recompute (where needed) and redraw the current and saved models
        """
        QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        t0 = time.perf_counter()
        try:
            g, scale = self.global_params()
            messages = [self._update_current(g, scale)]
            messages += self._update_combined(g, scale)
            messages.append(f"{time.perf_counter() - t0:.2f} s")
            self.status.showMessage(" | ".join(messages))
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.canvas.draw_idle()

    def _update_current(self, g, scale):
        """
        Update the red model. Returns a status message.
        """
        try:
            self.M = self.current.model(self.S.x01, g)
            message = (f"{self.current.label}: {self.current.n_lines} lines, "
                       f"{len(self.M)} px")
            # Slow models only update when the slider is released
            self.abundance.set_live(
                self.current.last_model_time < LIVE_UPDATE_MAX_TIME)
        except ModelError as err:
            self.M = None
            message = str(err)
        if self.M is not None:
            y = self.M.y * scale
            self.model_line.set_data(self.M.x, 1 + y if self.current.emission else y)
        self.model_line.set_visible(
            self.M is not None and self.show_current.isChecked())
        self.save_button.setEnabled(self.M is not None)
        return message

    def _update_combined(self, g, scale):
        """
        Update the blue model from the ticked saved ions, excluding the one
        being edited (shown in red). Returns status messages.
        """
        messages = []
        included = []
        for im in self.saved:
            if not im.include or im.key == self.current.key:
                continue
            try:
                included.append((im, im.model(self.S.x01, g)))
            except ModelError:
                messages.append(f"{im.label} has no lines in range")
        if included:
            x = self.M.x if self.M is not None else included[0][1].x
            self.combined_line.set_data(x, combine_models(included, x) * scale)
        self.combined_line.set_visible(bool(included))
        messages.append(f"{len(included)} saved ions shown")
        return messages

    # .........................................................................
    # Saved ions

    def _fill_saved_list(self):
        with blocked_signals(self.saved_list):
            self.saved_list.clear()
            for i, im in enumerate(self.saved):
                editing = im.key == self.current.key
                text = f"{im.label}   log A={im.logA:.2f}"
                item = QtWidgets.QListWidgetItem(
                    text + ("  (editing)" if editing else ""))
                item.setData(Qt.ItemDataRole.UserRole, i)
                item.setToolTip(f"Lorentz FWHM={im.wl} Å, N={im.N}")
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(
                    Qt.CheckState.Checked if im.include else Qt.CheckState.Unchecked)
                if editing:
                    # Shown in red instead, so not part of the blue model
                    font = item.font()
                    font.setItalic(True)
                    item.setFont(font)
                    item.setForeground(QtGui.QColor("grey"))
                self.saved_list.addItem(item)

    def _saved_from_item(self, item):
        return self.saved[item.data(Qt.ItemDataRole.UserRole)]

    def _selected_index(self):
        item = self.saved_list.currentItem()
        if item is None:
            self.status.showMessage("Select a saved ion first")
            return None
        return item.data(Qt.ItemDataRole.UserRole)

    def _saved_item_changed(self, item):
        self._saved_from_item(item).include = (
            item.checkState() == Qt.CheckState.Checked)
        # The list already shows the change (and must not be rebuilt from
        # within its own signal)
        self._saved_set_changed(refill=False)

    def _set_all_saved(self, state):
        for im in self.saved:
            im.include = state
        self._saved_set_changed()

    def _edit_selected(self):
        i = self._selected_index()
        if i is not None:
            self._load_into_current(self.saved[i])

    def _remove_selected(self):
        i = self._selected_index()
        if i is None:
            return
        im = self.saved.pop(i)
        self._saved_set_changed()
        self.status.showMessage(f"Removed {im.label} from the saved ions")

    def save_current(self):
        """
        Add the current ion to the saved ions, replacing any saved version
        """
        im = self.current.copy()
        im.include = True
        for i, other in enumerate(self.saved):
            if other.key == im.key:
                self.saved[i] = im
                break
        else:
            self.saved.append(im)
        self._saved_set_changed()
        self.status.showMessage(
            f"Saved {im.label} (log A={im.logA:.2f}) to {self.session_file}")

    def _saved_set_changed(self, refill=True):
        if refill:
            self._fill_saved_list()
        self.write_session()
        self.update_models()

    def write_session(self):
        g, scale = self.global_params()
        write_session(self.session_file, {**asdict(g), "scale": scale}, self.saved)

    def closeEvent(self, event):
        # Keep the global parameters for next time, but don't create a
        # session file if nothing was ever saved
        if self.saved or os.path.exists(self.session_file):
            self.write_session()
        super().closeEvent(event)


def run_gui(S, has_data, initial, saved, session_file):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    window = ElementSpecWindow(S, has_data, initial, saved, session_file)
    window.resize(1400, 800)
    window.show()
    return app.exec()
