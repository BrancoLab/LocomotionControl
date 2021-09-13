import matplotlib
import matplotlib.pyplot as plt


def move_figure(f: plt.figure, x: int, y: int):
    """Move figure's upper left corner to pixel (x, y)"""
    backend = matplotlib.get_backend()
    if backend == "TkAgg":
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == "WXAgg":
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        f.canvas.manager.window.move(x, y)


def get_window_ticks(window: int, shifted: bool = True) -> dict:
    half_window, quarter_window = window / 2, window / 4

    if shifted:
        xticks = [
            -half_window,
            -quarter_window,
            0,
            quarter_window,
            half_window,
        ]
        labels = [round(t / 60, 3) for t in xticks]
    else:
        xticks = [
            0,
            quarter_window,
            half_window,
            half_window + quarter_window,
            window,
        ]
        labels = [round((t - half_window) / 60, 3) for t in xticks]
    return dict(xticks=xticks, xticklabels=labels, xlabel="time (s)",)
