from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional
import numpy as np

from .bundle import VisualizationBundle, load_bundle


def _equal_axes(ax, points):
    lo = np.min(points, axis=0)
    hi = np.max(points, axis=0)
    center = 0.5 * (lo + hi)
    radius = 0.55 * float(np.max(hi - lo))
    radius = max(radius, 0.15)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _marker_sizes(radii_m):
    # Schematic screen-space sizing; geometry is still represented by true centers/radii in bundle.
    r = np.asarray(radii_m, dtype=float)
    return np.clip((r / 0.02) ** 2 * 32.0, 8.0, 180.0)


def run_viewer(
    bundle: VisualizationBundle,
    *,
    point_limit: int = 30000,
    interval_ms: int = 60,
    show_start_goal_ghosts: bool = True,
):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button

    b = bundle
    n, s, _ = b.sphere_centers_base.shape

    # Stable point decimation for UI speed.
    points = b.scene_points_base
    colors = b.scene_colors_rgb
    if len(points) > point_limit:
        idx = np.linspace(0, len(points) - 1, point_limit).astype(np.int64)
        points = points[idx]
        if colors is not None:
            colors = colors[idx]

    fig = plt.figure(figsize=(11.5, 8.5))
    ax = fig.add_axes([0.05, 0.18, 0.72, 0.76], projection="3d")
    info_ax = fig.add_axes([0.79, 0.18, 0.19, 0.76])
    info_ax.axis("off")

    if colors is None:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1.0, alpha=0.18)
    else:
        cc = colors.astype(float)
        if cc.max(initial=1.0) > 1.0:
            cc = cc / 255.0
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1.0, c=cc, alpha=0.26)

    active_mask = b.sphere_active_mask
    static_mask = ~active_mask
    sizes = _marker_sizes(b.sphere_radii_m)

    # Static locked robot geometry is intentionally drawn once.
    if np.any(static_mask):
        c0 = b.sphere_centers_base[0, static_mask]
        ax.scatter(
            c0[:, 0], c0[:, 1], c0[:, 2],
            s=sizes[static_mask], alpha=0.18, label="locked robot spheres"
        )

    active0 = b.sphere_centers_base[0, active_mask]
    active_sc = ax.scatter(
        active0[:, 0], active0[:, 1], active0[:, 2],
        s=sizes[active_mask], alpha=0.82, label="moving right-arm spheres"
    )

    # End-effector path is fixed.
    ee = b.ee_positions_base
    ax.plot(ee[:, 0], ee[:, 1], ee[:, 2], linewidth=2.0, alpha=0.60, label="EE path")
    ee_now = ax.scatter([ee[0, 0]], [ee[0, 1]], [ee[0, 2]], s=70, label="EE current")

    if show_start_goal_ghosts:
        start = b.sphere_centers_base[0, active_mask]
        goal = b.sphere_centers_base[-1, active_mask]
        ax.scatter(start[:, 0], start[:, 1], start[:, 2],
                   s=sizes[active_mask] * 0.7, alpha=0.12, label="start ghost")
        ax.scatter(goal[:, 0], goal[:, 1], goal[:, 2],
                   s=sizes[active_mask] * 0.7, alpha=0.12, label="goal ghost")

    worst_sc = ax.scatter([], [], [], s=220, marker="o")
    worst_text = info_ax.text(0.0, 0.57, "", va="top", family="monospace")
    joint_text = info_ax.text(0.0, 0.98, "", va="top", family="monospace")
    status_text = info_ax.text(0.0, 0.73, "", va="top", family="monospace")

    combined = np.vstack([points, b.sphere_centers_base.reshape(-1, 3), ee])
    _equal_axes(ax, combined)
    ax.set_xlabel("X / m")
    ax.set_ylabel("Y / m")
    ax.set_zlabel("Z / m")
    ax.set_title("Route B — current → PREGRASP (right arm only)")
    ax.legend(loc="upper left", fontsize=8)

    slider_ax = fig.add_axes([0.09, 0.09, 0.60, 0.035])
    frame_slider = Slider(slider_ax, "Frame", 0, n - 1, valinit=0, valstep=1)
    play_ax = fig.add_axes([0.71, 0.083, 0.08, 0.05])
    play_btn = Button(play_ax, "Play")

    state = {"playing": False, "frame": 0}

    def _set_scatter_xyz(sc, xyz):
        sc._offsets3d = (xyz[:, 0], xyz[:, 1], xyz[:, 2])

    def _update(i):
        i = int(np.clip(i, 0, n - 1))
        state["frame"] = i
        centers = b.sphere_centers_base[i, active_mask]
        _set_scatter_xyz(active_sc, centers)
        _set_scatter_xyz(ee_now, ee[i:i+1])

        clear = None
        worst_idx = -1
        if b.frame_min_clearance_m is not None:
            clear = float(b.frame_min_clearance_m[i])
        if b.frame_worst_sphere_index is not None:
            worst_idx = int(b.frame_worst_sphere_index[i])

        if 0 <= worst_idx < s:
            p = b.sphere_centers_base[i, worst_idx:worst_idx+1]
            _set_scatter_xyz(worst_sc, p)
            worst_text.set_text(
                "Worst sphere\n"
                f"idx   : {worst_idx}\n"
                f"link  : {b.sphere_link_names[worst_idx]}\n"
                f"radius: {b.sphere_radii_m[worst_idx]*1000:.1f} mm"
            )
        else:
            _set_scatter_xyz(worst_sc, np.zeros((0, 3)))
            worst_text.set_text("Worst sphere\nnot provided")

        ctext = "n/a" if clear is None else f"{clear*1000:.1f} mm"
        status_text.set_text(
            f"Frame     : {i:02d}/{n-1:02d}\n"
            f"Time      : {b.time_s[i]:.3f} s\n"
            f"Clearance : {ctext}\n"
            f"Active DOF: 7\n"
            f"Locked DOF: 28"
        )
        qdeg = np.rad2deg(b.q_rad[i])
        lines = ["Right arm q (deg)", ""]
        for name, val in zip(b.joint_names, qdeg):
            lines.append(f"{name[-2:]:>2s}: {val:8.2f}")
        joint_text.set_text("\n".join(lines))
        fig.canvas.draw_idle()

    def _slider_changed(val):
        _update(int(val))

    def _toggle_play(event=None):
        state["playing"] = not state["playing"]
        play_btn.label.set_text("Pause" if state["playing"] else "Play")

    frame_slider.on_changed(_slider_changed)
    play_btn.on_clicked(_toggle_play)

    timer = fig.canvas.new_timer(interval=interval_ms)

    def _tick():
        if state["playing"]:
            nxt = state["frame"] + 1
            if nxt >= n:
                nxt = 0
            frame_slider.set_val(nxt)

    timer.add_callback(_tick)
    timer.start()

    def _key(event):
        if event.key == " ":
            _toggle_play()
        elif event.key == "right":
            frame_slider.set_val(min(n - 1, state["frame"] + 1))
        elif event.key == "left":
            frame_slider.set_val(max(0, state["frame"] - 1))
        elif event.key == "home":
            frame_slider.set_val(0)
        elif event.key == "end":
            frame_slider.set_val(n - 1)

    fig.canvas.mpl_connect("key_press_event", _key)
    _update(0)
    plt.show()


def main(argv: Optional[list[str]] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--point-limit", type=int, default=30000)
    parser.add_argument("--interval-ms", type=int, default=60)
    parser.add_argument("--no-ghosts", action="store_true")
    args = parser.parse_args(argv)
    b = load_bundle(args.bundle)
    run_viewer(
        b,
        point_limit=args.point_limit,
        interval_ms=args.interval_ms,
        show_start_goal_ghosts=not args.no_ghosts,
    )


if __name__ == "__main__":
    main()
