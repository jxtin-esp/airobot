"""Control logic for the NEXUS arm (no GUI, no serial), so it can be tested on its own.

A continuous-rotation servo has no position feedback. The webcam measures the real
joint angle, and this module decides, many times a second, whether the joint should
move left, right or stop, and how fast.

The webcam, the serial link and the servo all add delay, so a joint keeps moving for a
moment after a stop is requested. To still stop in time, the speed is capped by the
distance that is left (to a limit or to a target): the closer it gets, the slower it goes.
"""

MIN_PCT = 15            # below this a continuous servo does not turn
TOL_DEG = 2.0           # close enough to the target
LEAD_S = 0.30           # total delay: camera + processing + serial + servo spin-down (seconds)
TRACKED_MAX_PCT = 60    # speed cap while the webcam is protecting a limit
DEG_PER_PCT = 4.0       # assumed (cautious) servo speed: degrees per second for each 1 % of speed


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def servo_angle(joint_angle):
    """Joint angle -90..+90 (0 = neutral) to a positional servo angle 0..180."""
    return int(round(clamp(joint_angle, -90.0, 90.0) + 90.0))


def safe_pct(distance_left, wanted_pct, deg_per_pct=DEG_PER_PCT):
    """Highest speed (percent) that can still stop within distance_left degrees."""
    allowed = distance_left / (deg_per_pct * LEAD_S)
    return int(clamp(allowed, MIN_PCT, wanted_pct))


def update_speed_estimate(previous, velocity, pct):
    """Learn how fast a servo really is, from the webcam, while it moves.

    Never goes below the cautious default, so a slow reading cannot make it unsafe.
    """
    if pct < 20 or abs(velocity) < 10.0:
        return previous
    observed = abs(velocity) / pct
    return max(DEG_PER_PCT, 0.7 * previous + 0.3 * observed)


def decide_motion(angle, fresh, velocity, held_dir, seek_target,
                  lo, hi, margin, speed_pct, limits_on, deg_per_pct=DEG_PER_PCT):
    """Decide what one continuous joint should do right now.

    angle        webcam-measured joint angle in degrees, or None
    fresh        True if the measurement is recent
    velocity     measured speed in degrees per second (signed), 0 if unknown
    held_dir     -1 / 0 / +1 from the Left / Right buttons
    seek_target  target angle for a closed-loop move, or None
    lo, hi       joint limits in degrees
    margin       stop this many degrees before a limit
    speed_pct    speed cap, percent
    limits_on    whether limit protection is enabled
    deg_per_pct  estimated servo speed, degrees per second per 1 % (see update_speed_estimate)

    Returns (direction, percent, note, seek_done)
      note is "", "lost", "limit_min" or "limit_max"
    """
    direction = 0
    pct = speed_pct

    if held_dir:
        direction = held_dir
    elif seek_target is not None:
        if not fresh or angle is None:
            return 0, pct, "lost", False
        err = seek_target - angle
        # stop early by part of the distance the joint will still travel while the stop arrives
        coast = LEAD_S * abs(velocity) if velocity * err > 0 else 0.0
        if abs(err) <= TOL_DEG + coast * 0.5:
            return 0, pct, "", True
        direction = 1 if err > 0 else -1
        pct = safe_pct(abs(err) - TOL_DEG, pct, deg_per_pct)

    if direction and limits_on and fresh and angle is not None:
        distance = (hi - angle) if direction > 0 else (angle - lo)
        note = "limit_max" if direction > 0 else "limit_min"
        if distance - margin <= 0:
            return 0, pct, note, False
        pct = safe_pct(distance - margin, min(pct, TRACKED_MAX_PCT), deg_per_pct)
        ahead = abs(velocity) * LEAD_S if velocity * direction > 0 else 0.0
        if ahead >= distance - margin:
            return 0, pct, note, False

    return direction, pct, "", False
