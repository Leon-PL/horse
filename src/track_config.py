"""
Track Configuration — Static Lookup
====================================
Physical characteristics of UK & Ireland racecourses.

These attributes are **fixed** properties of the venue and provide
genuinely new information that cannot be derived from race results
alone.  A tight, left-handed track like Chester behaves very
differently from a galloping, right-handed track like Ascot.

Columns
-------
direction : str
    "Left", "Right", "Straight", or "Both" (figure-of-eight courses).
shape : str
    "Galloping" (long sweeping bends, suits big-striding horses),
    "Tight" (sharp turns, favours handy/agile types),
    "Undulating" (significant gradients — tests stamina),
    "Stiff" (uphill finish — often catches out tired horses),
    "Sharp" (very tight — similar to Tight but more extreme).
is_flat_track : int
    1 if the course hosts flat racing as its primary code.
is_jumps_track : int
    1 if the course hosts National Hunt (jumps) racing.
is_aw : int
    1 if the track has an all-weather (artificial) surface.
circumference_furlongs : float
    Approximate round-course circumference in furlongs.
    0 for straight-only courses.
    Values are approximate — exact figures vary by rail placement.
has_uphill_finish : int
    1 if the finish involves a notable uphill gradient.
has_downhill_section : int
    1 if the course has a significant downhill stretch.
draw_bias_strength : int
    0 = negligible draw bias, 1 = moderate, 2 = strong.
    Based on well-documented biases (e.g. Chester low draws,
    Beverley high draws at 5f, etc.).
"""

# -------------------------------------------------------------------
# Lookup table
# -------------------------------------------------------------------
# Each entry: {
#   "direction": str,
#   "shape": str,
#   "is_flat_track": 0/1,
#   "is_jumps_track": 0/1,
#   "is_aw": 0/1,
#   "circumference_f": float (furlongs),
#   "uphill_finish": 0/1,
#   "downhill_section": 0/1,
#   "draw_bias": 0/1/2,
# }

TRACK_CONFIG: dict[str, dict] = {
    # ── UK Flat ──────────────────────────────────────────────────
    "Ascot": {
        "direction": "Right", "shape": "Galloping",
        "is_flat_track": 1, "is_jumps_track": 1, "is_aw": 0,
        "circumference_f": 14.0, "uphill_finish": 1, "downhill_section": 1,
        "draw_bias": 1, "lat": 51.4107, "lon": -0.6747,
    },
    "Royal Ascot": {
        "direction": "Right", "shape": "Galloping",
        "is_flat_track": 1, "is_jumps_track": 1, "is_aw": 0,
        "circumference_f": 14.0, "uphill_finish": 1, "downhill_section": 1,
        "draw_bias": 1, "lat": 51.4107, "lon": -0.6747,
    },
    "Ayr": {
        "direction": "Left", "shape": "Galloping",
        "is_flat_track": 1, "is_jumps_track": 1, "is_aw": 0,
        "circumference_f": 12.0, "uphill_finish": 0, "downhill_section": 0,
        "draw_bias": 1, "lat": 55.4580, "lon": -4.6296,
    },
    "Bath": {
        "direction": "Left", "shape": "Undulating",
        "is_flat_track": 1, "is_jumps_track": 0, "is_aw": 0,
        "circumference_f": 10.0, "uphill_finish": 1, "downhill_section": 1,
        "draw_bias": 1, "lat": 51.3963, "lon": -2.3425,
    },
    "Beverley": {
        "direction": "Right", "shape": "Stiff",
        "is_flat_track": 1, "is_jumps_track": 0, "is_aw": 0,
        "circumference_f": 10.5, "uphill_finish": 1, "downhill_section": 0,
        "draw_bias": 2, "lat": 53.8575, "lon": -0.4128,
    },
    "Brighton": {
        "direction": "Left", "shape": "Undulating",
        "is_flat_track": 1, "is_jumps_track": 0, "is_aw": 0,
        "circumference_f": 0.0, "uphill_finish": 0, "downhill_section": 1,
        "draw_bias": 1, "lat": 50.8405, "lon": -0.1182,
    },
    "Carlisle": {
        "direction": "Right", "shape": "Undulating",
        "is_flat_track": 1, "is_jumps_track": 1, "is_aw": 0,
        "circumference_f": 11.0, "uphill_finish": 1, "downhill_section": 0,
        "draw_bias": 1, "lat": 54.9044, "lon": -2.9316,
    },
    "Catterick": {
        "direction": "Left", "shape": "Sharp",
        "is_flat_track": 1, "is_jumps_track": 1, "is_aw": 0,
        "circumference_f": 9.0, "uphill_finish": 0, "downhill_section": 1,
        "draw_bias": 1, "lat": 54.3806, "lon": -1.6436,
    },
    "Chelmsford City": {
        "direction": "Left", "shape": "Galloping",
        "is_flat_track": 1, "is_jumps_track": 0, "is_aw": 1,
        "circumference_f": 10.0, "uphill_finish": 0, "downhill_section": 0,
        "draw_bias": 1, "lat": 51.7293, "lon": 0.4816,
    },
    "Chester": {
        "direction": "Left", "shape": "Tight",
        "is_flat_track": 1, "is_jumps_track": 0, "is_aw": 0,
        "circumference_f": 8.0, "uphill_finish": 0, "downhill_section": 0,
        "draw_bias": 2, "lat": 53.1827, "lon": -2.8936,
    },
    "Doncaster": {
        "direction": "Left", "shape": "Galloping",
        "is_flat_track": 1, "is_jumps_track": 1, "is_aw": 0,
        "circumference_f": 15.0, "uphill_finish": 0, "downhill_section": 0,
        "draw_bias": 1, "lat": 53.5171, "lon": -1.1108,
    },
    "Epsom Downs": {
        "direction": "Left", "shape": "Undulating",
        "is_flat_track": 1, "is_jumps_track": 0, "is_aw": 0,
        "circumference_f": 12.0, "uphill_finish": 0, "downhill_section": 1,
        "draw_bias": 2, "lat": 51.3217, "lon": -0.2618,
    },
    "Goodwood": {
        "direction": "Right", "shape": "Undulating",
        "is_flat_track": 1, "is_jumps_track": 0, "is_aw": 0,
        "circumference_f": 11.0, "uphill_finish": 1, "downhill_section": 1,
        "draw_bias": 2, "lat": 50.8871, "lon": -0.7528,
    },
    "Hamilton": {
        "direction": "Right", "shape": "Undulating",
        "is_flat_track": 1, "is_jumps_track": 0, "is_aw": 0,
        "circumference_f": 10.0, "uphill_finish": 1, "downhill_section": 0,
        "draw_bias": 1, "lat": 55.7864, "lon": -4.0378,
    },
    "Haydock": {
        "direction": "Left", "shape": "Galloping",
        "is_flat_track": 1, "is_jumps_track": 1, "is_aw": 0,
        "circumference_f": 12.0, "uphill_finish": 0, "downhill_section": 0,
        "draw_bias": 1, "lat": 53.4750, "lon": -2.6350,
    },
    "Kempton": {
        "direction": "Right", "shape": "Tight",
        "is_flat_track": 1, "is_jumps_track": 1, "is_aw": 1,
        "circumference_f": 10.0, "uphill_finish": 0, "downhill_section": 0,
        "draw_bias": 1, "lat": 51.4115, "lon": -0.4109,
    },
    "Leicester": {
        "direction": "Right", "shape": "Stiff",
        "is_flat_track": 1, "is_jumps_track": 0, "is_aw": 0,
        "circumference_f": 11.0, "uphill_finish": 1, "downhill_section": 0,
        "draw_bias": 1, "lat": 52.6228, "lon": -1.0988,
    },
    "Lingfield": {
        "direction": "Left", "shape": "Sharp",
        "is_flat_track": 1, "is_jumps_track": 1, "is_aw": 1,
        "circumference_f": 9.0, "uphill_finish": 0, "downhill_section": 1,
        "draw_bias": 1, "lat": 51.1754, "lon": -0.0188,
    },
    "Musselburgh": {
        "direction": "Right", "shape": "Tight",
        "is_flat_track": 1, "is_jumps_track": 1, "is_aw": 0,
        "circumference_f": 8.5, "uphill_finish": 0, "downhill_section": 0,
        "draw_bias": 2, "lat": 55.9441, "lon": -3.0960,
    },
    "Newbury": {
        "direction": "Left", "shape": "Galloping",
        "is_flat_track": 1, "is_jumps_track": 1, "is_aw": 0,
        "circumference_f": 14.0, "uphill_finish": 0, "downhill_section": 0,
        "draw_bias": 0, "lat": 51.4019, "lon": -1.3086,
    },
    "Newcastle": {
        "direction": "Left", "shape": "Galloping",
        "is_flat_track": 1, "is_jumps_track": 1, "is_aw": 1,
        "circumference_f": 12.0, "uphill_finish": 0, "downhill_section": 0,
        "draw_bias": 1, "lat": 55.0044, "lon": -1.6318,
    },
    "Newmarket": {
        "direction": "Right", "shape": "Galloping",
        "is_flat_track": 1, "is_jumps_track": 0, "is_aw": 0,
        "circumference_f": 0.0, "uphill_finish": 1, "downhill_section": 1,
        "draw_bias": 2, "lat": 52.2472, "lon": 0.3713,
    },
    "Nottingham": {
        "direction": "Left", "shape": "Galloping",
        "is_flat_track": 1, "is_jumps_track": 0, "is_aw": 0,
        "circumference_f": 11.0, "uphill_finish": 0, "downhill_section": 0,
        "draw_bias": 1, "lat": 52.9271, "lon": -1.0990,
    },
    "Pontefract": {
        "direction": "Left", "shape": "Undulating",
        "is_flat_track": 1, "is_jumps_track": 0, "is_aw": 0,
        "circumference_f": 16.0, "uphill_finish": 1, "downhill_section": 1,
        "draw_bias": 1, "lat": 53.6925, "lon": -1.2989,
    },
    "Redcar": {
        "direction": "Left", "shape": "Galloping",
        "is_flat_track": 1, "is_jumps_track": 0, "is_aw": 0,
        "circumference_f": 12.0, "uphill_finish": 0, "downhill_section": 0,
        "draw_bias": 1, "lat": 54.6133, "lon": -1.0597,
    },
    "Ripon": {
        "direction": "Right", "shape": "Undulating",
        "is_flat_track": 1, "is_jumps_track": 0, "is_aw": 0,
        "circumference_f": 10.0, "uphill_finish": 1, "downhill_section": 0,
        "draw_bias": 1, "lat": 54.1358, "lon": -1.5203,
    },
    "Salisbury": {
        "direction": "Right", "shape": "Undulating",
        "is_flat_track": 1, "is_jumps_track": 0, "is_aw": 0,
        "circumference_f": 0.0, "uphill_finish": 1, "downhill_section": 1,
        "draw_bias": 1, "lat": 51.0985, "lon": -1.7671,
    },
    "Sandown": {
        "direction": "Right", "shape": "Stiff",
        "is_flat_track": 1, "is_jumps_track": 1, "is_aw": 0,
        "circumference_f": 11.0, "uphill_finish": 1, "downhill_section": 1,
        "draw_bias": 1, "lat": 51.3761, "lon": -0.3571,
    },
    "Southwell": {
        "direction": "Left", "shape": "Sharp",
        "is_flat_track": 1, "is_jumps_track": 1, "is_aw": 1,
        "circumference_f": 8.0, "uphill_finish": 0, "downhill_section": 0,
        "draw_bias": 1, "lat": 53.0747, "lon": -0.8933,
    },
    "Thirsk": {
        "direction": "Left", "shape": "Sharp",
        "is_flat_track": 1, "is_jumps_track": 0, "is_aw": 0,
        "circumference_f": 9.0, "uphill_finish": 0, "downhill_section": 0,
        "draw_bias": 1, "lat": 54.2278, "lon": -1.3336,
    },
    "Windsor": {
        "direction": "Right", "shape": "Tight",
        "is_flat_track": 1, "is_jumps_track": 0, "is_aw": 0,
        "circumference_f": 0.0, "uphill_finish": 0, "downhill_section": 0,
        "draw_bias": 1, "lat": 51.4876, "lon": -0.6105,
    },
    "Wolverhampton": {
        "direction": "Left", "shape": "Tight",
        "is_flat_track": 1, "is_jumps_track": 0, "is_aw": 1,
        "circumference_f": 8.0, "uphill_finish": 0, "downhill_section": 0,
        "draw_bias": 1, "lat": 52.5951, "lon": -2.0997,
    },
    "Yarmouth": {
        "direction": "Left", "shape": "Galloping",
        "is_flat_track": 1, "is_jumps_track": 0, "is_aw": 0,
        "circumference_f": 10.0, "uphill_finish": 0, "downhill_section": 0,
        "draw_bias": 0, "lat": 52.6132, "lon": 1.7225,
    },
    "York": {
        "direction": "Left", "shape": "Galloping",
        "is_flat_track": 1, "is_jumps_track": 0, "is_aw": 0,
        "circumference_f": 16.0, "uphill_finish": 0, "downhill_section": 0,
        "draw_bias": 1, "lat": 53.9511, "lon": -1.0878,
    },

    # ── UK National Hunt ─────────────────────────────────────────
    "Aintree": {
        "direction": "Left", "shape": "Galloping",
        "is_flat_track": 0, "is_jumps_track": 1, "is_aw": 0,
        "circumference_f": 16.0, "uphill_finish": 0, "downhill_section": 0,
        "draw_bias": 0, "lat": 53.4764, "lon": -2.9512,
    },
    "Bangor-on-Dee": {
        "direction": "Left", "shape": "Sharp",
        "is_flat_track": 0, "is_jumps_track": 1, "is_aw": 0,
        "circumference_f": 9.0, "uphill_finish": 0, "downhill_section": 0,
        "draw_bias": 0, "lat": 52.9905, "lon": -2.9282,
    },
    "Cartmel": {
        "direction": "Left", "shape": "Tight",
        "is_flat_track": 0, "is_jumps_track": 1, "is_aw": 0,
        "circumference_f": 6.0, "uphill_finish": 0, "downhill_section": 0,
        "draw_bias": 0, "lat": 54.1976, "lon": -2.9525,
    },
    "Cheltenham": {
        "direction": "Left", "shape": "Undulating",
        "is_flat_track": 0, "is_jumps_track": 1, "is_aw": 0,
        "circumference_f": 12.0, "uphill_finish": 1, "downhill_section": 1,
        "draw_bias": 0, "lat": 51.9196, "lon": -2.0669,
    },
    "Chepstow": {
        "direction": "Left", "shape": "Undulating",
        "is_flat_track": 1, "is_jumps_track": 1, "is_aw": 0,
        "circumference_f": 14.0, "uphill_finish": 1, "downhill_section": 1,
        "draw_bias": 0, "lat": 51.6356, "lon": -2.6814,
    },
    "Exeter": {
        "direction": "Right", "shape": "Undulating",
        "is_flat_track": 0, "is_jumps_track": 1, "is_aw": 0,
        "circumference_f": 14.0, "uphill_finish": 1, "downhill_section": 1,
        "draw_bias": 0, "lat": 50.7265, "lon": -3.4773,
    },
    "Fakenham": {
        "direction": "Left", "shape": "Sharp",
        "is_flat_track": 0, "is_jumps_track": 1, "is_aw": 0,
        "circumference_f": 7.0, "uphill_finish": 1, "downhill_section": 0,
        "draw_bias": 0, "lat": 52.8357, "lon": 0.8484,
    },
    "Ffos Las": {
        "direction": "Left", "shape": "Galloping",
        "is_flat_track": 1, "is_jumps_track": 1, "is_aw": 0,
        "circumference_f": 12.0, "uphill_finish": 0, "downhill_section": 0,
        "draw_bias": 0, "lat": 51.7917, "lon": -4.1650,
    },
    "Fontwell": {
        "direction": "Both", "shape": "Tight",
        "is_flat_track": 0, "is_jumps_track": 1, "is_aw": 0,
        "circumference_f": 7.0, "uphill_finish": 0, "downhill_section": 0,
        "draw_bias": 0, "lat": 50.8663, "lon": -0.6613,
    },
    "Hereford": {
        "direction": "Right", "shape": "Sharp",
        "is_flat_track": 0, "is_jumps_track": 1, "is_aw": 0,
        "circumference_f": 9.0, "uphill_finish": 0, "downhill_section": 0,
        "draw_bias": 0, "lat": 52.0545, "lon": -2.7337,
    },
    "Hexham": {
        "direction": "Left", "shape": "Undulating",
        "is_flat_track": 0, "is_jumps_track": 1, "is_aw": 0,
        "circumference_f": 10.0, "uphill_finish": 1, "downhill_section": 1,
        "draw_bias": 0, "lat": 54.9704, "lon": -2.0829,
    },
    "Huntingdon": {
        "direction": "Right", "shape": "Galloping",
        "is_flat_track": 0, "is_jumps_track": 1, "is_aw": 0,
        "circumference_f": 12.0, "uphill_finish": 0, "downhill_section": 0,
        "draw_bias": 0, "lat": 52.3428, "lon": -0.1673,
    },
    "Kelso": {
        "direction": "Left", "shape": "Sharp",
        "is_flat_track": 0, "is_jumps_track": 1, "is_aw": 0,
        "circumference_f": 9.0, "uphill_finish": 0, "downhill_section": 0,
        "draw_bias": 0, "lat": 55.5935, "lon": -2.4268,
    },
    "Ludlow": {
        "direction": "Right", "shape": "Sharp",
        "is_flat_track": 0, "is_jumps_track": 1, "is_aw": 0,
        "circumference_f": 9.0, "uphill_finish": 0, "downhill_section": 0,
        "draw_bias": 0, "lat": 52.3672, "lon": -2.7130,
    },
    "Market Rasen": {
        "direction": "Right", "shape": "Sharp",
        "is_flat_track": 0, "is_jumps_track": 1, "is_aw": 0,
        "circumference_f": 9.0, "uphill_finish": 0, "downhill_section": 0,
        "draw_bias": 0, "lat": 53.3785, "lon": -0.3293,
    },
    "Newton Abbot": {
        "direction": "Left", "shape": "Sharp",
        "is_flat_track": 0, "is_jumps_track": 1, "is_aw": 0,
        "circumference_f": 8.0, "uphill_finish": 0, "downhill_section": 0,
        "draw_bias": 0, "lat": 50.5235, "lon": -3.6078,
    },
    "Perth": {
        "direction": "Right", "shape": "Galloping",
        "is_flat_track": 0, "is_jumps_track": 1, "is_aw": 0,
        "circumference_f": 10.0, "uphill_finish": 0, "downhill_section": 0,
        "draw_bias": 0, "lat": 56.4098, "lon": -3.4476,
    },
    "Plumpton": {
        "direction": "Left", "shape": "Tight",
        "is_flat_track": 0, "is_jumps_track": 1, "is_aw": 0,
        "circumference_f": 7.0, "uphill_finish": 1, "downhill_section": 1,
        "draw_bias": 0, "lat": 50.9192, "lon": -0.0640,
    },
    "Sedgefield": {
        "direction": "Left", "shape": "Undulating",
        "is_flat_track": 0, "is_jumps_track": 1, "is_aw": 0,
        "circumference_f": 8.0, "uphill_finish": 1, "downhill_section": 0,
        "draw_bias": 0, "lat": 54.6512, "lon": -1.4555,
    },
    "Stratford": {
        "direction": "Left", "shape": "Sharp",
        "is_flat_track": 0, "is_jumps_track": 1, "is_aw": 0,
        "circumference_f": 8.0, "uphill_finish": 0, "downhill_section": 0,
        "draw_bias": 0, "lat": 52.1948, "lon": -1.7038,
    },
    "Taunton": {
        "direction": "Right", "shape": "Tight",
        "is_flat_track": 0, "is_jumps_track": 1, "is_aw": 0,
        "circumference_f": 9.0, "uphill_finish": 0, "downhill_section": 0,
        "draw_bias": 0, "lat": 51.0119, "lon": -3.0658,
    },
    "Uttoxeter": {
        "direction": "Left", "shape": "Galloping",
        "is_flat_track": 0, "is_jumps_track": 1, "is_aw": 0,
        "circumference_f": 10.0, "uphill_finish": 0, "downhill_section": 0,
        "draw_bias": 0, "lat": 52.8959, "lon": -1.8523,
    },
    "Warwick": {
        "direction": "Left", "shape": "Sharp",
        "is_flat_track": 0, "is_jumps_track": 1, "is_aw": 0,
        "circumference_f": 10.0, "uphill_finish": 1, "downhill_section": 0,
        "draw_bias": 0, "lat": 52.2897, "lon": -1.5852,
    },
    "Wetherby": {
        "direction": "Left", "shape": "Galloping",
        "is_flat_track": 0, "is_jumps_track": 1, "is_aw": 0,
        "circumference_f": 12.0, "uphill_finish": 0, "downhill_section": 0,
        "draw_bias": 0, "lat": 53.9241, "lon": -1.3826,
    },
    "Wincanton": {
        "direction": "Right", "shape": "Galloping",
        "is_flat_track": 0, "is_jumps_track": 1, "is_aw": 0,
        "circumference_f": 10.0, "uphill_finish": 1, "downhill_section": 0,
        "draw_bias": 0, "lat": 51.0590, "lon": -2.3857,
    },
    "Worcester": {
        "direction": "Left", "shape": "Galloping",
        "is_flat_track": 0, "is_jumps_track": 1, "is_aw": 0,
        "circumference_f": 12.0, "uphill_finish": 0, "downhill_section": 0,
        "draw_bias": 0, "lat": 52.1998, "lon": -2.2329,
    },

    # ── Ireland ──────────────────────────────────────────────────
    "Ballinrobe": {
        "direction": "Right", "shape": "Tight",
        "is_flat_track": 0, "is_jumps_track": 1, "is_aw": 0,
        "circumference_f": 9.0, "uphill_finish": 0, "downhill_section": 0,
        "draw_bias": 0, "lat": 53.6310, "lon": -9.2240,
    },
    "Bellewstown": {
        "direction": "Left", "shape": "Undulating",
        "is_flat_track": 1, "is_jumps_track": 1, "is_aw": 0,
        "circumference_f": 9.0, "uphill_finish": 1, "downhill_section": 1,
        "draw_bias": 0, "lat": 53.6870, "lon": -6.4224,
    },
    "Clonmel": {
        "direction": "Right", "shape": "Undulating",
        "is_flat_track": 0, "is_jumps_track": 1, "is_aw": 0,
        "circumference_f": 9.0, "uphill_finish": 1, "downhill_section": 0,
        "draw_bias": 0, "lat": 52.3397, "lon": -7.6930,
    },
    "Cork": {
        "direction": "Right", "shape": "Galloping",
        "is_flat_track": 1, "is_jumps_track": 1, "is_aw": 0,
        "circumference_f": 12.0, "uphill_finish": 1, "downhill_section": 0,
        "draw_bias": 0, "lat": 51.9233, "lon": -8.4838,
    },
    "Curragh": {
        "direction": "Right", "shape": "Galloping",
        "is_flat_track": 1, "is_jumps_track": 0, "is_aw": 0,
        "circumference_f": 16.0, "uphill_finish": 0, "downhill_section": 0,
        "draw_bias": 1, "lat": 53.1590, "lon": -6.8100,
    },
    "Down Royal": {
        "direction": "Right", "shape": "Galloping",
        "is_flat_track": 1, "is_jumps_track": 1, "is_aw": 0,
        "circumference_f": 11.0, "uphill_finish": 1, "downhill_section": 0,
        "draw_bias": 0, "lat": 54.4766, "lon": -6.1161,
    },
    "Downpatrick": {
        "direction": "Right", "shape": "Undulating",
        "is_flat_track": 0, "is_jumps_track": 1, "is_aw": 0,
        "circumference_f": 9.0, "uphill_finish": 1, "downhill_section": 1,
        "draw_bias": 0, "lat": 54.3176, "lon": -5.7133,
    },
    "Dundalk": {
        "direction": "Left", "shape": "Tight",
        "is_flat_track": 1, "is_jumps_track": 0, "is_aw": 1,
        "circumference_f": 9.0, "uphill_finish": 0, "downhill_section": 0,
        "draw_bias": 1, "lat": 53.9963, "lon": -6.3847,
    },
    "Fairyhouse": {
        "direction": "Right", "shape": "Galloping",
        "is_flat_track": 1, "is_jumps_track": 1, "is_aw": 0,
        "circumference_f": 12.0, "uphill_finish": 0, "downhill_section": 0,
        "draw_bias": 0, "lat": 53.5232, "lon": -6.5280,
    },
    "Galway": {
        "direction": "Right", "shape": "Undulating",
        "is_flat_track": 1, "is_jumps_track": 1, "is_aw": 0,
        "circumference_f": 10.0, "uphill_finish": 1, "downhill_section": 1,
        "draw_bias": 1, "lat": 53.2896, "lon": -8.9759,
    },
    "Gowran Park": {
        "direction": "Right", "shape": "Stiff",
        "is_flat_track": 1, "is_jumps_track": 1, "is_aw": 0,
        "circumference_f": 11.0, "uphill_finish": 1, "downhill_section": 0,
        "draw_bias": 0, "lat": 52.6264, "lon": -7.1561,
    },
    "Kilbeggan": {
        "direction": "Right", "shape": "Tight",
        "is_flat_track": 0, "is_jumps_track": 1, "is_aw": 0,
        "circumference_f": 8.0, "uphill_finish": 1, "downhill_section": 0,
        "draw_bias": 0, "lat": 53.3696, "lon": -7.5037,
    },
    "Killarney": {
        "direction": "Left", "shape": "Sharp",
        "is_flat_track": 1, "is_jumps_track": 1, "is_aw": 0,
        "circumference_f": 9.0, "uphill_finish": 0, "downhill_section": 0,
        "draw_bias": 0, "lat": 52.0610, "lon": -9.5060,
    },
    "Laytown": {
        "direction": "Straight", "shape": "Galloping",
        "is_flat_track": 1, "is_jumps_track": 0, "is_aw": 0,
        "circumference_f": 0.0, "uphill_finish": 0, "downhill_section": 0,
        "draw_bias": 1, "lat": 53.6822, "lon": -6.2376,
    },
    "Leopardstown": {
        "direction": "Left", "shape": "Galloping",
        "is_flat_track": 1, "is_jumps_track": 1, "is_aw": 0,
        "circumference_f": 14.0, "uphill_finish": 0, "downhill_section": 0,
        "draw_bias": 0, "lat": 53.2756, "lon": -6.2080,
    },
    "Limerick": {
        "direction": "Right", "shape": "Undulating",
        "is_flat_track": 1, "is_jumps_track": 1, "is_aw": 0,
        "circumference_f": 10.0, "uphill_finish": 1, "downhill_section": 0,
        "draw_bias": 0, "lat": 52.6049, "lon": -8.7135,
    },
    "Listowel": {
        "direction": "Left", "shape": "Sharp",
        "is_flat_track": 1, "is_jumps_track": 1, "is_aw": 0,
        "circumference_f": 8.0, "uphill_finish": 0, "downhill_section": 0,
        "draw_bias": 0, "lat": 52.4438, "lon": -9.4888,
    },
    "Naas": {
        "direction": "Left", "shape": "Galloping",
        "is_flat_track": 1, "is_jumps_track": 1, "is_aw": 0,
        "circumference_f": 10.0, "uphill_finish": 0, "downhill_section": 0,
        "draw_bias": 0, "lat": 53.2169, "lon": -6.6506,
    },
    "Navan": {
        "direction": "Left", "shape": "Galloping",
        "is_flat_track": 1, "is_jumps_track": 1, "is_aw": 0,
        "circumference_f": 12.0, "uphill_finish": 0, "downhill_section": 0,
        "draw_bias": 0, "lat": 53.6540, "lon": -6.6960,
    },
    "Punchestown": {
        "direction": "Right", "shape": "Galloping",
        "is_flat_track": 0, "is_jumps_track": 1, "is_aw": 0,
        "circumference_f": 16.0, "uphill_finish": 0, "downhill_section": 0,
        "draw_bias": 0, "lat": 53.1780, "lon": -6.6510,
    },
    "Roscommon": {
        "direction": "Right", "shape": "Undulating",
        "is_flat_track": 1, "is_jumps_track": 1, "is_aw": 0,
        "circumference_f": 10.0, "uphill_finish": 1, "downhill_section": 0,
        "draw_bias": 0, "lat": 53.6338, "lon": -8.1904,
    },
    "Sligo": {
        "direction": "Right", "shape": "Undulating",
        "is_flat_track": 1, "is_jumps_track": 1, "is_aw": 0,
        "circumference_f": 8.0, "uphill_finish": 1, "downhill_section": 0,
        "draw_bias": 0, "lat": 54.2636, "lon": -8.4760,
    },
    "Thurles": {
        "direction": "Right", "shape": "Tight",
        "is_flat_track": 0, "is_jumps_track": 1, "is_aw": 0,
        "circumference_f": 8.0, "uphill_finish": 0, "downhill_section": 0,
        "draw_bias": 0, "lat": 52.6845, "lon": -7.8086,
    },
    "Tipperary": {
        "direction": "Left", "shape": "Sharp",
        "is_flat_track": 1, "is_jumps_track": 1, "is_aw": 0,
        "circumference_f": 9.0, "uphill_finish": 0, "downhill_section": 0,
        "draw_bias": 0, "lat": 52.4720, "lon": -8.1610,
    },
    "Tramore": {
        "direction": "Right", "shape": "Undulating",
        "is_flat_track": 1, "is_jumps_track": 1, "is_aw": 0,
        "circumference_f": 8.0, "uphill_finish": 1, "downhill_section": 1,
        "draw_bias": 0, "lat": 52.1592, "lon": -7.1271,
    },
    "Wexford": {
        "direction": "Right", "shape": "Tight",
        "is_flat_track": 0, "is_jumps_track": 1, "is_aw": 0,
        "circumference_f": 8.0, "uphill_finish": 0, "downhill_section": 0,
        "draw_bias": 0, "lat": 52.3363, "lon": -6.4530,
    },
}

# -------------------------------------------------------------------
# Encode helpers
# -------------------------------------------------------------------
# Maps for one-hot-like numeric encoding (avoids sparse dummies)
_DIRECTION_MAP = {"Left": 0, "Right": 1, "Straight": 2, "Both": 3}
_SHAPE_MAP = {
    "Galloping": 0,
    "Tight": 1,
    "Sharp": 2,
    "Undulating": 3,
    "Stiff": 4,
}

# Default config for tracks not in the lookup table
_DEFAULT = {
    "direction": "Right",
    "shape": "Galloping",
    "is_flat_track": 1,
    "is_jumps_track": 0,
    "is_aw": 0,
    "circumference_f": 10.0,
    "uphill_finish": 0,
    "downhill_section": 0,
    "draw_bias": 0,
    "lat": 52.0,       # UK centroid fallback
    "lon": -1.5,
}


# Build a case-insensitive lookup for matching
_TRACK_CONFIG_LOWER = {k.lower(): v for k, v in TRACK_CONFIG.items()}


def get_track_config(track_name: str) -> dict:
    """Lookup a track's static configuration, with case-insensitive matching."""
    # Try exact match first
    if track_name in TRACK_CONFIG:
        return TRACK_CONFIG[track_name]

    # Case-insensitive fallback
    lower = track_name.strip().lower()
    if lower in _TRACK_CONFIG_LOWER:
        return _TRACK_CONFIG_LOWER[lower]

    return _DEFAULT


def direction_code(direction: str) -> int:
    """Numeric code for track direction."""
    return _DIRECTION_MAP.get(direction, 1)


def shape_code(shape: str) -> int:
    """Numeric code for track shape."""
    return _SHAPE_MAP.get(shape, 0)
