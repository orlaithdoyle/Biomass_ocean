"""
bm_tools.py

shared bits for the biomass ocean notebooks so they don't each carry a copy:
login, reading L1a SCS products, geocoding, multilooking, circular stats,
seawater + bragg physics and a few readers for the ancillary data.

Orlaith Doyle (Ni Dhuill), ESA ESTEC Biomass internship 2026
"""

import re
import datetime as dt
from pathlib import Path

import numpy as np
import tifffile
from lxml import etree
from scipy.interpolate import CubicSpline, RegularGridInterpolator
from scipy.optimize import least_squares

C = 299792458.0
EPS0 = 8.854187817e-12
G = 9.81
WGS_A = 6378137.0
WGS_E2 = 6.69437999014e-3

_trapz = getattr(np, "trapezoid", None) or np.trapz   # numpy 2 renamed it


# ----------------------------------------------------------------- login

def load_creds(path):
    """reads key = value lines from the local credentials file (never commit this file)"""
    creds = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            creds[k.strip()] = v.strip().strip('"').strip("'")
    return creds


def get_token(creds, token_url):
    import requests
    r = requests.post(token_url, data={
        "grant_type": "refresh_token",
        "refresh_token": creds["maap_offline_token"],
        "client_id": creds["maap_client_id"],
        "client_secret": creds["maap_client_secret"],
        "scope": "offline_access openid"}, timeout=60)
    if r.status_code in (400, 401):
        # the offline token dies after ~90 days or a quiet spell, with no warning
        raise RuntimeError(f"keycloak said {r.status_code}, offline token has probably expired, make a new one on MAAP")
    r.raise_for_status()
    return r.json()["access_token"]


# ----------------------------------------------------------------- reading products

DEFAULT_ORDER = ["VV", "VH", "HV", "HH"]   # what the SCS tiffs actually hold, reversed vs the annotation list


def product_files(prod):
    prod = Path(prod)
    tifs = sorted(prod.rglob("*.tif*"))
    amp = [t for t in tifs if re.search(r"abs|amp", t.name, re.I)]
    ph = [t for t in tifs if re.search(r"phase", t.name, re.I)]
    xmls = sorted(prod.rglob("*.xml"))
    if not amp or not ph:
        raise FileNotFoundError(f"no amplitude/phase tiffs found under {prod}")
    return amp[0], ph[0], xmls


def band_order(tif_path):
    """polarisation of each band, read from the tiff metadata instead of assumed.
    getting this wrong swaps HH and VV which flips the CPD sign and inverts HH/VV"""
    txt = ""
    with tifffile.TiffFile(tif_path) as tf:
        for tag in tf.pages[0].tags.values():
            if isinstance(tag.value, str):
                txt += tag.value + "\n"
    hits = re.findall(r'sample="(\d+)"[^>]*>\s*(HH|HV|VH|VV)\s*<', txt)
    if len(hits) == 4:
        return [p for _, p in sorted(hits, key=lambda h: int(h[0]))]
    seen = []
    for p in re.findall(r"\b(HH|HV|VH|VV)\b", txt):
        if p not in seen:
            seen.append(p)
    if len(seen) == 4:
        return seen
    print(f"  ! no band names in {Path(tif_path).name}, falling back to {DEFAULT_ORDER}. CHECK THIS")
    return DEFAULT_ORDER


def read_slc(prod, pols=("HH", "HV", "VH", "VV")):
    """complex SLC per polarisation, amplitude * exp(i phase)"""
    amp_f, ph_f, _ = product_files(prod)
    order = band_order(amp_f)
    amp = tifffile.imread(amp_f)
    ph = tifffile.imread(ph_f)
    if amp.ndim == 3 and amp.shape[-1] == len(order):     # (rows, cols, bands)
        amp, ph = np.moveaxis(amp, -1, 0), np.moveaxis(ph, -1, 0)
    out = {}
    for p in pols:
        i = order.index(p)
        out[p] = (amp[i] * np.exp(1j * ph[i])).astype(np.complex64)
    return out, order


def _xml_root(prod):
    for x in product_files(prod)[2]:
        try:
            root = etree.parse(str(x)).getroot()
        except etree.XMLSyntaxError:
            continue
        if root.xpath("//*[local-name()='orbitPass']"):
            return root
    raise FileNotFoundError("no annotation xml containing orbitPass")


def _first(node, names, cast=str):
    for n in names:
        hit = node.xpath(f".//*[local-name()='{n}']")
        if hit and hit[0].text and hit[0].text.strip():
            return cast(hit[0].text.strip())
    raise KeyError(f"none of {names} in the annotation, check the tag names against your product")


def _t64(s):
    return np.datetime64(s.strip().replace("Z", "")[:26])


def read_annotation(prod):
    """the bits of the annotation the geocoding and the wave chain need.
    the tag name lists are the spellings i've met; add to them if a product uses another"""
    r = _xml_root(prod)
    ann = {
        "orbit_pass": _first(r, ["orbitPass"]).upper(),
        "t0": _t64(_first(r, ["firstLineAzimuthTime", "azimuthTimeStart", "firstLineTime"])),
        "dt_az": _first(r, ["azimuthTimeInterval"], float),
        "tau0": _first(r, ["firstSampleSlantRangeTime", "slantRangeTimeStart", "rangeTimeStart"], float),
        "dt_rg": _first(r, ["rangeTimeInterval"], float),
        "prf": _first(r, ["prf", "pulseRepetitionFrequency"], float),
        "f0": _first(r, ["radarFrequency", "radarCarrierFrequency"], float),
    }
    try:
        ann["pols_xml"] = _first(r, ["polarisationList", "polarisations"]).split()
    except KeyError:
        ann["pols_xml"] = None
    try:
        ann["look"] = _first(r, ["lookSide", "antennaPointing", "lookDirection"]).lower()
    except KeyError:
        ann["look"] = "right"     # not in the annotation, assumed. a wrong side mirrors the geolocation, so check the corners against the KML

    t, pos, vel = [], [], []
    for sv in r.xpath("//*[local-name()='stateVector' or local-name()='orbitStateVector']"):
        tt = _first(sv, ["time", "utcTime", "azimuthTime"])
        try:
            p = [_first(sv, [a], float) for a in ("x", "y", "z")]
            v = [_first(sv, [a], float) for a in ("vx", "vy", "vz")]
        except KeyError:
            p = [float(x) for x in _first(sv, ["position"]).split()]
            v = [float(x) for x in _first(sv, ["velocity"]).split()]
        t.append(_t64(tt)); pos.append(p); vel.append(v)
    if len(t) < 4:
        raise KeyError("found fewer than 4 orbit state vectors, check the orbit tag names")
    ann["sv_t"] = (np.array(t) - ann["t0"]) / np.timedelta64(1, "us") * 1e-6   # seconds after first line
    ann["sv_pos"] = np.array(pos)
    ann["sv_vel"] = np.array(vel)
    return ann


# ----------------------------------------------------------------- geocoding

def ecef(lat, lon, h=0.0):
    la, lo = np.radians(lat), np.radians(lon)
    N = WGS_A / np.sqrt(1 - WGS_E2 * np.sin(la) ** 2)
    return np.array([(N + h) * np.cos(la) * np.cos(lo),
                     (N + h) * np.cos(la) * np.sin(lo),
                     (N * (1 - WGS_E2) + h) * np.sin(la)])


def _rough_latlon(p):
    return np.degrees(np.arcsin(p[2] / np.linalg.norm(p))), np.degrees(np.arctan2(p[1], p[0]))


def _sat_at(ann, row):
    if "_sp" not in ann:
        ann["_sp"] = CubicSpline(ann["sv_t"], ann["sv_pos"], axis=0)
        ann["_sv"] = CubicSpline(ann["sv_t"], ann["sv_vel"], axis=0)
    t = row * ann["dt_az"]
    return ann["_sp"](t), ann["_sv"](t)


def geocode_point(ann, row, col, guess=None, look="right"):
    """lat, lon, incidence of one pixel: where its slant range sphere meets the zero doppler
    plane on the ellipsoid (sea level, fine over the ocean)"""
    S, V = _sat_at(ann, row)
    vhat = V / np.linalg.norm(V)
    R = C * (ann["tau0"] + col * ann["dt_rg"]) / 2
    if guess is None:
        down = -S / np.linalg.norm(S)
        side = np.cross(down, vhat)
        side = side if look == "right" else -side
        H = np.linalg.norm(S) - 6371e3
        guess = np.array(_rough_latlon(S + H * down + np.sqrt(max(R**2 - H**2, 1)) * side))

    def f(x):
        d = ecef(x[0], x[1]) - S
        return [(np.linalg.norm(d) - R) / 1e3, np.dot(d, vhat) / 1e3]

    x = least_squares(f, guess, xtol=1e-12, ftol=1e-12).x
    P = ecef(*x)
    la, lo = np.radians(x)
    nrm = np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    los = (S - P) / np.linalg.norm(S - P)
    inc = np.degrees(np.arccos(np.clip(np.dot(los, nrm), -1, 1)))
    return x[0], x[1], inc


def geocode_grid(ann, shape, n=20, look="right"):
    """range-doppler geocoding on an n x n grid of tie points, interpolated in between later.
    much better than interpolating the 4 footprint corners (that was up to 55 km off inside the frame)"""
    rows = np.linspace(0, shape[0] - 1, n)
    cols = np.linspace(0, shape[1] - 1, n)
    lat, lon, inc = (np.zeros((n, n)) for _ in range(3))
    row_start = None
    for i, rr in enumerate(rows):
        guess = row_start
        for j, cc in enumerate(cols):
            lat[i, j], lon[i, j], inc[i, j] = geocode_point(ann, rr, cc, guess, look)
            guess = np.array([lat[i, j], lon[i, j]])
            if j == 0:
                row_start = guess
    lon = np.degrees(np.unwrap(np.radians(lon), axis=1))   # in case the frame crosses 180
    return rows, cols, lat, lon, inc


def grid_interp(rows, cols, arr, out_rows, out_cols):
    f = RegularGridInterpolator((rows, cols), arr, bounds_error=False, fill_value=None)
    R, Cc = np.meshgrid(out_rows, out_cols, indexing="ij")
    return f(np.column_stack([R.ravel(), Cc.ravel()])).reshape(R.shape)


def haversine_m(lat1, lon1, lat2, lon2):
    la1, la2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371e3 * np.arcsin(np.sqrt(a))


def pixel_spacing(rows, cols, lat, lon):
    """mean ground metres per pixel along azimuth (rows) and range (cols), from the tie grid"""
    d_az = haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:]) / np.diff(rows)[:, None]
    d_rg = haversine_m(lat[:, :-1], lon[:, :-1], lat[:, 1:], lon[:, 1:]) / np.diff(cols)[None, :]
    return float(np.mean(d_az)), float(np.mean(d_rg))


def track_heading(lat, lon):
    """compass bearing of increasing azimuth (rows), degrees clockwise from north"""
    la1, la2 = np.radians(lat[0].mean()), np.radians(lat[-1].mean())
    dlo = np.radians(lon[-1].mean() - lon[0].mean())
    y = np.sin(dlo) * np.cos(la2)
    x = np.cos(la1) * np.sin(la2) - np.sin(la1) * np.cos(la2) * np.cos(dlo)
    return np.degrees(np.arctan2(y, x)) % 360


# ----------------------------------------------------------------- land mask

def worldcover_water(lat, lon, res_m=100):
    """True where ESA WorldCover says water. class 80 = permanent water, 0 = no data
    which is what you get offshore. tiles that don't exist at all are open ocean -> water"""
    import rasterio
    from rasterio.windows import from_bounds
    from rasterio.enums import Resampling
    from rasterio.transform import rowcol
    from affine import Affine

    lat, lon = np.asarray(lat), np.asarray(lon)
    lon = (lon + 180) % 360 - 180
    water = np.ones(lat.shape, bool)
    tla, tlo = (np.floor(lat / 3) * 3).astype(int), (np.floor(lon / 3) * 3).astype(int)
    for a, b in sorted(set(zip(tla.ravel(), tlo.ravel()))):
        name = (f"ESA_WorldCover_10m_2021_v200_{'N' if a >= 0 else 'S'}{abs(a):02d}"
                f"{'E' if b >= 0 else 'W'}{abs(b):03d}_Map.tif")
        url = f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/{name}"
        sel = (tla == a) & (tlo == b)
        try:
            with rasterio.open(url) as src:
                w = from_bounds(lon[sel].min() - .01, lat[sel].min() - .01,
                                lon[sel].max() + .01, lat[sel].max() + .01, src.transform)
                w = w.round_offsets().round_lengths()
                sc = max(1, int(round(res_m / 10)))
                shp = (max(1, int(w.height // sc)), max(1, int(w.width // sc)))
                arr = src.read(1, window=w, out_shape=shp, resampling=Resampling.mode, boundless=True, fill_value=0)
                tr = src.window_transform(w) * Affine.scale(w.width / shp[1], w.height / shp[0])
        except rasterio.errors.RasterioIOError:
            continue
        r, c = rowcol(tr, lon[sel], lat[sel])
        r = np.clip(np.asarray(r), 0, shp[0] - 1); c = np.clip(np.asarray(c), 0, shp[1] - 1)
        water[sel] = np.isin(arr[r, c], [0, 80])
    return water


def near_land(land, d_az_m, d_rg_m, km):
    from scipy.ndimage import distance_transform_edt
    if not land.any():
        return np.zeros_like(land)
    return distance_transform_edt(~land, sampling=(d_az_m, d_rg_m)) < km * 1e3


# ----------------------------------------------------------------- multilook + circular stats

def multilook(a, naz, nrg):
    nr, nc = a.shape[0] // naz, a.shape[1] // nrg
    return a[:nr * naz, :nc * nrg].reshape(nr, naz, nc, nrg).mean(axis=(1, 3))


def ml_stats(slc, water, naz, nrg, block=None, every=5):
    """masked multilook of every second order product the later steps need.
    land pixels get weight 0 so they don't leak into the ocean cells.
    done in blocks of rows so a full frame fits in memory"""
    HH, HV, VH, VV = slc["HH"], slc["HV"], slc["VH"], slc["VV"]
    nr, nc = HH.shape[0] // naz, HH.shape[1] // nrg
    block = block or max(1, 2000 // naz)          # multilook rows per block
    keys = ["hh", "vv", "x", "hhvv", "hvvh", "t11", "t22", "t33", "t12", "t13", "t23", "z1221"]
    out = {k: np.full((nr, nc), np.nan, complex if k in ("hhvv", "hvvh", "t12", "t13", "t23", "z1221") else float) for k in keys}
    out["wfrac"] = np.zeros((nr, nc))
    nb = int(np.ceil(nr / block))
    for b in range(nb):
        r0, r1 = b * block, min(nr, (b + 1) * block)
        sl = slice(r0 * naz, r1 * naz)
        cs = slice(0, nc * nrg)
        w = water[sl, cs].astype(np.float32)
        hh, hv, vh, vv = HH[sl, cs], HV[sl, cs], VH[sl, cs], VV[sl, cs]
        k1 = (hh + vv) / np.sqrt(2); k2 = (hh - vv) / np.sqrt(2); k3 = (hv + vh) / np.sqrt(2)  # sqrt2 * mean(hv, vh)
        z12 = hv - vh + 1j * (hh + vv); z21 = vh - hv + 1j * (hh + vv)
        prods = {"hh": np.abs(hh) ** 2, "vv": np.abs(vv) ** 2, "x": (np.abs(hv) ** 2 + np.abs(vh) ** 2) / 2,
                 "hhvv": hh * np.conj(vv), "hvvh": hv * np.conj(vh),
                 "t11": np.abs(k1) ** 2, "t22": np.abs(k2) ** 2, "t33": np.abs(k3) ** 2,
                 "t12": k1 * np.conj(k2), "t13": k1 * np.conj(k3), "t23": k2 * np.conj(k3),
                 "z1221": z12 * np.conj(z21)}
        ws = multilook(w, naz, nrg) * naz * nrg
        ok = ws >= 0.5 * naz * nrg               # cells less than half water are dropped
        for k, p in prods.items():
            m = multilook(p * w, naz, nrg) * naz * nrg / np.where(ws > 0, ws, np.nan)
            out[k][r0:r1] = np.where(ok, m, np.nan)
        out["wfrac"][r0:r1] = ws / (naz * nrg)
        if b % every == 0 or b == nb - 1:
            print(f"  multilook block {b + 1}/{nb}")
    return out


def h_alpha(ml):
    """cloude pottier entropy H and mean alpha (deg) from the multilooked coherency matrix"""
    T = np.zeros(ml["t11"].shape + (3, 3), complex)
    T[..., 0, 0], T[..., 1, 1], T[..., 2, 2] = ml["t11"], ml["t22"], ml["t33"]
    T[..., 0, 1], T[..., 0, 2], T[..., 1, 2] = ml["t12"], ml["t13"], ml["t23"]
    T[..., 1, 0], T[..., 2, 0], T[..., 2, 1] = np.conj(ml["t12"]), np.conj(ml["t13"]), np.conj(ml["t23"])
    bad = ~np.isfinite(T).all(axis=(-1, -2))
    T[bad] = np.eye(3)
    lam, U = np.linalg.eigh(T)
    lam = np.clip(lam, 1e-30, None)
    p = lam / lam.sum(-1, keepdims=True)
    H = -(p * np.log(p) / np.log(3)).sum(-1)
    alpha = np.degrees((p * np.arccos(np.clip(np.abs(U[..., 0, :]), 0, 1))).sum(-1))
    H[bad], alpha[bad] = np.nan, np.nan
    return H, alpha


def parse_name(name):
    """swath, start time, track and frame out of a product name"""
    m = re.search(r"BIO_(S\d)_", name)
    t = re.search(r"_(\d{8}T\d{6})_", name)
    tr = re.search(r"_T(\d{3})_", name)
    fr = re.search(r"_F(\d{3})_", name)
    return {"swath": m.group(1) if m else "?",
            "start": dt.datetime.strptime(t.group(1), "%Y%m%dT%H%M%S") if t else None,
            "track": int(tr.group(1)) if tr else -1,
            "frame": int(fr.group(1)) if fr else -1}


def circ_mean(ph):
    return float(np.angle(np.nanmean(np.exp(1j * np.asarray(ph)))))


def circ_std(ph):
    R = np.abs(np.nanmean(np.exp(1j * np.asarray(ph))))
    return float(np.sqrt(max(0.0, -2 * np.log(R))))


def cpd_std_deg(N, coh):
    """eq 6 in the report, CPD precision for N looks at coherence coh"""
    return np.degrees(1 / np.sqrt(2 * N) * np.sqrt(1 - coh ** 2) / coh)


# ----------------------------------------------------------------- seawater + bragg

def eps_klein_swift(S, T, f):
    """complex seawater permittivity, eps = eps' - i eps'' (Klein & Swift 1977). S psu, T degC, f Hz"""
    S = np.asarray(S, float); T = np.asarray(T, float)
    w = 2 * np.pi * f
    eps_inf = 4.9
    eps_s = (87.134 - 1.949e-1 * T - 1.276e-2 * T**2 + 2.491e-4 * T**3) * \
            (1 + 1.613e-5 * S * T - 3.656e-3 * S + 3.210e-5 * S**2 - 4.232e-7 * S**3)
    tau = (1.768e-11 - 6.086e-13 * T + 1.104e-14 * T**2 - 8.111e-17 * T**3) * \
          (1 + 2.282e-5 * S * T - 7.638e-4 * S - 7.760e-6 * S**2 + 1.105e-8 * S**3)
    d = 25 - T
    sig25 = S * (0.182521 - 1.46192e-3 * S + 2.09324e-5 * S**2 - 1.28205e-7 * S**3)
    beta = 2.033e-2 + 1.266e-4 * d + 2.464e-6 * d**2 - S * (1.849e-5 - 2.551e-7 * d + 2.551e-8 * d**2)
    sig = sig25 * np.exp(-d * beta)
    return eps_inf + (eps_s - eps_inf) / (1 + 1j * w * tau) - 1j * sig / (w * EPS0)


def bragg_g(eps, th_deg):
    th = np.radians(th_deg)
    c, s = np.cos(th), np.sin(th)
    rt = np.sqrt(eps - s**2)
    ghh = (eps - 1) / (c + rt) ** 2
    # sign picked so HH and VV come back in phase off a flat sea (CPD near 0, like the data).
    # the other common form is just this times -1, which puts the CPD near 180
    gvv = (eps - 1) * ((eps - 1) * s**2 + eps) / (eps * c + rt) ** 2
    return ghh, gvv


def _ustar(U10):
    return U10 * np.sqrt((0.8 + 0.065 * U10) * 1e-3)


def elfouhaily(k, U10, omega_c=0.84):
    """omnidirectional elfouhaily et al. 1997 spectrum S(k), m^3/rad. omega_c = 0.84 is fully developed"""
    k = np.asarray(k, float)
    km, cm = 370.0, 0.23
    kp = G / U10**2 * omega_c**2
    cp = U10 / omega_c
    c = np.sqrt(G / k * (1 + (k / km) ** 2))
    us = _ustar(U10)
    a_p = 6e-3 * np.sqrt(omega_c)
    a_m = 1e-2 * (1 + np.log(us / cm)) if us < cm else 1e-2 * (1 + 3 * np.log(us / cm))
    gam = 1.7 if omega_c < 1 else 1.7 + 6 * np.log(omega_c)
    sig = 0.08 * (1 + 4 * omega_c ** -3)
    Lpm = np.exp(-1.25 * (kp / k) ** 2)
    Jp = gam ** np.exp(-(np.sqrt(k / kp) - 1) ** 2 / (2 * sig**2))
    Fp = Lpm * Jp * np.exp(-omega_c / np.sqrt(10) * (np.sqrt(k / kp) - 1))
    Fm = Lpm * Jp * np.exp(-0.25 * (k / km - 1) ** 2)
    B = 0.5 * a_p * (cp / c) * Fp + 0.5 * a_m * (cm / c) * Fm
    return np.clip(B, 0, None) / k**3     # alpha_m goes negative at very low wind, clip it


def elf_spread(k, U10, omega_c=0.84):
    km, cm = 370.0, 0.23
    cp = U10 / omega_c
    c = np.sqrt(G / k * (1 + (k / km) ** 2))
    return np.tanh(np.log(2) / 4 + 4 * (c / cp) ** 2.5 + 0.13 * _ustar(U10) / cm * (cm / c) ** 2.5)


def psi2d(k, U10, phi=0.0, omega_c=0.84):
    """2D spectrum at wavenumber k, phi = angle between look and wind (0 = looking upwind)"""
    return elfouhaily(k, U10, omega_c) / k * (1 + elf_spread(k, U10, omega_c) * np.cos(2 * phi)) / (2 * np.pi)


def pure_bragg(eps, th_deg, f, U10, phi=0.0):
    """sigma0 HH, VV (linear) and CPD in degrees, first order bragg on a flat sea"""
    k = 2 * np.pi * f / C
    th = np.radians(th_deg)
    ghh, gvv = bragg_g(eps, th_deg)
    pre = 16 * np.pi * k**4 * np.cos(th) ** 4 * psi2d(2 * k * np.sin(th), U10, phi)
    return pre * np.abs(ghh) ** 2, pre * np.abs(gvv) ** 2, np.degrees(np.angle(ghh * np.conj(gvv)))


def long_wave_mss(f, th_deg, U10, cut=3.0):
    """slope variance of the waves longer than the bragg wave (k < kB / cut) that do the tilting"""
    kb = 2 * (2 * np.pi * f / C) * np.sin(np.radians(th_deg))
    kk = np.logspace(-4, np.log10(kb / cut), 3000)
    return float(_trapz(kk**2 * elfouhaily(kk, U10), kk))


def tilted_bragg(eps, th_deg, f, U10, phi=0.0, n=31, cut=3.0):
    """two scale model, eq 19. bragg patches tilted along (local incidence) and across
    (rotates the patch by beta, eq 18, which mixes HH and VV). gaussian slopes, split evenly
    between the two directions, which is a simplification"""
    k = 2 * np.pi * f / C
    s = np.sqrt(long_wave_mss(f, th_deg, U10, cut) / 2)
    x, w = np.polynomial.hermite_e.hermegauss(n)
    w = w / w.sum()
    dpar = np.arctan(s * x)[:, None]
    dperp = np.arctan(s * x)[None, :]
    W2 = w[:, None] * w[None, :]
    thl = np.radians(th_deg) - dpar
    thl = np.clip(thl, np.radians(0.5), None)
    ghh, gvv = bragg_g(eps, np.degrees(thl))
    beta = np.arctan(np.tan(dperp) / np.sin(np.radians(th_deg)))
    c2, s2 = np.cos(beta) ** 2, np.sin(beta) ** 2
    shh = ghh * c2 + gvv * s2
    svv = gvv * c2 + ghh * s2
    pre = 16 * np.pi * k**4 * np.cos(thl) ** 4 * psi2d(2 * k * np.sin(thl), U10, phi) * W2
    s_hh = np.sum(pre * np.abs(shh) ** 2)
    s_vv = np.sum(pre * np.abs(svv) ** 2)
    cpd = np.degrees(np.angle(np.sum(pre * shh * np.conj(svv))))
    return s_hh, s_vv, cpd


def db(x):
    return 10 * np.log10(x)


# ----------------------------------------------------------------- ancillary readers

def nearest_time(ds, t):
    tname = "valid_time" if "valid_time" in ds.coords else "time"
    return ds.sel({tname: np.datetime64(t)}, method="nearest")


def interp_ll(da, lat, lon):
    """bilinear sample of a lat/lon field at points. copes with 0-360 longitudes and
    lat stored north to south (ERA5)"""
    da = da.squeeze(drop=True)
    latn = "latitude" if "latitude" in da.dims else "lat"
    lonn = "longitude" if "longitude" in da.dims else "lon"
    la, lo = da[latn].values, da[lonn].values
    v = np.asarray(da.transpose(latn, lonn).values, float)
    if la[0] > la[-1]:
        la, v = la[::-1], v[::-1]
    lon = np.asarray(lon, float)
    lon = np.mod(lon, 360) if lo.max() > 180 else (lon + 180) % 360 - 180
    f = RegularGridInterpolator((la, lo), v, bounds_error=False, fill_value=np.nan)
    return f(np.column_stack([np.ravel(lat), np.ravel(lon)])).reshape(np.shape(lat))


def imerg_nearest(folder, t):
    """half hourly IMERG granule closest to t. the time comes from the file name,
    Grid/time inside the file didn't always match the half hour"""
    import h5py
    t = np.datetime64(t)
    best = None
    for fp in Path(folder).glob("*3IMERG*.HDF5"):
        m = re.search(r"3IMERG\.(\d{8})-S(\d{6})", fp.name)
        if not m:
            continue
        tc = np.datetime64(dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")) + np.timedelta64(15, "m")
        if best is None or abs(tc - t) < abs(best[0] - t):
            best = (tc, fp)
    if best is None:
        raise FileNotFoundError(f"no IMERG HDF5 files in {folder}")
    with h5py.File(best[1], "r") as h:
        p = h["Grid/precipitation"][0].T          # stored (lon, lat)
        la, lo = h["Grid/lat"][:], h["Grid/lon"][:]
    p = np.where(p < 0, np.nan, p)
    return la, lo, p, best[0]


def sample_grid(la, lo, field, lat, lon):
    f = RegularGridInterpolator((la, lo), field, bounds_error=False, fill_value=np.nan, method="nearest")
    lon = (np.asarray(lon) + 180) % 360 - 180
    return f(np.column_stack([np.ravel(lat), np.ravel(lon)])).reshape(np.shape(lat))
