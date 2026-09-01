
########################################################
##
## Terrestrial planet civilization areas and sites
## based on mountains, rivers and climate of planet
##
## Python 3 source code
#
## 28.08.2026 v. 0000.0000 
##
########################################################

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from scipy.ndimage import (
    gaussian_filter,
    distance_transform_edt
)

from matplotlib.colors import ListedColormap

import noise


# ============================================================
# TERRESTRIC V8.3
# ============================================================
#
# Maailmangeneraattori
#
#     geometria
#         ↓
#     mantereet
#         ↓
#     tektoniset laatat
#         ↓
#     vuoristot / riftit / tulivuoret
#         ↓
#     korkeus
#         ↓
#     eroosio
#         ↓
#     joet
#         ↓
#     globaalit tuulet
#         ↓
#     merivirrat
#         ↓
#     lämpötila
#         ↓
#     merikosteus
#         ↓
#     TUULEN KULJETTAMA KOSTEUS
#         ↓
#     vuoriston tuulenpuoli
#         ↓
#     OROGRAFINEN SADE
#         ↓
#     VUORISTON SADEVARJO
#         ↓
#     monsuuni
#         ↓
#     vuodenajat
#         ↓
#     biomit
#         ↓
#     resurssit
#         ↓
#     sivilisaatio
#         ↓
#     kauppa
#
# ============================================================


# ============================================================
# ASETUKSET
# ============================================================

WIDTH = 720
HEIGHT = 360

#SEED=42
#SEED = 32


#SEED=12

SEED=777

rng = np.random.default_rng(SEED)


# ============================================================
# MAAILMAN RAKENNE
# ============================================================

NUM_CONTINENTS = 6
NUM_ISLANDS = 55
NUM_PLATES = 13

#SEA_LEVEL = 0.45
#SEA_LEVEL = 0.50
SEA_LEVEL = 0.35


# ============================================================
# ULKOINEN DEM — OHITUSKAISTA
# ============================================================
#
# False = TERRESTRIC generoi maaston normaalisti
# True  = luetaan valmis DEM/heightmap levyltä ja ohitetaan
#         Telluruksen oma geologinen maastogenerointi.
#
# DEM-tiedosto voi olla esimerkiksi:
#   PNG / TIFF / JPG / BMP / WEBP
#   8-bit grayscale
#   16-bit grayscale
#   RGB / RGBA
#
# HEIGHTMAP: arvot 0..1 tai 8/16-bit -> 0..1, SEA_LEVEL
#            määrää merenpinnan (oletus 0.30).
# ELEVATION: raakaarvot tulkitaan metreiksi; 0 m = merenpinta
#            ja arvot normalisoidaan Telluruksen 0..1-avaruuteen.

USE_EXTERNAL_DEM = False

EXTERNAL_DEM_PATH = r"orogen1.png"

EXTERNAL_DEM_MODE = "HEIGHTMAP"

# Jos DEM:n mittasuhde on eri kuin Telluruksen WIDTH x HEIGHT,
# se skaalataan automaattisesti tähän kokoon.
EXTERNAL_DEM_RESAMPLING = "BILINEAR"

NUM_RIVERS = 260
NUM_CIV_CENTERS = 90


# ============================================================
# TEKTONIIKAN SÄÄTÖ
# ============================================================

PLATE_DISTORTION = 4.5

MOUNTAIN_STRENGTH = 0.34

MOUNTAIN_BRANCH_STRENGTH = 0.35

TERRAIN_NOISE_STRENGTH = 0.10


# ============================================================
# ILMASTON SÄÄTÖ
# ============================================================

OCEAN_INFLUENCE_SCALE = 35.0

CONTINENTALITY_RAIN_EFFECT = 0.40

REFERENCE_OCEAN_COVERAGE = 0.70

OCEAN_COVERAGE_RAIN_EFFECT = 0.45

GLOBAL_RAINFALL_SCALE = 1.25

MARITIME_MOISTURE_STRENGTH = 1.0


# ============================================================
# TUULEN KOSTEUSKULJETUS
# ============================================================

WIND_MOISTURE_STRENGTH = 850

WIND_DRYING_STRENGTH = 650

WIND_MOISTURE_DISTANCE = 42.0

WIND_MOISTURE_DECAY = 1.35


# ============================================================
# MONSUUNI
# ============================================================

MONSOON_RAIN_STRENGTH = 900


# ============================================================
# OROGRAFIA
# ============================================================

OROGRAPHIC_RAIN_STRENGTH = 2500

OROGRAPHIC_DRYING_STRENGTH = 1000


# ============================================================
# UUSI VUORISTON SADEVARJO
# ============================================================

RAIN_SHADOW_STRENGTH = 1150

RAIN_SHADOW_DISTANCE = 42.0

RAIN_SHADOW_DECAY = 1.30

RAIN_SHADOW_MOUNTAIN_THRESHOLD = 0.58

RAIN_SHADOW_WIND_STRENGTH = 1.25

RAIN_SHADOW_MAX_STEPS = 30

RAIN_SHADOW_STEP = 2.0


# ============================================================
# KOORDINAATIT
# ============================================================

longitude = np.linspace(
    -180,
    180,
    WIDTH,
    endpoint=False
)

latitude = np.linspace(
    90,
    -90,
    HEIGHT
)

LON, LAT = np.meshgrid(
    longitude,
    latitude
)


# ============================================================
# PALLOKOORDINAATIT
# ============================================================

theta = np.radians(
    90 - LAT
)

phi = np.radians(
    LON
)

X = (
    np.sin(theta)
    * np.cos(phi)
)

Y = (
    np.sin(theta)
    * np.sin(phi)
)

Z = np.cos(theta)


# ============================================================
# APUFUNKTIOT
# ============================================================

def normalize(array):

    array = np.asarray(
        array,
        dtype=float
    )

    amin = np.nanmin(array)
    amax = np.nanmax(array)

    if amax - amin < 1e-12:

        return np.zeros_like(
            array
        )

    return (
        array - amin
    ) / (
        amax - amin
    )


# ============================================================
# PALLOETÄISYYS
# ============================================================

def spherical_distance(
    lon1,
    lat1,
    lon2,
    lat2
):

    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)

    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)

    value = (
        np.sin(lat1)
        * np.sin(lat2)
        +
        np.cos(lat1)
        * np.cos(lat2)
        * np.cos(
            lon1 - lon2
        )
    )

    value = np.clip(
        value,
        -1,
        1
    )

    return np.degrees(
        np.arccos(value)
    )


# ============================================================
# PALLOLLA JATKUVA PERLIN-NOISE
# ============================================================

def spherical_noise(
    frequencies=None,
    amplitudes=None,
    base_offset=0
):

    if frequencies is None:

        frequencies = [
            1.8,
            3.6,
            7.2,
            14.4
        ]

    if amplitudes is None:

        amplitudes = [
            1.0,
            0.45,
            0.20,
            0.08
        ]

    field = np.zeros(
        (HEIGHT, WIDTH),
        dtype=float
    )

    for frequency, amplitude in zip(
        frequencies,
        amplitudes
    ):

        layer = np.zeros_like(
            field
        )

        for y in range(
            HEIGHT
        ):

            for x in range(
                WIDTH
            ):

                layer[y, x] = noise.pnoise3(
                    X[y, x] * frequency,
                    Y[y, x] * frequency,
                    Z[y, x] * frequency,
                    octaves=1,
                    persistence=0.5,
                    lacunarity=2.0,
                    repeatx=1024,
                    repeaty=1024,
                    repeatz=1024,
                    base=SEED + base_offset
                )

        field += (
            layer * amplitude
        )

    return normalize(
        field
    )


# ============================================================
# ORGANIC NOISE
# ============================================================

def organic_noise():

    n1 = gaussian_filter(
        rng.random(
            (HEIGHT, WIDTH)
        ),
        sigma=(18, 35),
        mode=("nearest", "wrap")
    )

    n2 = gaussian_filter(
        rng.random(
            (HEIGHT, WIDTH)
        ),
        sigma=(8, 18),
        mode=("nearest", "wrap")
    )

    n3 = gaussian_filter(
        rng.random(
            (HEIGHT, WIDTH)
        ),
        sigma=(3, 7),
        mode=("nearest", "wrap")
    )

    n1 = normalize(n1)
    n2 = normalize(n2)
    n3 = normalize(n3)

    return (
        n1 * 0.60
        +
        n2 * 0.30
        +
        n3 * 0.10
    )


# ============================================================
# ORGAANINEN MANNER
# ============================================================

def create_continent(
    center_lon,
    center_lat,
    size
):

    distance = spherical_distance(
        LON,
        LAT,
        center_lon,
        center_lat
    )

    lon_diff = np.radians(
        LON - center_lon
    )

    lat_diff = np.radians(
        LAT - center_lat
    )

    angle = np.arctan2(
        lat_diff,
        lon_diff
    )

    coast_noise = organic_noise()

    coast_variation = (
        coast_noise - 0.5
    ) * size * 0.75

    radius = (
        size
        +
        coast_variation
    )

    stretch = rng.uniform(
        0.65,
        1.8
    )

    directional = (
        1
        +
        0.30
        *
        np.sin(
            angle
            *
            rng.integers(
                2,
                5
            )
            +
            rng.random() * 6
        )
    )

    radius *= directional

    radial_distance = (
        distance
        *
        (
            1
            +
            0.25
            * np.cos(angle * 2)
        )
        /
        stretch
    )

    continent = (
        radial_distance < radius
    ).astype(float)

    edge = np.exp(
        -(
            np.maximum(
                radial_distance - radius,
                0
            )
            / 2.5
        ) ** 2
    )

    continent += (
        edge * 0.14
    )

    return continent


# ============================================================
# MANTEREET
# ============================================================

def generate_continents():

    land = np.zeros(
        (HEIGHT, WIDTH)
    )

    data = []

    for _ in range(
        NUM_CONTINENTS
    ):

        lon = rng.uniform(
            -180,
            180
        )

        lat = rng.uniform(
            -60,
            60
        )

        size = rng.uniform(
            18,
            45
        )

        continent = create_continent(
            lon,
            lat,
            size
        )

        land += continent

        data.append(
            (
                lon,
                lat,
                size
            )
        )

    broad = gaussian_filter(
        rng.random(
            (HEIGHT, WIDTH)
        ),
        sigma=(12, 25),
        mode=("nearest", "wrap")
    )

    broad = normalize(
        broad
    )

    land += (
        broad - 0.5
    ) * 0.20

    land = gaussian_filter(
        land,
        sigma=(2, 4),
        mode=("nearest", "wrap")
    )

    land = normalize(
        land
    )

    return land, data


# ============================================================
# SAARET
# ============================================================

def add_islands(
    land
):

    for _ in range(
        NUM_ISLANDS
    ):

        lon = rng.uniform(
            -180,
            180
        )

        lat = rng.uniform(
            -75,
            75
        )

        size = rng.uniform(
            1,
            5
        )

        distance = spherical_distance(
            LON,
            LAT,
            lon,
            lat
        )

        island = np.exp(
            -(
                distance / size
            ) ** 4
        )

        land += (
            island
            *
            rng.uniform(
                0.25,
                0.65
            )
        )

    return normalize(
        land
    )


# ============================================================
# TEKTONISET LAATAT
# ============================================================

def create_plates():

    plates = []

    for i in range(
        NUM_PLATES
    ):

        direction = rng.uniform(
            0,
            2 * np.pi
        )

        speed = rng.uniform(
            0.15,
            1.0
        )

        crust_type = (
            "continental"
            if rng.random() < 0.48
            else "oceanic"
        )

        plates.append({

            "id":
                i,

            "lon":
                rng.uniform(
                    -180,
                    180
                ),

            "lat":
                rng.uniform(
                    -75,
                    75
                ),

            "direction":
                direction,

            "speed":
                speed,

            "type":
                crust_type
        })

    return plates


# ============================================================
# LAATTAMAP
# ============================================================

def create_plate_map(
    plates
):

    distances = np.zeros(
        (
            NUM_PLATES,
            HEIGHT,
            WIDTH
        )
    )

    for i, plate in enumerate(
        plates
    ):

        distances[i] = spherical_distance(
            LON,
            LAT,
            plate["lon"],
            plate["lat"]
        )

    low_noise = spherical_noise(
        frequencies=[
            0.8,
            1.6,
            3.2
        ],

        amplitudes=[
            1.0,
            0.35,
            0.10
        ],

        base_offset=1000
    )

    low_noise = (
        low_noise - 0.5
    ) * 2

    medium_noise = spherical_noise(
        frequencies=[
            2.0,
            4.0
        ],

        amplitudes=[
            1.0,
            0.25
        ],

        base_offset=2000
    )

    medium_noise = (
        medium_noise - 0.5
    ) * 2

    distortion = (
        low_noise * 0.78
        +
        medium_noise * 0.22
    )

    for i in range(
        NUM_PLATES
    ):

        plate_bias = rng.uniform(
            0.75,
            1.25
        )

        distances[i] += (
            distortion
            *
            PLATE_DISTORTION
            *
            plate_bias
        )

    return np.argmin(
        distances,
        axis=0
    )


# ============================================================
# LAATTOJEN RAJAT
# ============================================================

def find_boundaries(
    plate_map
):

    boundary = np.zeros(
        plate_map.shape,
        dtype=bool
    )

    boundary[:-1, :] |= (
        plate_map[:-1, :]
        !=
        plate_map[1:, :]
    )

    boundary[:, :-1] |= (
        plate_map[:, :-1]
        !=
        plate_map[:, 1:]
    )

    boundary[:, 0] |= (
        plate_map[:, 0]
        !=
        plate_map[:, -1]
    )

    return boundary


# ============================================================
# LAATAN NOPEUS
# ============================================================

def plate_velocity(
    plate
):

    return np.array([

        np.cos(
            plate["direction"]
        )
        *
        plate["speed"],

        np.sin(
            plate["direction"]
        )
        *
        plate["speed"]

    ])


# ============================================================
# RAJATYYPIT
# ============================================================

def classify_boundaries(
    plate_map,
    plates
):

    boundary_type = np.zeros(
        plate_map.shape,
        dtype=np.uint8
    )

    for y in range(
        1,
        HEIGHT - 1
    ):

        for x in range(
            WIDTH
        ):

            a = plate_map[
                y,
                x
            ]

            b = plate_map[
                y,
                (x + 1) % WIDTH
            ]

            if a == b:
                continue

            va = plate_velocity(
                plates[a]
            )

            vb = plate_velocity(
                plates[b]
            )

            relative = (
                vb - va
            )

            normal = np.array([
                1.0,
                0.0
            ])

            value = np.dot(
                relative,
                normal
            )

            if value < -0.15:

                boundary_type[
                    y,
                    x
                ] = 1

            elif value > 0.15:

                boundary_type[
                    y,
                    x
                ] = 2

            else:

                boundary_type[
                    y,
                    x
                ] = 3

    for y in range(
        HEIGHT - 1
    ):

        for x in range(
            WIDTH
        ):

            a = plate_map[
                y,
                x
            ]

            b = plate_map[
                y + 1,
                x
            ]

            if a == b:
                continue

            va = plate_velocity(
                plates[a]
            )

            vb = plate_velocity(
                plates[b]
            )

            relative = (
                vb - va
            )

            normal = np.array([
                0.0,
                1.0
            ])

            value = np.dot(
                relative,
                normal
            )

            if value < -0.15:

                boundary_type[
                    y,
                    x
                ] = 1

            elif value > 0.15:

                boundary_type[
                    y,
                    x
                ] = 2

            else:

                boundary_type[
                    y,
                    x
                ] = 3

    return boundary_type


# ============================================================
# ULKOISEN DEM:N LUKU
# ============================================================

def load_external_dem(
    path,
    target_width,
    target_height,
    mode="HEIGHTMAP"
):
    """
    Lukee ulkoisen DEM:n / heightmapin ja palauttaa Telluruksen
    sisäisessä 0..1-korkeusavaruudessa olevan float-arrayn.

    Tuetut tavalliset kuvamuodot:
        - 8-bit grayscale
        - 16-bit grayscale
        - RGB / RGBA
        - muut Pillow'n avaamat numeeriset kuvamuodot

    HEIGHTMAP:
        pikseliarvo -> 0..1 suoraan bittisyvyyden perusteella.
        Merenpinta on tällöin SEA_LEVEL (oletus 0.30).

    ELEVATION:
        pikseliarvot tulkitaan korkeuksina metreinä.
        0 m asetetaan merenpinnaksi ja koko korkeusalue
        muunnetaan Telluruksen 0..1-avaruuteen.
        SEA_LEVEL päivitetään vastaamaan 0 metriä.
    """

    image = Image.open(path)

    print(
        "    DEM-tiedosto:",
        path
    )

    print(
        "    DEM alkuperäinen koko:",
        image.size
    )

    print(
        "    DEM formaatti:",
        image.mode
    )

    array = np.asarray(
        image
    )

    # --------------------------------------------------------
    # RGB / RGBA -> luminanssi
    # --------------------------------------------------------

    if array.ndim == 3:

        if array.shape[2] < 3:
            raise ValueError(
                "Ulkoisen DEM:n värikanavia ei voitu tulkita."
            )

        # Säilytetään mahdollinen 16-bit tarkkuus ennen
        # luminanssilaskua. Alpha-kanavaa ei käytetä.
        array = (
            array[..., 0].astype(np.float64) * 0.299
            +
            array[..., 1].astype(np.float64) * 0.587
            +
            array[..., 2].astype(np.float64) * 0.114
        )

    elif array.ndim != 2:

        raise ValueError(
            f"DEM:n odotettiin olevan 2D-harmaasävy tai RGB-kuva; "
            f"saatu shape={array.shape}"
        )

    array = array.astype(
        np.float64,
        copy=False
    )

    # --------------------------------------------------------
    # NaN / inf -tarkistus
    # --------------------------------------------------------

    finite = np.isfinite(
        array
    )

    if not np.any(finite):

        raise ValueError(
            "DEM ei sisällä yhtään kelvollista numeerista arvoa."
        )

    raw_min = float(
        np.nanmin(array)
    )

    raw_max = float(
        np.nanmax(array)
    )

    print(
        "    DEM raaka-arvot:",
        round(raw_min, 3),
        "...",
        round(raw_max, 3)
    )

    mode = str(mode).upper().strip()

    # --------------------------------------------------------
    # HEIGHTMAP
    # --------------------------------------------------------

    if mode == "HEIGHTMAP":

        # RGB ja 8-bit grayscale
        if raw_min >= 0 and raw_max <= 255:

            source_max = 255.0

        # 16-bit grayscale
        elif raw_min >= 0 and raw_max <= 65535:

            source_max = 65535.0

        else:

            # Esimerkiksi float-DEM, joka on jo 0..1.
            if raw_min >= 0.0 and raw_max <= 1.0:
                source_max = 1.0
            else:
                source_max = raw_max

        elevation = array / source_max

        elevation = np.nan_to_num(
            elevation,
            nan=0.0,
            posinf=1.0,
            neginf=0.0
        )

        elevation = np.clip(
            elevation,
            0.0,
            1.0
        )

        print(
            "    DEM-tila: HEIGHTMAP"
        )

        print(
            "    Sisäinen korkeus: 0.000 ... 1.000"
        )

        print(
            "    Merenpinta: TERRESTRIC SEA_LEVEL =",
            round(float(SEA_LEVEL), 3)
        )

    # --------------------------------------------------------
    # ELEVATION / METRIT
    # --------------------------------------------------------

    elif mode == "ELEVATION":

        # Tässä tilassa pikseliarvo 0 tarkoittaa merenpintaa.
        # Negatiiviset arvot ovat merenpinnan alapuolella.
        # Korkein positiivinen arvo saa sisäisen arvon 1.

        valid = np.isfinite(
            array
        )

        positive_max = float(
            np.max(
                array[valid]
            )
        )

        negative_min = float(
            np.min(
                array[valid]
            )
        )

        if positive_max <= 0:

            raise ValueError(
                "ELEVATION-DEM: aineistossa ei ole yhtään "
                "merenpinnan yläpuolista korkeutta."
            )

        # Normalisoidaan niin, että 0 metriä vastaa 0.0 ja
        # korkein maanpinnan kohta vastaa 1.0. Merialueet
        # jäävät välille 0..SEA_LEVEL.
        elevation = array / positive_max

        # Merisyvyys ei saa tehdä normaalista maastosta negatiivista
        # sisäistä korkeutta. Syvyydet pakataan 0..SEA_LEVEL-välille.
        sea_mask = elevation < 0.0

        if negative_min < 0.0:

            sea_depth = np.clip(
                array / abs(negative_min),
                -1.0,
                0.0
            )

            elevation[sea_mask] = (
                1.0 + sea_depth[sea_mask]
            ) * float(SEA_LEVEL)

        elevation[~np.isfinite(elevation)] = 0.0

        elevation = np.clip(
            elevation,
            0.0,
            1.0
        )

        # ELEVATION-tilassa merenpinta on nyt täsmälleen 0.0
        # raakadatassa ja sen tulee vastata sisäistä SEA_LEVEL-arvoa.
        # Koska maan arvot ovat välillä 0..1, käytämme 0 m = SEA_LEVEL.
        # Siirrämme maan 0..1-avaruuteen tämän ympärille.
        sea_level_norm = float(SEA_LEVEL)

        above_sea = array >= 0.0
        elevation[above_sea] = (
            sea_level_norm
            +
            (array[above_sea] / positive_max)
            * (1.0 - sea_level_norm)
        )

        elevation = np.clip(
            elevation,
            0.0,
            1.0
        )

        print(
            "    DEM-tila: ELEVATION (metrit)"
        )

        print(
            "    Merenpinta: 0 m -> TERRESTRIC",
            round(float(SEA_LEVEL), 3)
        )

        print(
            "    Korkein maanpinnan arvo:",
            round(positive_max, 3),
            "m"
        )

    else:

        raise ValueError(
            "EXTERNAL_DEM_MODE pitää olla 'HEIGHTMAP' "
            "tai 'ELEVATION'."
        )

    # --------------------------------------------------------
    # RESAMPLING TELLURUKSEN KOKOON
    # --------------------------------------------------------

    # Pillow'n mode F säilyttää float-arvot ilman 8-bit muunnosta.
    image_float = Image.fromarray(
        np.float32(elevation),
        mode="F"
    )

    resampling_name = str(
        EXTERNAL_DEM_RESAMPLING
    ).upper().strip()

    resampling_methods = {
        "NEAREST": Image.Resampling.NEAREST,
        "BILINEAR": Image.Resampling.BILINEAR,
        "BICUBIC": Image.Resampling.BICUBIC,
        "LANCZOS": Image.Resampling.LANCZOS
    }

    resampling = resampling_methods.get(
        resampling_name,
        Image.Resampling.BILINEAR
    )

    image_float = image_float.resize(
        (
            int(target_width),
            int(target_height)
        ),
        resample=resampling
    )

    elevation = np.asarray(
        image_float,
        dtype=np.float64
    )

    elevation = np.nan_to_num(
        elevation,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    elevation = np.clip(
        elevation,
        0.0,
        1.0
    )

    print(
        "    DEM Telluruksen koko:",
        f"{target_width} x {target_height}"
    )

    print(
        "    DEM uudelleenskaalaus:",
        resampling_name
    )

    return elevation


# ============================================================
# PERUSKORKEUS
# ============================================================

def create_base_elevation(
    land
):

    elevation = np.where(

        land > SEA_LEVEL,

        0.47
        +
        land * 0.23,

        0.16
        +
        land * 0.08
    )

    return elevation


# ============================================================
# KRATONIT
# ============================================================

def add_cratons(
    elevation,
    land
):

    land_mask = (
        land > SEA_LEVEL
    )

    yy, xx = np.indices(
        elevation.shape
    )

    ys, xs = np.where(
        land_mask
    )

    if len(xs) == 0:
        return elevation

    for _ in range(
        12
    ):

        i = rng.integers(
            len(xs)
        )

        cy = ys[i]
        cx = xs[i]

        ry = rng.uniform(
            12,
            30
        )

        rx = rng.uniform(
            20,
            60
        )

        d = (
            ((yy - cy) / ry) ** 2
            +
            ((xx - cx) / rx) ** 2
        )

        craton = np.exp(
            -(d ** 2)
        )

        craton[
            ~land_mask
        ] = 0

        elevation += (
            craton * 0.06
        )

    return elevation


# ============================================================
# VUORISTON SELKÄRANKA
# ============================================================

def create_mountain_ridge(
    boundary_type,
    land
):

    convergent = (
        boundary_type == 1
    )

    ridge = gaussian_filter(
        convergent.astype(float),
        sigma=(1.2, 3.5),
        mode=("nearest", "wrap")
    )

    ridge = normalize(
        ridge
    )

    long_noise = spherical_noise(
        frequencies=[
            0.7,
            1.4,
            2.8
        ],

        amplitudes=[
            1.0,
            0.30,
            0.10
        ],

        base_offset=3000
    )

    long_noise = (
        0.55
        +
        0.85
        *
        long_noise
    )

    ridge *= (
        long_noise
    )

    branch_noise = spherical_noise(
        frequencies=[
            1.2,
            2.4,
            4.8
        ],

        amplitudes=[
            1.0,
            0.35,
            0.10
        ],

        base_offset=4000
    )

    branches = (
        gaussian_filter(
            convergent.astype(float),
            sigma=(5, 12),
            mode=("nearest", "wrap")
        )
        *
        np.clip(
            branch_noise - 0.42,
            0,
            1
        )
    )

    branches = normalize(
        branches
    )

    ridge += (
        branches
        *
        MOUNTAIN_BRANCH_STRENGTH
    )

    ridge[
        land < SEA_LEVEL
    ] *= 0.18

    return normalize(
        ridge
    )


# ============================================================
# VUORISTOT
# ============================================================

def add_mountains(
    elevation,
    land,
    boundary_type
):

    ridge = create_mountain_ridge(
        boundary_type,
        land
    )

    terrain_noise = spherical_noise(
        frequencies=[
            3.0,
            6.0,
            12.0
        ],

        amplitudes=[
            1.0,
            0.35,
            0.10
        ],

        base_offset=5000
    )

    terrain_noise = (
        terrain_noise ** 1.8
    )

    mountain_height = (
        ridge
        *
        (
            0.06
            +
            terrain_noise
            *
            MOUNTAIN_STRENGTH
        )
    )

    elevation += (
        mountain_height
    )

    peak_noise = spherical_noise(
        frequencies=[
            4.0,
            8.0,
            16.0
        ],

        amplitudes=[
            1.0,
            0.35,
            0.08
        ],

        base_offset=6000
    )

    peaks = (
        ridge
        *
        np.clip(
            peak_noise - 0.60,
            0,
            1
        )
    )

    peaks = gaussian_filter(
        peaks,
        sigma=(1.0, 2.0),
        mode=("nearest", "wrap")
    )

    elevation += (
        peaks
        * 0.12
    )

    return elevation


# ============================================================
# TULIVUORIVYÖHYKKEET
# ============================================================

def add_volcanic_arcs(
    elevation,
    land,
    boundary_type
):

    convergent = (
        boundary_type == 1
    )

    field = gaussian_filter(
        convergent.astype(float),
        sigma=(2, 5),
        mode=("nearest", "wrap")
    )

    volcano_noise = spherical_noise(
        frequencies=[
            3,
            6,
            12
        ],

        amplitudes=[
            1,
            0.3,
            0.1
        ],

        base_offset=7000
    )

    volcanoes = (
        field
        *
        (
            volcano_noise > 0.72
        )
    )

    elevation += (
        volcanoes
        * 0.10
    )

    return elevation


# ============================================================
# RIFTIT
# ============================================================

def add_rifts(
    elevation,
    boundary_type
):

    divergent = (
        boundary_type == 2
    )

    field = gaussian_filter(
        divergent.astype(float),
        sigma=(2, 5),
        mode=("nearest", "wrap")
    )

    elevation -= (
        field
        * 0.055
    )

    return elevation


# ============================================================
# EROOSIO
# ============================================================

def erode_terrain(
    elevation
):

    smooth = gaussian_filter(
        elevation,
        sigma=(2, 5),
        mode=("nearest", "wrap")
    )

    high = np.clip(
        elevation - 0.55,
        0,
        1
    )

    strength = (
        high * 0.25
    )

    return (
        elevation * (1 - strength)
        +
        smooth * strength
    )


# ============================================================
# JOET
# ============================================================

def generate_rivers(
    elevation,
    land,
    number=260
):

    rivers = np.zeros(
        elevation.shape,
        dtype=float
    )

    land_mask = (
        elevation >= SEA_LEVEL
    )

    candidates = np.argwhere(
        land_mask
        &
        (elevation > 0.62)
    )

    if len(candidates) == 0:
        return rivers

    for _ in range(
        number
    ):

        y, x = candidates[
            rng.integers(
                len(candidates)
            )
        ]

        visited = set()

        for step in range(
            900
        ):

            if (
                y,
                x
            ) in visited:

                break

            visited.add(
                (
                    y,
                    x
                )
            )

            rivers[
                y,
                x
            ] += 1

            current = elevation[
                y,
                x
            ]

            best_y = y
            best_x = x
            best_height = current

            for dy in (
                -1,
                0,
                1
            ):

                for dx in (
                    -1,
                    0,
                    1
                ):

                    if (
                        dy == 0
                        and
                        dx == 0
                    ):
                        continue

                    ny = y + dy

                    nx = (
                        x + dx
                    ) % WIDTH

                    if (
                        ny < 0
                        or
                        ny >= HEIGHT
                    ):
                        continue

                    h = elevation[
                        ny,
                        nx
                    ]

                    if (
                        h < best_height
                    ):

                        best_height = h
                        best_y = ny
                        best_x = nx

            if (
                best_y == y
                and
                best_x == x
            ):
                break

            y = best_y
            x = best_x

            if not land_mask[
                y,
                x
            ]:
                break

    rivers = gaussian_filter(
        rivers,
        sigma=(0.5, 0.8),
        mode=("nearest", "wrap")
    )

    return normalize(
        rivers
    )


# ============================================================
# GLOBAALIT TUULET
# ============================================================

def calculate_global_winds():

    abs_lat = np.abs(
        LAT
    )

    wind_u = np.zeros_like(
        LAT,
        dtype=float
    )

    wind_v = np.zeros_like(
        LAT,
        dtype=float
    )

    trades = np.clip(
        1 - abs_lat / 35,
        0,
        1
    )

    wind_u -= (
        trades * 1.0
    )

    wind_v += np.where(
        LAT > 0,
        -trades * 0.35,
        trades * 0.35
    )

    westerlies = np.exp(
        -(
            (abs_lat - 45)
            / 16
        ) ** 2
    )

    wind_u += (
        westerlies * 1.0
    )

    wind_v += np.where(
        LAT > 0,
        westerlies * 0.12,
        -westerlies * 0.12
    )

    polar = np.clip(
        (abs_lat - 60) / 30,
        0,
        1
    )

    wind_u -= (
        polar * 0.55
    )

    wind_v += np.where(
        LAT > 0,
        -polar * 0.15,
        polar * 0.15
    )

    calm_equator = np.exp(
        -(
            LAT / 7
        ) ** 2
    )

    wind_u *= (
        1
        -
        calm_equator * 0.55
    )

    wind_v *= (
        1
        -
        calm_equator * 0.35
    )

    return wind_u, wind_v


# ============================================================
# MEREN LÄMPÖTILA
# ============================================================

def calculate_ocean_temperature():

    ocean_temperature = (
        28
        *
        np.cos(
            np.radians(LAT)
        ) ** 0.65
        -
        2
    )

    ocean_temperature -= (
        np.maximum(
            np.abs(LAT) - 55,
            0
        )
        * 0.18
    )

    return ocean_temperature


# ============================================================
# MERIVIRRAT
# ============================================================

def calculate_ocean_currents(
    wind_u,
    wind_v
):

    current_u = wind_u.copy()
    current_v = wind_v.copy()

    coriolis_strength = (
        np.sin(
            np.radians(LAT)
        )
    )

    turn = 0.38

    rotated_u = (
        current_u
        -
        coriolis_strength
        * current_v
        * turn
    )

    rotated_v = (
        current_v
        +
        coriolis_strength
        * current_u
        * turn
    )

    current_u = rotated_u
    current_v = rotated_v

    tropical = np.exp(
        -(
            LAT / 28
        ) ** 4
    )

    current_u += (
        -0.30
        * tropical
    )

    temperate = np.exp(
        -(
            (np.abs(LAT) - 40)
            / 18
        ) ** 2
    )

    current_u += (
        0.25
        * temperate
    )

    current_u = gaussian_filter(
        current_u,
        sigma=(8, 15),
        mode=("nearest", "wrap")
    )

    current_v = gaussian_filter(
        current_v,
        sigma=(8, 15),
        mode=("nearest", "wrap")
    )

    return current_u, current_v


# ============================================================
# MERIVIRTOJEN LÄMPÖVAIKUTUS
# ============================================================

def calculate_ocean_current_temperature(
    ocean_temperature,
    current_u,
    current_v
):

    poleward_flow = (
        current_v
        *
        np.sign(LAT)
    )

    warm_transport = gaussian_filter(
        poleward_flow,
        sigma=(12, 20),
        mode=("nearest", "wrap")
    )

    equatorward_flow = (
        -poleward_flow
    )

    cold_transport = gaussian_filter(
        equatorward_flow,
        sigma=(10, 18),
        mode=("nearest", "wrap")
    )

    ocean_effect = (
        warm_transport * 5.0
        -
        cold_transport * 3.0
    )

    tropical = np.exp(
        -(
            LAT / 20
        ) ** 2
    )

    ocean_effect *= (
        1
        -
        tropical * 0.25
    )

    return ocean_effect


# ============================================================
# MANTEREISUUS
# ============================================================

def calculate_continentality(
    land_mask
):

    distance_from_ocean = (
        distance_transform_edt(
            land_mask
        )
    )

    continentality = (
        1
        -
        np.exp(
            -distance_from_ocean
            /
            OCEAN_INFLUENCE_SCALE
        )
    )

    continentality *= (
        land_mask
    )

    return np.clip(
        continentality,
        0,
        1
    )


# ============================================================
# MERIEN KOKONAISVAIKUTUS
# ============================================================

def calculate_ocean_coverage_factor(
    ocean_coverage
):

    factor = (
        1.0
        +
        OCEAN_COVERAGE_RAIN_EFFECT
        *
        (
            ocean_coverage
            -
            REFERENCE_OCEAN_COVERAGE
        )
    )

    return np.clip(
        factor,
        0.70,
        1.30
    )


# ============================================================
# TUULEN YKSIKKÖVEKTORIT
# ============================================================

def calculate_wind_direction(
    wind_u,
    wind_v
):

    speed = np.sqrt(
        wind_u ** 2
        +
        wind_v ** 2
    )

    speed = np.maximum(
        speed,
        1e-6
    )

    unit_u = (
        wind_u
        /
        speed
    )

    unit_v = (
        wind_v
        /
        speed
    )

    return (
        unit_u,
        unit_v,
        speed
    )


# ============================================================
# MEREN SUUNTA
# ============================================================

def calculate_ocean_direction(
    land_mask
):

    distance_from_ocean = (
        distance_transform_edt(
            land_mask
        )
    )

    grad_y, grad_x = np.gradient(
        distance_from_ocean.astype(float)
    )

    magnitude = np.sqrt(
        grad_x ** 2
        +
        grad_y ** 2
    )

    magnitude = np.maximum(
        magnitude,
        1e-6
    )

    ocean_dir_x = (
        -grad_x
        /
        magnitude
    )

    ocean_dir_y = (
        -grad_y
        /
        magnitude
    )

    ocean_dir_x *= land_mask
    ocean_dir_y *= land_mask

    return (
        ocean_dir_x,
        ocean_dir_y,
        distance_from_ocean
    )


# ============================================================
# TUULEN JA MEREN SUUNNAN SUHDE
# ============================================================

def calculate_wind_ocean_alignment(
    wind_u,
    wind_v,
    land_mask
):

    wind_x, wind_y, wind_speed = (
        calculate_wind_direction(
            wind_u,
            wind_v
        )
    )

    ocean_x, ocean_y, distance = (
        calculate_ocean_direction(
            land_mask
        )
    )

    alignment = (
        wind_x * ocean_x
        +
        wind_y * ocean_y
    )

    alignment *= (
        land_mask
    )

    return (
        np.clip(
            alignment,
            -1,
            1
        ),
        wind_speed,
        distance
    )


# ============================================================
# TUULEN KULJETTAMA KOSTEUS
# ============================================================

def calculate_wind_moisture(
    wind_u,
    wind_v,
    ocean_mask,
    ocean_temperature,
    land_mask
):

    wind_x, wind_y, wind_speed = (
        calculate_wind_direction(
            wind_u,
            wind_v
        )
    )

    evaporation = (
        ocean_temperature
        + 5
    )

    evaporation = np.clip(
        evaporation,
        0,
        None
    )

    evaporation *= (
        ocean_mask.astype(float)
    )

    evaporation *= (
        0.65
        +
        0.45
        *
        np.clip(
            wind_speed,
            0,
            1.5
        )
    )

    transported = np.zeros_like(
        evaporation
    )

    max_steps = 22

    for step in range(
        1,
        max_steps + 1
    ):

        distance = (
            step * 2.0
        )

        shift_x = np.rint(
            wind_x * distance
        ).astype(int)

        shift_y = np.rint(
            wind_y * distance
        ).astype(int)

        shifted = np.zeros_like(
            evaporation
        )

        for y in range(
            HEIGHT
        ):

            source_y = (
                y
                -
                shift_y[y]
            )

            source_y = np.clip(
                source_y,
                0,
                HEIGHT - 1
            )

            source_x = (
                np.arange(WIDTH)
                -
                shift_x[y]
            ) % WIDTH

            shifted[y] = (
                evaporation[
                    source_y,
                    source_x
                ]
            )

        decay = np.exp(
            -(
                distance
                /
                WIND_MOISTURE_DISTANCE
            )
            ** WIND_MOISTURE_DECAY
        )

        wind_transport = np.clip(
            wind_speed / 0.65,
            0.20,
            1.80
        )

        transported += (
            shifted
            *
            decay
            *
            wind_transport
        )

    transported /= (
        max_steps
    )

    transported *= (
        land_mask
    )

    return transported


# ============================================================
# TUULEN MERI -> MAA / MAA -> MERI
# ============================================================

def calculate_wind_rain_effect(
    wind_u,
    wind_v,
    land_mask
):

    alignment, wind_speed, ocean_distance = (
        calculate_wind_ocean_alignment(
            wind_u,
            wind_v,
            land_mask
        )
    )

    distance_factor = np.exp(
        -ocean_distance
        /
        WIND_MOISTURE_DISTANCE
    )

    speed_factor = np.clip(
        wind_speed / 0.65,
        0,
        1.5
    )

    effect = (
        alignment
        *
        distance_factor
        *
        speed_factor
    )

    effect *= (
        land_mask
    )

    return np.clip(
        effect,
        -1,
        1
    )


# ============================================================
# LÄMPÖTILA
# ============================================================

def calculate_temperature(
    elevation,
    land_mask,
    ocean_temperature,
    ocean_current_effect
):

    #latitude_effect = np.cos(
    #    np.radians(LAT)
    #)
    #latitude_effect = np.cos(np.radians(np.abs(LAT)))
    #land_temperature = 30 * latitude_effect**0.5 - 25
    lat = np.abs(LAT)

    land_temperature = (
    30
    - 0.0055 * lat**2
    )    
    land_temperature -= (
        np.maximum(
            elevation - SEA_LEVEL,
            0
        )
        * 16
    )

    temperature = np.where(
        land_mask,
        land_temperature,
        ocean_temperature
    )

    ocean_effect = gaussian_filter(
        ocean_current_effect,
        sigma=(5, 10),
        mode=("nearest", "wrap")
    )

    temperature += (
        ocean_effect
        *
        (
            0.35
            +
            0.65 * (~land_mask)
        )
    )

    coastal_ocean = gaussian_filter(
        (~land_mask).astype(float),
        sigma=(5, 12),
        mode=("nearest", "wrap")
    )

    temperature = (
        temperature
        * (
            1
            -
            coastal_ocean * 0.25
        )
        +
        ocean_temperature
        * (
            coastal_ocean * 0.25
        )
    )

    return temperature


# ============================================================
# MONSUUNIPOTENTIAALI
# ============================================================

def calculate_monsoon(
    temperature,
    ocean_temperature,
    land_mask,
    elevation,
    wind_u,
    wind_v
):

    abs_lat = np.abs(
        LAT
    )

    thermal_contrast = (
        temperature
        -
        ocean_temperature
    )

    tropical_band = np.exp(
        -(
            (abs_lat - 18)
            / 20
        ) ** 2
    )

    nearby_land = gaussian_filter(
        land_mask.astype(float),
        sigma=(12, 25),
        mode=("nearest", "wrap")
    )

    continentality = (
        nearby_land
        * land_mask
    )

    plateau_factor = np.clip(
        (elevation - 0.58) / 0.25,
        0,
        1
    )

    plateau_factor *= (
        continentality
    )

    warm_ocean = np.clip(
        (ocean_temperature - 18) / 12,
        0,
        1
    )

    ocean_moisture = gaussian_filter(
        warm_ocean * (~land_mask),
        sigma=(8, 18),
        mode=("nearest", "wrap")
    )

    monsoon = (
        thermal_contrast
        / 12.0
    )

    monsoon *= (
        tropical_band
    )

    monsoon *= (
        0.45
        +
        0.55 * continentality
    )

    monsoon += (
        ocean_moisture
        * 0.9
    )

    monsoon += (
        plateau_factor
        * 0.8
    )

    land_gradient_y, land_gradient_x = np.gradient(
        nearby_land
    )

    landward_flow = (
        wind_u * land_gradient_x
        +
        wind_v * land_gradient_y
    )

    monsoon += (
        landward_flow
        * 0.8
        * continentality
    )

    monsoon = gaussian_filter(
        monsoon,
        sigma=(6, 12),
        mode=("nearest", "wrap")
    )

    return monsoon


# ============================================================
# VUODENAJAT
# ============================================================

def calculate_seasonal_temperature(
    base_temperature,
    land_mask,
    ocean_temperature
):

    seasonal_amplitude = (
        2
        +
        13
        *
        np.sin(
            np.radians(
                np.abs(LAT)
            )
        ) ** 1.7
    )

    land_factor = np.where(
        land_mask,
        1.0,
        0.35
    )

    amplitude = (
        seasonal_amplitude
        * land_factor
    )

    hemisphere = np.where(
        LAT >= 0,
        1,
        -1
    )

    summer_temperature = (
        base_temperature
        +
        amplitude
        * hemisphere
    )

    winter_temperature = (
        base_temperature
        -
        amplitude
        * hemisphere
    )

    summer_temperature = (
        summer_temperature
        * 0.90
        +
        ocean_temperature
        * 0.10
    )

    winter_temperature = (
        winter_temperature
        * 0.90
        +
        ocean_temperature
        * 0.10
    )

    return (
        summer_temperature,
        winter_temperature
    )


# ============================================================
# VUORISTON SADEVARJO
# ============================================================

def calculate_rain_shadow(
    elevation,
    land_mask,
    wind_u,
    wind_v
):

    """
    Laskee vuoriston tuulen alapuolelle syntyvän
    sadevarjon.

    Periaate:

        TUULI →
        
        kostea ilma
             ↓
        /\/\/\/\/\
        vuoristo
        /\/\/\/\/\
             ↓
        kuiva ilma
             ↓
        sadevarjo

    Shadow-kenttä kasvaa sitä voimakkaammaksi,
    mitä korkeampi vuoristo ilma on juuri ylittänyt.

    Varjo kulkee tuulen mukana ja vaimenee
    vähitellen etäisyyden kasvaessa.
    """

    # --------------------------------------------------------
    # TUULEN SUUNTA JA NOPEUS
    # --------------------------------------------------------

    wind_x, wind_y, wind_speed = (
        calculate_wind_direction(
            wind_u,
            wind_v
        )
    )

    # --------------------------------------------------------
    # VUORISTON KORKEUS
    # --------------------------------------------------------

    mountain_height = np.clip(
        (
            elevation
            -
            RAIN_SHADOW_MOUNTAIN_THRESHOLD
        )
        /
        (
            1.0
            -
            RAIN_SHADOW_MOUNTAIN_THRESHOLD
        ),
        0,
        1
    )

    mountain_height *= (
        land_mask
    )

    mountain_height = (
        mountain_height ** 1.35
    )

    # --------------------------------------------------------
    # PEHMEÄ VUORISTOKENTTÄ
    # --------------------------------------------------------

    mountain_field = gaussian_filter(
        mountain_height,
        sigma=(2.0, 4.0),
        mode=("nearest", "wrap")
    )

    mountain_field *= (
        land_mask
    )

    # --------------------------------------------------------
    # MERELTÄ MAAHAN PUHALTAMINEN
    # --------------------------------------------------------

    wind_effect = (
        calculate_wind_rain_effect(
            wind_u,
            wind_v,
            land_mask
        )
    )

    onshore = np.maximum(
        wind_effect,
        0
    )

    # --------------------------------------------------------
    # VUORISTON LÄPI KULKEVA VARJO
    # --------------------------------------------------------

    shadow = np.zeros_like(
        elevation,
        dtype=float
    )

    for step in range(
        1,
        RAIN_SHADOW_MAX_STEPS + 1
    ):

        distance = (
            step
            *
            RAIN_SHADOW_STEP
        )

        shift_x = np.rint(
            wind_x * distance
        ).astype(int)

        shift_y = np.rint(
            wind_y * distance
        ).astype(int)

        shifted = np.zeros_like(
            mountain_field
        )

        for y in range(
            HEIGHT
        ):

            source_y = (
                y
                -
                shift_y[y]
            )

            source_y = np.clip(
                source_y,
                0,
                HEIGHT - 1
            )

            source_x = (
                np.arange(WIDTH)
                -
                shift_x[y]
            ) % WIDTH

            shifted[y] = (
                mountain_field[
                    source_y,
                    source_x
                ]
            )

        decay = np.exp(
            -(
                distance
                /
                RAIN_SHADOW_DISTANCE
            )
            ** RAIN_SHADOW_DECAY
        )

        wind_factor = np.clip(
            wind_speed / 0.65,
            0.20,
            1.80
        )

        shadow += (
            shifted
            *
            decay
            *
            wind_factor
        )

    shadow /= (
        RAIN_SHADOW_MAX_STEPS
    )

    # --------------------------------------------------------
    # KORKEUS + TUULI
    # --------------------------------------------------------

    shadow *= (
        0.60
        +
        mountain_height
        * 1.25
    )

    shadow *= (
        0.55
        +
        onshore
        *
        RAIN_SHADOW_WIND_STRENGTH
    )

    # --------------------------------------------------------
    # VAIN MAA-ALUEILLE
    # --------------------------------------------------------

    shadow *= (
        land_mask
    )

    return np.clip(
        shadow,
        0,
        None
    )


# ============================================================
# OROGRAFINEN SADE
# ============================================================

def calculate_orographic_rainfall(
    rainfall,
    elevation,
    land_mask,
    wind_u,
    wind_v
):

    """
    Vuoriston tuulenpuolen nousu + sadevarjo.

    Ensin ilma pakotetaan nousemaan vuoren
    tuulenpuolella.

    Sen jälkeen kosteutta vähennetään
    vuoriston tuulen alapuolella.
    """

    # --------------------------------------------------------
    # MAASTON GRADIENTTI
    # --------------------------------------------------------

    grad_y, grad_x = np.gradient(
        elevation
    )

    slope_into_wind = (
        grad_x * wind_u
        +
        grad_y * wind_v
    )

    # --------------------------------------------------------
    # TUULENPUOLEN NOUSU
    # --------------------------------------------------------

    uplift = np.maximum(
        slope_into_wind,
        0
    )

    uplift = gaussian_filter(
        uplift,
        sigma=(1.5, 3.0),
        mode=("nearest", "wrap")
    )

    # --------------------------------------------------------
    # VUORISTON KORKEUS
    # --------------------------------------------------------

    mountain_factor = np.clip(
        (
            elevation
            -
            SEA_LEVEL
        )
        /
        (
            1.0
            -
            SEA_LEVEL
        ),
        0,
        1
    )

    mountain_factor = (
        mountain_factor ** 1.5
    )

    mountain_factor *= (
        land_mask
    )

    # --------------------------------------------------------
    # MERELTÄ TULEVA TUULI
    # --------------------------------------------------------

    wind_effect = (
        calculate_wind_rain_effect(
            wind_u,
            wind_v,
            land_mask
        )
    )

    onshore = np.maximum(
        wind_effect,
        0
    )

    # --------------------------------------------------------
    # OROGRAFINEN SADE
    # --------------------------------------------------------

    rainfall += (
        uplift
        *
        mountain_factor
        *
        (
            1.0
            +
            onshore * 1.8
        )
        *
        OROGRAPHIC_RAIN_STRENGTH
    )

    # --------------------------------------------------------
    # VANHA LÄHIALUEEN KUIVUMINEN
    # --------------------------------------------------------

    downslope = np.maximum(
        -slope_into_wind,
        0
    )

    downslope = gaussian_filter(
        downslope,
        sigma=(3, 7),
        mode=("nearest", "wrap")
    )

    rainfall -= (
        downslope
        *
        mountain_factor
        *
        (
            1.0
            +
            onshore * 1.5
        )
        *
        OROGRAPHIC_DRYING_STRENGTH
    )

    # --------------------------------------------------------
    # UUSI PITKÄ SADEVARJO
    # --------------------------------------------------------

    rain_shadow = calculate_rain_shadow(
        elevation,
        land_mask,
        wind_u,
        wind_v
    )

    rainfall -= (
        rain_shadow
        *
        RAIN_SHADOW_STRENGTH
    )

    return np.maximum(
        rainfall,
        0
    )


# ============================================================
# YHDEN VUODENAJAN SADE
# ============================================================

def calculate_seasonal_rainfall(
    temperature,
    elevation,
    land_mask,
    wind_u,
    wind_v,
    ocean_temperature,
    monsoon,
    continentality,
    ocean_humidity_factor,
    season
):

    abs_lat = np.abs(
        LAT
    )

    rainfall = np.zeros_like(
        temperature
    )

    # ========================================================
    # 1. GLOBAALI PERUSSADE
    # ========================================================

    equatorial = np.exp(
        -(
            abs_lat / 15
        ) ** 2
    )

    rainfall += (
        equatorial * 700
    )

    # ========================================================
    # 2. SUBTROOPPINEN KUIVUUS
    # ========================================================

    subtropical = np.exp(
        -(
            (abs_lat - 28)
            / 11
        ) ** 2
    )

    rainfall -= (
        subtropical * 600
    )

    # ========================================================
    # 3. LAUHKEAN VYÖHYKKEEN SADE
    # ========================================================

    temperate = np.exp(
        -(
            (abs_lat - 50)
            / 18
        ) ** 2
    )

    rainfall += (
        temperate * 450
    )

    # ========================================================
    # 4. TUULEN KULJETTAMA MERIKOSTEUS
    # ========================================================

    moisture = (
        calculate_wind_moisture(
            wind_u,
            wind_v,
            ~land_mask,
            ocean_temperature,
            land_mask
        )
    )

    rainfall += (
        moisture
        *
        MARITIME_MOISTURE_STRENGTH
        *
        land_mask
    )

    # ========================================================
    # 5. MERI -> MAA / MAA -> MERI
    # ========================================================

    wind_effect = (
        calculate_wind_rain_effect(
            wind_u,
            wind_v,
            land_mask
        )
    )

    onshore = np.maximum(
        wind_effect,
        0
    )

    rainfall += (
        onshore
        *
        WIND_MOISTURE_STRENGTH
        *
        land_mask
    )

    offshore = np.maximum(
        -wind_effect,
        0
    )

    rainfall -= (
        offshore
        *
        WIND_DRYING_STRENGTH
        *
        land_mask
    )

    # ========================================================
    # 6. LÄMMIN MERI
    # ========================================================

    warm_ocean = np.maximum(
        ocean_temperature - 12,
        0
    )

    rainfall += (
        warm_ocean
        *
        (~land_mask)
        *
        1.3
    )

    # ========================================================
    # 7. MONSUUNI
    # ========================================================

    if season == "summer":

        monsoon_wet = np.maximum(
            monsoon,
            0
        )

        rainfall += (
            monsoon_wet
            *
            MONSOON_RAIN_STRENGTH
            *
            land_mask
        )

    elif season == "winter":

        monsoon_dry = np.maximum(
            -monsoon,
            0
        )

        rainfall -= (
            monsoon_dry
            *
            300
            *
            land_mask
        )

    # ========================================================
    # 8. MANTEREISUUS
    # ========================================================

    continentality_factor = (
        1.0
        -
        CONTINENTALITY_RAIN_EFFECT
        *
        continentality
    )

    continentality_factor += (
        onshore
        * 0.35
    )

    continentality_factor -= (
        offshore
        * 0.40
    )

    continentality_factor = np.clip(
        continentality_factor,
        0.12,
        1.15
    )

    rainfall *= np.where(
        land_mask,
        continentality_factor,
        1.0
    )

    # ========================================================
    # 9. MERIEN KOKONAISVAIKUTUS
    # ========================================================

    rainfall *= (
        ocean_humidity_factor
    )

    # ========================================================
    # 10. LÄMPÖTILAN VAIKUTUS
    # ========================================================

    moisture_capacity = np.clip(
        (temperature + 5) / 30,
        0.15,
        1.35
    )

    rainfall *= (
        0.55
        +
        moisture_capacity * 0.45
    )

    # ========================================================
    # 11. OROGRAFIA + SADEVARJOT
    # ========================================================

    rainfall = calculate_orographic_rainfall(
        rainfall,
        elevation,
        land_mask,
        wind_u,
        wind_v
    )

    # ========================================================
    # 12. MERISADE
    # ========================================================

    rainfall += (
        (~land_mask)
        *
        120
    )

    # ========================================================
    # 13. GLOBAALI SÄÄTÖ
    # ========================================================

    rainfall *= (
        GLOBAL_RAINFALL_SCALE
    )

    return np.maximum(
        rainfall,
        0
    )


# ============================================================
# KASVUKAUSI
# ============================================================

def calculate_growing_season(
    summer_temperature,
    winter_temperature,
    summer_rainfall,
    winter_rainfall,
    land_mask
):

    warm_score = np.clip(
        (
            summer_temperature - 5
        ) / 20,
        0,
        1
    )

    winter_survival = np.clip(
        (
            winter_temperature + 5
        ) / 15,
        0,
        1
    )

    moisture = (
        0.5
        *
        normalize(
            summer_rainfall
        )
        +
        0.5
        *
        normalize(
            winter_rainfall
        )
    )

    season = (
        warm_score
        *
        (
            0.55
            +
            0.45 * winter_survival
        )
        *
        (
            0.55
            +
            0.45 * moisture
        )
    )

    season *= (
        land_mask
    )

    return normalize(
        season
    )


# ============================================================
# VUOSISADE
# ============================================================

def calculate_annual_rainfall(
    summer_rainfall,
    winter_rainfall
):

    return (
        summer_rainfall
        +
        winter_rainfall
    )


# ============================================================
# SADEKAUSIVAIHTELU
# ============================================================

def calculate_rainfall_seasonality(
    summer_rainfall,
    winter_rainfall
):

    total = (
        summer_rainfall
        +
        winter_rainfall
        +
        1e-9
    )

    return (
        np.abs(
            summer_rainfall
            -
            winter_rainfall
        )
        /
        total
    )


# ============================================================
# BIOMIT
# ============================================================

def classify_biomes(
    temperature,
    rainfall,
    seasonality,
    land_mask
):

    biome = np.zeros(
        temperature.shape,
        dtype=np.uint8
    )

    hot = (
        temperature > 23
    )

    temperate = (
        (temperature >= 5)
        &
        (temperature < 23)
    )

    cold = (
        temperature < 3
    )

    very_cold = (
        temperature < -5
    )

    dry = (
        rainfall < 250
    )

    semi_dry = (
        (rainfall >= 250)
        &
        (rainfall < 500)
    )

    wet = (
        rainfall >= 1000
    )

    biome[
        hot
        &
        wet
        &
        land_mask
    ] = 4

    biome[
        (~hot)
        &
        dry
        &
        land_mask
    ] = 1

    biome[
        hot
        &
        dry
        &
        land_mask
    ] = 1

    biome[
        hot
        &
        semi_dry
        &
        land_mask
    ] = 2

    biome[
        temperate
        &
        semi_dry
        &
        land_mask
    ] = 2

    biome[
        temperate
        &
        wet
        &
        land_mask
    ] = 3

    moderate = (
        (rainfall >= 500)
        &
        (rainfall < 1000)
    )

    biome[
        temperate
        &
        moderate
        &
        (seasonality > 0.50)
        &
        land_mask
    ] = 2

    biome[
        temperate
        &
        moderate
        &
        (seasonality <= 0.50)
        &
        land_mask
    ] = 3

    biome[
        cold
        &
        land_mask
    ] = 5

    biome[
        very_cold
        &
        land_mask
    ] = 6

    return biome


# ============================================================
# RANNIKKOLÄHEISYYS
# ============================================================

def calculate_coastal_proximity(
    land_mask
):

    distance_from_ocean = (
        distance_transform_edt(
            land_mask
        )
    )

    proximity = np.exp(
        -distance_from_ocean
        / 18
    )

    proximity *= (
        land_mask
    )

    return normalize(
        proximity
    )


# ============================================================
# VUORISTON LÄHEISYYS
# ============================================================

def calculate_mountain_proximity(
    elevation,
    land_mask
):

    land_elevation = np.where(
        land_mask,
        elevation,
        0
    )

    max_land = np.max(
        land_elevation
    )

    mountain_threshold = (
        SEA_LEVEL
        +
        (
            max_land - SEA_LEVEL
        )
        * 0.33
    )

    mountain_zone = (
        land_mask
        &
        (
            elevation
            >= mountain_threshold
        )
    )

    distance = distance_transform_edt(
        ~mountain_zone
    )

    proximity = np.exp(
        -distance / 25
    )

    proximity *= (
        land_mask
    )

    return normalize(
        proximity
    )


# ============================================================
# VUORISTORESURSSIT
# ============================================================

def calculate_mountain_resources(
    elevation,
    mountain_proximity,
    rainfall
):

    highland = np.clip(
        (
            elevation
            - SEA_LEVEL
        )
        /
        (
            1 - SEA_LEVEL
        ),
        0,
        1
    )

    resource = (
        mountain_proximity
        * 0.65
        +
        highland
        * 0.35
    )

    forest_bonus = np.clip(
        rainfall / 1200,
        0,
        1
    )

    resource += (
        forest_bonus
        *
        mountain_proximity
        *
        0.20
    )

    return normalize(
        resource
    )

import numpy as np
from scipy.ndimage import label


# ============================================================
# LAND REGION PARAMETERS
# ============================================================

CONTINENT_AREA_FRACTION = 0.05
LARGE_ISLAND_AREA_FRACTION = 0.005

MIN_LAND_AREA = 100



import numpy as np
from scipy.ndimage import label


# ============================================================
# LAND REGION PARAMETERS
# ============================================================

CONTINENT_AREA_FRACTION = 0.05
LARGE_ISLAND_AREA_FRACTION = 0.005

MIN_LAND_AREA = 100


# ============================================================
# LAND REGION ANALYSIS
# ============================================================

def analyze_land_regions(
    land_mask,
    min_land_area=MIN_LAND_AREA
):

    # --------------------------------------------------------
    # MAA / MERI
    #
    # 1 = maa
    # 0 = meri
    # --------------------------------------------------------

    land = (
        land_mask > 0
    )

    height, width = land.shape

    # --------------------------------------------------------
    # LEVEYSASTEEN MUKAINEN PINTA-ALAPAINO
    #
    # Equirectangular 720 x 360 -kartassa pikselit eivät
    # ole saman kokoisia todellisella pallopinnalla.
    # --------------------------------------------------------

    latitudes = np.linspace(
        90,
        -90,
        height
    )

    area_weight = np.cos(
        np.deg2rad(latitudes)
    )

    # --------------------------------------------------------
    # YHTENÄISTEN MAA-ALUEIDEN TUNNISTUS
    # --------------------------------------------------------

    structure = np.ones(
        (3, 3),
        dtype=int
    )

    region_map, num_regions = label(
        land,
        structure=structure
    )

    # --------------------------------------------------------
    # UNION-FIND
    #
    # Longitude on periodinen:
    #
    # vasen reuna <-> oikea reuna
    # --------------------------------------------------------

    parent = np.arange(
        num_regions + 1
    )

    def find(a):

        while parent[a] != a:

            parent[a] = parent[
                parent[a]
            ]

            a = parent[a]

        return a

    def union(a, b):

        a = find(a)
        b = find(b)

        if a != b:
            parent[b] = a

    # --------------------------------------------------------
    # VASEN / OIKEA REUNA
    # --------------------------------------------------------

    for y in range(height):

        left = region_map[
            y,
            0
        ]

        right = region_map[
            y,
            width - 1
        ]

        if left > 0 and right > 0:

            union(
                left,
                right
            )

    # --------------------------------------------------------
    # DIAGONAALISET REUNAKONTAKTIT
    # --------------------------------------------------------

    for y in range(
        height - 1
    ):

        left = region_map[
            y,
            0
        ]

        right = region_map[
            y + 1,
            width - 1
        ]

        if left > 0 and right > 0:

            union(
                left,
                right
            )

        left = region_map[
            y + 1,
            0
        ]

        right = region_map[
            y,
            width - 1
        ]

        if left > 0 and right > 0:

            union(
                left,
                right
            )

    # --------------------------------------------------------
    # UUDELLEENNUMEROINTI
    # --------------------------------------------------------

    root_to_new = {}

    new_id = 1

    for old_id in range(
        1,
        num_regions + 1
    ):

        root = find(
            old_id
        )

        if root not in root_to_new:

            root_to_new[
                root
            ] = new_id

            new_id += 1

    new_region_map = np.zeros_like(
        region_map
    )

    for old_id in range(
        1,
        num_regions + 1
    ):

        root = find(
            old_id
        )

        new_region_map[
            region_map == old_id
        ] = root_to_new[
            root
        ]

    region_map = new_region_map

    num_regions = (
        new_id - 1
    )

    # --------------------------------------------------------
    # REGION-ALAT
    #
    # region_area[id] = koko regionin suhteellinen pinta-ala
    # --------------------------------------------------------

    region_area = np.zeros(
        num_regions + 1,
        dtype=float
    )

    for region_id in range(
        1,
        num_regions + 1
    ):

        mask = (
            region_map == region_id
        )

        y, x = np.where(
            mask
        )

        if len(y) == 0:
            continue

        region_area[
            region_id
        ] = np.sum(
            area_weight[y]
        )

    # --------------------------------------------------------
    # POISTA HYVIN PIENET MAA-ALUEET
    # --------------------------------------------------------

    small_regions = (
        region_area
        < min_land_area
    )

    small_regions[0] = False

    region_map[
        small_regions[
            region_map
        ]
    ] = 0

    # --------------------------------------------------------
    # PLANEETAN KOKONAISMAA-ALA
    # --------------------------------------------------------

    valid_region_ids = [
        region_id
        for region_id in range(
            1,
            num_regions + 1
        )
        if (
            region_area[region_id]
            >= min_land_area
        )
    ]

    if not valid_region_ids:

        return (
            region_map,
            np.zeros_like(
                region_map,
                dtype=float
            ),
            np.zeros_like(
                region_map,
                dtype=float
            ),
            []
        )

    total_land_area = np.sum(
        region_area[
            valid_region_ids
        ]
    )

    # --------------------------------------------------------
    # SUURIMMAN MAA-ALUEEN ALA
    # --------------------------------------------------------

    largest_region_area = np.max(
        region_area[
            valid_region_ids
        ]
    )

    # --------------------------------------------------------
    # REGION AREA MAP
    #
    # Jokainen pikseli saa koko sen regionin pinta-alan,
    # johon pikseli kuuluu.
    #
    # Meri = 0
    # --------------------------------------------------------

    region_area_map = (
        region_area[
            region_map
        ]
    )

    # --------------------------------------------------------
    # REGION SIZE FACTOR MAP
    #
    # Suurin maa-alue = 1.0
    #
    # Pienemmät alueet pienenevät neliöjuuren mukaan.
    # --------------------------------------------------------

    region_size_factor = np.zeros(
        num_regions + 1,
        dtype=float
    )

    region_size_factor[
        valid_region_ids
    ] = np.sqrt(
        region_area[
            valid_region_ids
        ]
        /
        largest_region_area
    )

    region_size_factor_map = (
        region_size_factor[
            region_map
        ]
    )

    # --------------------------------------------------------
    # REGION METADATA
    # --------------------------------------------------------

    regions = []

    for region_id in valid_region_ids:

        area = region_area[
            region_id
        ]

        relative_area = (
            area
            /
            total_land_area
        )

        # ----------------------------------------------------
        # REGION TYPE
        # ----------------------------------------------------

        if (
            relative_area
            >= CONTINENT_AREA_FRACTION
        ):

            region_type = "continent"

        elif (
            relative_area
            >= LARGE_ISLAND_AREA_FRACTION
        ):

            region_type = "large_island"

        else:

            region_type = "island"

        # ----------------------------------------------------
        # PIXELIT
        # ----------------------------------------------------

        mask = (
            region_map == region_id
        )

        y, x = np.where(
            mask
        )

        # ----------------------------------------------------
        # REGION CENTER
        # ----------------------------------------------------

        center_y = np.mean(y)
        center_x = np.mean(x)

        regions.append(
            {
                "id": region_id,

                "area": area,

                "relative_area": (
                    relative_area
                ),

                "size_factor": (
                    region_size_factor[
                        region_id
                    ]
                ),

                "pixel_count": len(y),

                "center": (
                    center_y,
                    center_x
                ),

                "type": region_type,

                "is_continent": (
                    region_type
                    == "continent"
                ),

                "is_island": (
                    region_type
                    != "continent"
                )
            }
        )

    return (
        region_map,
        region_area_map,
        region_size_factor_map,
        regions
    )


# ============================================================
# SIVILISAATIO
# ============================================================

# ============================================================
# SIVILISAATIOPOTENTIAALI
# ============================================================

def civilization_potential(
    temperature,
    rainfall,
    elevation,
    rivers,
    land_mask,
    region_size_factor_map
):

    # --------------------------------------------------------
    # ALUSTUS
    # --------------------------------------------------------

    score = np.zeros_like(
        temperature,
        dtype=float
    )

    land = (
        land_mask > 0
    )

    # --------------------------------------------------------
    # 1. LÄMPÖTILA
    #
    # Lämmin lauhkea -> subtrooppinen on hyvä.
    #
    # Kuumuus ei vielä tuhoa potentiaalia:
    # kastelu ja jokivedet mahdollistavat hyvin kuuman
    # ilmaston sivilisaatiot.
    # --------------------------------------------------------
    plt.imshow(temperature)
    plt.show()
    temperature_score = np.exp(
        -(
            (temperature - 20.0)
            / 11 ##11.0
        ) ** 2
    )

    score += (
        temperature_score
        * 2.0
    )

    # --------------------------------------------------------
    # LIIAN KYLMÄ
    #
    # Erittäin kylmä ilmasto on huono primäärisen
    # sivilisaation syntyalue.
    # --------------------------------------------------------

    cold_penalty = np.clip(
        (12.0 - temperature)
        / 12.0,
        0.0,
        1.0
    )

    score *= (
        1.0
        -
        0.90 * cold_penalty
    )
    not_too_cold =  np.copy(temperature)
    not_too_cold =  np.where(not_too_cold<12,0,1)
    score *= not_too_cold
    # --------------------------------------------------------
    # LIIAN KUUMA
    #
    # Kuuma ei ole mahdoton, mutta äärimmäinen kuumuus
    # vaikeuttaa luonnollista maataloutta ja asumista.
    #
    # Joki voi silti tehdä tällaisesta alueesta hyvän.
    # --------------------------------------------------------

    hot_penalty = np.clip(
        (temperature - 28.0)
        / 8.0,
        0.0,
        1.0
    )

    score *= (
        1.0
        -
        0.15 * hot_penalty  ## 0.45*
    )

    # --------------------------------------------------------
    # 2. SADE
    #
    # Kohtuullisen kostea ilmasto on hyvä.
    # --------------------------------------------------------

    rainfall_score = np.exp(
        -(
            (rainfall - 650.0)
            / 400.0
        ) ** 2
    )

    score += (
        rainfall_score
        * 1.5
    )

    # --------------------------------------------------------
    # 3. KUIVA / PUOLIKUIVA
    #
    # Kuivuus muuttuu eduksi erityisesti silloin,
    # kun paikalla on suuri joki.
    # --------------------------------------------------------

    dry_score = np.exp(
        -(
            (rainfall - 350.0)
            / 250.0
        ) ** 2
    )

    # --------------------------------------------------------
    # 4. JOET
    #
    # Suuret joet ovat erittäin tärkeä sivilisaation
    # syntyä suosiva tekijä.
    # --------------------------------------------------------

    river_score = np.clip(
        rivers,
        0,
        1
    ) ** 1.5

    score += (
        river_score
        * 3.5
    )

    # --------------------------------------------------------
    # 5. KUIVA + JOKI
    #
    # Egypti / Mesopotamia -tyyppinen ympäristö.
    # --------------------------------------------------------

    river_desert_bonus = (
        dry_score
        * river_score
    )

    score += (
        river_desert_bonus
        * 3.0
    )

    # --------------------------------------------------------
    # 6. ALANKO
    #
    # Matala maa on yleisesti parempi.
    #
    # Korkeaa maata ei kuitenkaan tuhota kokonaan:
    # se voi edelleen synnyttää sivilisaation.
    # --------------------------------------------------------

    lowland_score = np.clip(
        1.0
        -
        elevation / 0.75,
        0.05,
        1.0
    )

    score += (
        lowland_score
        * 1.5
    )

    # --------------------------------------------------------
    # 7. VUORISTON LÄHEISYYS
    #
    # Vuoristo itsessään ei ole tavoite.
    #
    # Bonus tulee alueelle, joka on vuoriston lähellä
    # ja samalla riittävän matala.
    # --------------------------------------------------------

    mountain_threshold = (
        0.48
        +
        (
            elevation.max()
            - 0.48
        )
        * 0.33
    )

    mountain_mask = (
        elevation
        >= mountain_threshold
    )

    mountain_field = gaussian_filter(
        mountain_mask.astype(float),
        sigma=(10, 18),
        mode=("nearest", "wrap")
    )

    mountain_field = np.clip(
        mountain_field * 5.0,
        0,
        1
    )

    mountain_proximity = (
        mountain_field
        * lowland_score
    )

    score += (
        mountain_proximity
        * 1.3 ##*1.3
    )

    # --------------------------------------------------------
    # 8. RANNIKKO
    #
    # Meren läheisyys tarjoaa liikennettä, kalastusta ja
    # kaupankäyntimahdollisuuksia.
    # --------------------------------------------------------

    sea_mask = (
        ~land
    )

    coast_field = gaussian_filter(
        sea_mask.astype(float),
        #sigma=(8, 15),
        sigma=(2, 5),
        mode=("nearest", "wrap")
    )

    coast_field = np.clip(
        coast_field * 4.0,
        0,
        1
    )

    coast_field[
        ~land
    ] = 0

    score += (
        coast_field
        * 0.8
    )

    # --------------------------------------------------------
    # 9. JOKI + RANNIKKO
    #
    # Jokisuu ja rannikon suuri jokilaakso.
    #
    # Tämä on erittäin tyypillinen sivilisaation
    # synty-ympäristö.
    # --------------------------------------------------------

    river_coast_bonus = (
        river_score
        * coast_field
    )

    score += (
        river_coast_bonus
        * 2.0
    )

    # --------------------------------------------------------
    # 10. ALANKO + RANNIKKO
    # --------------------------------------------------------

    coastal_lowland_bonus = (
        coast_field
        * lowland_score
    )

    score += (
        coastal_lowland_bonus
        * 1.5
    )

    # --------------------------------------------------------
    # 11. VUORISTO + RANNIKKO
    #
    # Esimerkiksi Andien länsipuolisten alueiden kaltaiset
    # ympäristöt.
    # --------------------------------------------------------

    mountain_coast_bonus = (
        mountain_proximity
        * coast_field
    )

    score += (
        mountain_coast_bonus
        * 1.5
    )

    # --------------------------------------------------------
    # 12. KAUPPA / LIIKENNE
    #
    # Joki + rannikko + vuoriston läheisyys.
    # --------------------------------------------------------

    trade = (
        river_score
        *
        (
            0.5
            +
            coast_field
        )
    )

    trade += (
        river_score
        *
        mountain_proximity
        * 0.6
    )

    trade += (
        coast_field
        *
        lowland_score
        * 0.5
    )

    trade = np.clip(
        trade,
        0,
        1
    )

    score += (
        trade
        * 1.5
    )

    # --------------------------------------------------------
    # 13. MAA-ALUEEN KOKO
    #
    # Suuri yhtenäinen manner saa etua.
    #
    # Saaret säilyttävät kuitenkin merkittävän potentiaalin.
    # --------------------------------------------------------

    region_factor = (
        0.20
        +
        0.80
        *
        region_size_factor_map
    )

    score *= (
        region_factor
    )

    score *= not_too_cold ## ADDED
    score *= coast_field ##ADDED
    score *= river_desert_bonus 
    # --------------------------------------------------------
    # 14. MERI POIS
    # --------------------------------------------------------

    score[
        ~land
    ] = 0

    # --------------------------------------------------------
    # 15. NORMALISOINTI
    # --------------------------------------------------------

    score = normalize(
        score
    )

    score[
        ~land
    ] = 0
    #plt.imshow(score)
    #plt.show()
    #quit(-1)
    return score





# ============================================================
# KAUPPAPOTENTIAALI
# ============================================================

def calculate_trade_potential(
    civilization,
    coastal_proximity,
    rivers
):

    river_mouths = (
        rivers
        *
        coastal_proximity
    )

    trade = (
        civilization
        * 0.60
        +
        coastal_proximity
        * 0.30
        +
        river_mouths
        * 1.20
    )

    return normalize(
        trade
    )


# ============================================================
# SIVILISAATIOKESKUSTEN VALINTA
# ============================================================

def select_civilization_centers(
    civilization,
    number=3,
    min_distance=12
):

    field = civilization.copy()

    centers = []

    yy, xx = np.indices(
        field.shape
    )

    for _ in range(number):

        y, x = np.unravel_index(
            np.argmax(field),
            field.shape
        )

        value = field[y, x]

        if value <= 0.05:
            break

        centers.append(
            (y, x, value)
        )

        distance = np.sqrt(
            (yy - y) ** 2
            +
            (xx - x) ** 2
        )

        field[
            distance < min_distance
        ] = 0

    return centers[:4]




# ============================================================
# HILLSHADE
# ============================================================

def create_hillshade(
    elevation
):

    smooth = gaussian_filter(
        elevation,
        sigma=(1.2, 1.2),
        mode=("nearest", "wrap")
    )

    dy, dx = np.gradient(
        smooth
    )

    azimuth = np.radians(
        315
    )

    altitude = np.radians(
        45
    )

    slope = np.pi / 2 - np.arctan(
        np.sqrt(
            dx ** 2
            +
            dy ** 2
        )
    )

    aspect = np.arctan2(
        -dx,
        dy
    )

    illumination = (
        np.sin(altitude)
        * np.sin(slope)
        +
        np.cos(altitude)
        * np.cos(slope)
        *
        np.cos(
            azimuth - aspect
        )
    )

    return normalize(
        illumination
    )


# ============================================================
# RELIEF-SHADOW
# ============================================================

def create_relief_shadow(
    elevation
):

    shadow = np.zeros_like(
        elevation,
        dtype=float
    )

    directions = [
        (1, 1),
        (2, 2),
        (4, 4),
        (7, 7),
        (12, 12)
    ]

    for dy, dx in directions:

        shifted = np.roll(
            elevation,
            shift=(
                dy,
                dx
            ),
            axis=(
                0,
                1
            )
        )

        difference = (
            elevation
            -
            shifted
        )

        shadow += np.maximum(
            difference,
            0
        )

    return normalize(
        shadow
    )


# ============================================================
# VUORIJONOKARTTA
# ============================================================

def draw_mountain_ranges(
    ridge,
    extent
):

    plt.figure(
        figsize=(16, 8)
    )

    plt.imshow(
        ridge,
        cmap="magma",
        extent=extent,
        aspect="auto"
    )

    plt.colorbar(
        label="Vuorijonopotentiaali"
    )

    plt.title(
        "TERRESTRIC V8.3 — tektonisten vuorijonojen rakenne"
    )

    plt.xlabel(
        "Pituusaste"
    )

    plt.ylabel(
        "Leveysaste"
    )

    plt.grid(
        alpha=0.12
    )

    plt.tight_layout()


# ============================================================
# MAAILMANKARTTA
# ============================================================

def draw_world_relief(
    elevation,
    land_mask,
    rivers,
    civilization,
    trade_potential,
    biomes,
    centers
):

    fig, ax = plt.subplots(
        figsize=(18, 9)
    )

    extent = [
        -180,
        180,
        -90,
        90
    ]

    sea = np.zeros(
        (
            HEIGHT,
            WIDTH,
            3
        )
    )

    sea[:, :, 0] = 0.035
    sea[:, :, 1] = 0.16
    sea[:, :, 2] = 0.25

    ax.imshow(
        sea,
        extent=extent,
        aspect="auto"
    )

    hillshade = create_hillshade(
        elevation
    )

    relief_shadow = create_relief_shadow(
        elevation
    )

    terrain = plt.cm.terrain(
        normalize(
            elevation
        )
    )[:, :, :3]

    shade = (
        0.62
        +
        hillshade * 0.38
    )

    shade *= (
        0.88
        +
        relief_shadow * 0.18
    )

    terrain *= (
        shade[:, :, None]
    )

    terrain[
        ~land_mask
    ] = 0

    ax.imshow(
        terrain,
        extent=extent,
        aspect="auto"
    )

    river_mask = (
        rivers > 0.12
    )

    ax.imshow(
        river_mask,
        cmap=ListedColormap([
            (0, 0, 0, 0),
            (0.03, 0.35, 0.85, 0.80)
        ]),
        extent=extent,
        aspect="auto"
    )

    trade_layer = np.where(
        trade_potential > 0.62,
        trade_potential,
        np.nan
    )

    ax.imshow(
        trade_layer,
        cmap="YlOrBr",
        alpha=0.20,
        extent=extent,
        aspect="auto"
    )

    civ_layer = np.where(
        civilization > 0.60,
        civilization,
        np.nan
    )

    ax.imshow(
        civ_layer,
        cmap="inferno",
        alpha=0.30,
        extent=extent,
        aspect="auto"
    )

    for y, x, value in centers:

        lon = longitude[
            x
        ]

        lat = latitude[
            y
        ]

        size = (
            15
            +
            value * 70
        )

        ax.scatter(
            lon,
            lat,
            s=size,
            c="#ffd166",
            edgecolors="#301934",
            linewidths=0.6,
            zorder=10
        )
    for y, x, value in centers[0:1]:

        lon = longitude[
            x
        ]

        lat = latitude[
            y
        ]

        size = (
            200
        )

        ax.scatter(
            lon,
            lat,
            s=size,
            c="#ff0000",
            edgecolors="#300000",
            linewidths=1,
            zorder=10
        )

    ax.set_title(
        f"TERRESTRIC V8.3 — relief, joet, kauppa ja sivilisaatio — seed {SEED}",
        fontsize=17
    )

    ax.set_xlabel(
        "Pituusaste"
    )

    ax.set_ylabel(
        "Leveysaste"
    )

    ax.grid(
        alpha=0.15
    )

    plt.tight_layout()

    return fig, ax


# ============================================================
# PÄÄOHJELMA
# ============================================================

print()
print("=" * 60)
print("TERRESTRIC V8.3")
print("=" * 60)


# ============================================================
# 1. MANTEREET
# ============================================================

print(
    "1/18 Mantereet..."
)

land, continent_data = (
    generate_continents()
)

land = add_islands(
    land
)


# ============================================================
# 2. TEKTONISET LAATAT
# ============================================================

print(
    "2/18 Tektoniset laatat..."
)

plates = create_plates()

plate_map = create_plate_map(
    plates
)

boundary = find_boundaries(
    plate_map
)

boundary_type = (
    classify_boundaries(
        plate_map,
        plates
    )
)


# ============================================================
# 3–9. KORKEUS / GEOLOGIA
# ============================================================

print(
    "3/18 Korkeus..."
)

# Alustetaan aina, jotta loppupään kartat toimivat myös
# ulkoisen DEM-ohituksen kanssa.
mountain_ridge = np.zeros(
    (HEIGHT, WIDTH),
    dtype=float
)


if USE_EXTERNAL_DEM:

    # ========================================================
    # ULKOINEN DEM — OHITUSKAISTA
    # ========================================================

    elevation = load_external_dem(
        EXTERNAL_DEM_PATH,
        WIDTH,
        HEIGHT,
        EXTERNAL_DEM_MODE
    )

    # Ulkoinen DEM määrää myös maa/meri-jaon.
    # Tässä land toimii jatkuvana heightmapina, kuten Telluruksen
    # normaalissa mantereiden generoinnissa.
    land = elevation.copy()

    print(
        "    Ulkoinen DEM käytössä — geologinen generointi ohitettu."
    )

    # Ulkoisen DEM:n tapauksessa tektonista mountain_ridge-dataa
    # ei synny, koska kohdat 4–9 ohitetaan. Muodostetaan siksi
    # DEM:stä diagnostinen vuoristopotentiaali, jotta loppupään
    # kartat ja yhteenveto voivat käyttää samaa muuttujaa.
    high_terrain = np.maximum(
        elevation - 0.55,
        0.0
    )

    mountain_ridge = normalize(
        gaussian_filter(
            high_terrain,
            sigma=(2.0, 4.0),
            mode=("nearest", "wrap")
        )
    )


else:

    # ========================================================
    # 3. PERUSKORKEUS
    # ========================================================

    elevation = (
        create_base_elevation(
            land
        )
    )


    # ========================================================
    # 4. KRATONIT
    # ========================================================

    print(
        "4/18 Kratonit..."
    )

    elevation = add_cratons(
        elevation,
        land
    )


    # ========================================================
    # 5. VUORIJONOT
    # ========================================================

    print(
        "5/18 Vuorijonot..."
    )

    mountain_ridge = (
        create_mountain_ridge(
            boundary_type,
            land
        )
    )

    elevation = add_mountains(
        elevation,
        land,
        boundary_type
    )


    # ========================================================
    # 6. TULIVUORET
    # ========================================================

    print(
        "6/18 Tulivuoret..."
    )

    elevation = add_volcanic_arcs(
        elevation,
        land,
        boundary_type
    )


    # ========================================================
    # 7. RIFTIT
    # ========================================================

    print(
        "7/18 Riftit..."
    )

    elevation = add_rifts(
        elevation,
        boundary_type
    )


    # ========================================================
    # 8. MAAN PINTA
    # ========================================================

    print(
        "8/18 Maaston pienimuodot..."
    )

    terrain_noise = spherical_noise(
        frequencies=[
            2.0,
            4.0,
            8.0,
            16.0
        ],

        amplitudes=[
            1.0,
            0.40,
            0.15,
            0.05
        ],

        base_offset=8000
    )

    elevation = (
        elevation * 0.93
        +
        terrain_noise
        * TERRAIN_NOISE_STRENGTH
    )


    # ========================================================
    # 9. EROOSIO
    # ========================================================

    print(
        "9/18 Eroosio..."
    )

    for _ in range(
        4
    ):

        elevation = erode_terrain(
            elevation
        )

    elevation = normalize(
        elevation
    )


# ============================================================
# 10. MAA / MERI
# ============================================================

print(
    "10/18 Merenpinta..."
)

land_mask = (
    elevation >= SEA_LEVEL
)

ocean_mask = (
    ~land_mask
)



land_coverage = (
    land_mask.mean()
)

ocean_coverage = (
    ocean_mask.mean()
)

region_map, region_area_map, region_size_factor_map, regions = \
    analyze_land_regions(
        land_mask
    )



#plt.imshow(region_size_factor_map)

#plt.show()


#quit(-1)



ocean_humidity_factor = (
    calculate_ocean_coverage_factor(
        ocean_coverage
    )
)

print(
    "    Maa:",
    round(
        land_coverage * 100,
        2
    ),
    "%"
)

print(
    "    Meri:",
    round(
        ocean_coverage * 100,
        2
    ),
    "%"
)

print(
    "    Merien kosteuskerroin:",
    round(
        ocean_humidity_factor,
        3
    )
)


# ============================================================
# 11. JOET
# ============================================================

print(
    "11/18 Joet..."
)

rivers = generate_rivers(
    elevation,
    land,
    number=NUM_RIVERS
)


# ============================================================
# 12. TUULET
# ============================================================

print(
    "12/18 Globaalit tuulet..."
)

wind_u, wind_v = (
    calculate_global_winds()
)


# ============================================================
# 13. MERIVIRRAT
# ============================================================

print(
    "13/18 Merivirrat..."
)

ocean_temperature = (
    calculate_ocean_temperature()
)

current_u, current_v = (
    calculate_ocean_currents(
        wind_u,
        wind_v
    )
)

ocean_current_effect = (
    calculate_ocean_current_temperature(
        ocean_temperature,
        current_u,
        current_v
    )
)


# ============================================================
# 14. LÄMPÖ + VUODENAJAT
# ============================================================

print(
    "14/18 Vuodenajat..."
)

base_temperature = (
    calculate_temperature(
        elevation,
        land_mask,
        ocean_temperature,
        ocean_current_effect
    )
)

summer_temperature, winter_temperature = (
    calculate_seasonal_temperature(
        base_temperature,
        land_mask,
        ocean_temperature
    )
)


# ============================================================
# 15. ILMASTO
# ============================================================

print(
    "15/18 Ilmasto..."
)

continentality = (
    calculate_continentality(
        land_mask
    )
)

monsoon = calculate_monsoon(
    base_temperature,
    ocean_temperature,
    land_mask,
    elevation,
    wind_u,
    wind_v
)

summer_rainfall = (
    calculate_seasonal_rainfall(
        summer_temperature,
        elevation,
        land_mask,
        wind_u,
        wind_v,
        ocean_temperature,
        monsoon,
        continentality,
        ocean_humidity_factor,
        "summer"
    )
)

winter_rainfall = (
    calculate_seasonal_rainfall(
        winter_temperature,
        elevation,
        land_mask,
        wind_u,
        wind_v,
        ocean_temperature,
        monsoon,
        continentality,
        ocean_humidity_factor,
        "winter"
    )
)

annual_rainfall = (
    calculate_annual_rainfall(
        summer_rainfall,
        winter_rainfall
    )
)

rainfall_seasonality = (
    calculate_rainfall_seasonality(
        summer_rainfall,
        winter_rainfall
    )
)

mean_temperature = (
    summer_temperature
    +
    winter_temperature
) / 2


# ============================================================
# SADEVARJO ERILLISENÄ KENTTÄNÄ
# ============================================================

print(
    "    Lasketaan vuoristojen sadevarjot..."
)

rain_shadow = calculate_rain_shadow(
    elevation,
    land_mask,
    wind_u,
    wind_v
)

print(
    "    Sadevarjon maksimi:",
    round(
        rain_shadow.max(),
        3
    )
)


# ============================================================
# 16. KASVUKAUSI + BIOMIT
# ============================================================

print(
    "16/18 Biomit..."
)

growing_season = (
    calculate_growing_season(
        summer_temperature,
        winter_temperature,
        summer_rainfall,
        winter_rainfall,
        land_mask
    )
)

biomes = classify_biomes(
    mean_temperature,
    annual_rainfall,
    rainfall_seasonality,
    land_mask
)


# ============================================================
# 17. RESURSSIT + SIVILISAATIO
# ============================================================

print(
    "17/18 Resurssit ja sivilisaatio..."
)

coastal_proximity = (
    calculate_coastal_proximity(
        land_mask
    )
)

mountain_proximity = (
    calculate_mountain_proximity(
        elevation,
        land_mask
    )
)

mountain_resources = (
    calculate_mountain_resources(
        elevation,
        mountain_proximity,
        annual_rainfall
    )
)

civilization = civilization_potential(
    mean_temperature,
    annual_rainfall,
    elevation,
    rivers,
    land_mask, region_size_factor_map
)

trade_potential = (
    calculate_trade_potential(
        civilization,
        coastal_proximity,
        rivers
    )
)

MAX_NUM_PRIMARY_CIVS=3

centers = (
    select_civilization_centers(
        civilization,
        number=NUM_CIV_CENTERS
    )
)


# ============================================================
# 18. KARTAT
# ============================================================

print(
    "18/18 Kartat..."
)

extent = [
    -180,
    180,
    -90,
    90
]

fig, axes = plt.subplots(
    4,
    4,
    figsize=(20, 16)
)


# ============================================================
# 1. MANNER
# ============================================================

axes[0, 0].imshow(
    land_mask,
    cmap="Blues_r",
    extent=extent,
    aspect="auto"
)

axes[0, 0].set_title(
    "Mannerjakauma"
)


# ============================================================
# 2. LAATAT
# ============================================================

axes[0, 1].imshow(
    plate_map,
    cmap="tab20",
    extent=extent,
    aspect="auto"
)

axes[0, 1].contour(
    boundary,
    levels=[0.5],
    colors="black",
    linewidths=0.45,
    extent=extent
)

axes[0, 1].set_title(
    "Epäsäännölliset tektoniset laatat"
)


# ============================================================
# 3. RAJATYYPIT
# ============================================================

axes[0, 2].imshow(
    boundary_type,
    cmap=ListedColormap([
        "#000000",
        "#d62828",
        "#2a9d8f",
        "#f4a261"
    ]),
    extent=extent,
    aspect="auto",
    vmin=0,
    vmax=3
)

axes[0, 2].set_title(
    "Laattojen rajatyypit"
)


# ============================================================
# 4. VUORIJONOT
# ============================================================

axes[0, 3].imshow(
    mountain_ridge,
    cmap="magma",
    extent=extent,
    aspect="auto"
)

axes[0, 3].set_title(
    "Vuorijonojen rakenne"
)


# ============================================================
# 5. KORKEUS
# ============================================================

im = axes[1, 0].imshow(
    elevation,
    cmap="terrain",
    extent=extent,
    aspect="auto"
)

axes[1, 0].set_title(
    "Korkeuskartta"
)

plt.colorbar(
    im,
    ax=axes[1, 0],
    shrink=0.8
)


# ============================================================
# 6. RELIEF
# ============================================================

hillshade = create_hillshade(
    elevation
)

axes[1, 1].imshow(
    elevation,
    cmap="terrain",
    extent=extent,
    aspect="auto"
)

axes[1, 1].imshow(
    hillshade,
    cmap="gray",
    alpha=0.35,
    extent=extent,
    aspect="auto"
)

axes[1, 1].set_title(
    "Relief / hillshade"
)


# ============================================================
# 7. JOET
# ============================================================

axes[1, 2].imshow(
    elevation,
    cmap="terrain",
    extent=extent,
    aspect="auto"
)

axes[1, 2].imshow(
    rivers,
    cmap="Blues",
    alpha=0.80,
    extent=extent,
    aspect="auto"
)

axes[1, 2].set_title(
    "Jokiverkosto"
)


# ============================================================
# 8. VUORISTOT + RAJAT
# ============================================================

axes[1, 3].imshow(
    elevation,
    cmap="terrain",
    extent=extent,
    aspect="auto"
)

axes[1, 3].imshow(
    mountain_ridge,
    cmap="Reds",
    alpha=0.45,
    extent=extent,
    aspect="auto"
)

axes[1, 3].contour(
    boundary_type == 1,
    levels=[0.5],
    colors="white",
    linewidths=0.35,
    extent=extent
)

axes[1, 3].set_title(
    "Konvergenssi + vuorijonot"
)


# ============================================================
# 9. TUULET
# ============================================================

axes[2, 0].imshow(
    land_mask,
    cmap=ListedColormap([
        "#183b4a",
        "#c9a66b"
    ]),
    extent=extent,
    aspect="auto"
)

skip_y = 15
skip_x = 30

axes[2, 0].quiver(
    LON[
        ::skip_y,
        ::skip_x
    ],
    LAT[
        ::skip_y,
        ::skip_x
    ],
    wind_u[
        ::skip_y,
        ::skip_x
    ],
    wind_v[
        ::skip_y,
        ::skip_x
    ],
    color="white",
    alpha=0.65,
    scale=18
)

axes[2, 0].set_title(
    "Globaalit tuulet"
)


# ============================================================
# 10. MERIVIRRAT
# ============================================================

axes[2, 1].imshow(
    ocean_temperature,
    cmap="RdYlBu_r",
    extent=extent,
    aspect="auto"
)

axes[2, 1].quiver(
    LON[
        ::skip_y,
        ::skip_x
    ],
    LAT[
        ::skip_y,
        ::skip_x
    ],
    current_u[
        ::skip_y,
        ::skip_x
    ],
    current_v[
        ::skip_y,
        ::skip_x
    ],
    color="black",
    alpha=0.55,
    scale=18
)

axes[2, 1].set_title(
    "Merivirrat"
)


# ============================================================
# 11. KESÄLÄMPÖ
# ============================================================

im = axes[2, 2].imshow(
    summer_temperature,
    cmap="RdYlBu_r",
    extent=extent,
    aspect="auto"
)

axes[2, 2].set_title(
    "Kesälämpötila"
)

plt.colorbar(
    im,
    ax=axes[2, 2],
    shrink=0.8,
    label="°C"
)


# ============================================================
# 12. TALVILÄMPÖ
# ============================================================

im = axes[2, 3].imshow(
    winter_temperature,
    cmap="RdYlBu_r",
    extent=extent,
    aspect="auto"
)

axes[2, 3].set_title(
    "Talvilämpötila"
)

plt.colorbar(
    im,
    ax=axes[2, 3],
    shrink=0.8,
    label="°C"
)


# ============================================================
# 13. VUOSISADE
# ============================================================

im = axes[3, 0].imshow(
    annual_rainfall,
    cmap="YlGnBu",
    extent=extent,
    aspect="auto"
)

axes[3, 0].set_title(
    "Vuosisade + sadevarjot"
)

plt.colorbar(
    im,
    ax=axes[3, 0],
    shrink=0.8,
    label="mm/v"
)


# ============================================================
# 14. BIOMIT
# ============================================================

biome_colors = [

    "#184a63",
    "#e8d18a",
    "#c8b86a",
    "#43834b",
    "#176b3a",
    "#8ca6a3",
    "#f4f4f4"
]

biome_map = ListedColormap(
    biome_colors
)

axes[3, 1].imshow(
    biomes,
    cmap=biome_map,
    extent=extent,
    aspect="auto",
    vmin=0,
    vmax=6
)

axes[3, 1].set_title(
    "Biomit"
)


# ============================================================
# 15. SIVILISAATIO
# ============================================================

im = axes[3, 2].imshow(
    civilization,
    cmap="inferno",
    extent=extent,
    aspect="auto"
)

axes[3, 2].set_title(
    "Sivilisaatiopotentiaali"
)

plt.colorbar(
    im,
    ax=axes[3, 2],
    shrink=0.8
)


# ============================================================
# 16. KAUPPA
# ============================================================

im = axes[3, 3].imshow(
    trade_potential,
    cmap="magma",
    extent=extent,
    aspect="auto"
)

axes[3, 3].set_title(
    "Kauppapotentiaali"
)

plt.colorbar(
    im,
    ax=axes[3, 3],
    shrink=0.8
)


# ============================================================
# VIIMEISTELY
# ============================================================

for row in axes:

    for ax in row:

        ax.set_xlabel(
            "pituusaste"
        )

        ax.set_ylabel(
            "leveysaste"
        )

        ax.grid(
            alpha=0.10
        )


plt.suptitle(
    f"TERRESTRIC V8.3 — seed {SEED}",
    fontsize=20
)

plt.tight_layout()

plt.show()


# ============================================================
# ERILLINEN MAAILMANKARTTA
# ============================================================

draw_world_relief(
    elevation,
    land_mask,
    rivers,
    civilization,
    trade_potential,
    biomes,
    centers
)

plt.show()


# ============================================================
# VUORIJONOKARTTA
# ============================================================

draw_mountain_ranges(
    mountain_ridge,
    extent
)

plt.show()


# ============================================================
# SIVILISAATIOKARTTA
# ============================================================

plt.figure(
    figsize=(16, 8)
)

plt.imshow(
    civilization,
    cmap="inferno",
    extent=extent,
    aspect="auto"
)

plt.colorbar(
    label="Sivilisaatiopotentiaali"
)

plt.contour(
    land_mask,
    levels=[0.5],
    colors="white",
    linewidths=0.5,
    extent=extent,
    origin="upper"
)

plt.title(
    "TERRESTRIC V8.3 — sivilisaatiopotentiaali"
)

plt.xlabel(
    "Pituusaste"
)

plt.ylabel(
    "Leveysaste"
)

plt.grid(
    alpha=0.12
)

plt.tight_layout()

plt.show()


# ============================================================
# MANTEREISUUSKARTTA
# ============================================================

plt.figure(
    figsize=(16, 8)
)

plt.imshow(
    continentality,
    cmap="YlOrBr",
    extent=extent,
    aspect="auto",
    vmin=0,
    vmax=1
)

plt.colorbar(
    label="Mantereisuus"
)

plt.contour(
    land_mask,
    levels=[0.5],
    colors="black",
    linewidths=0.4,
    extent=extent, origin="upper"
)

plt.title(
    "TERRESTRIC V8.3 — mantereisuus"
)

plt.xlabel(
    "Pituusaste"
)

plt.ylabel(
    "Leveysaste"
)

plt.grid(
    alpha=0.12
)

plt.tight_layout()

plt.show()


# ============================================================
# SADE + MANTEREISUUS
# ============================================================

plt.figure(
    figsize=(16, 8)
)

plt.imshow(
    annual_rainfall,
    cmap="YlGnBu",
    extent=extent,
    aspect="auto"
)

plt.colorbar(
    label="Vuosisade mm/v"
)

plt.contour(
    continentality,
    levels=[
        0.25,
        0.50,
        0.75
    ],
    colors=[
        "white",
        "orange",
        "red"
    ],
    linewidths=0.5,
    extent=extent, origin="upper"
)

plt.title(
    "TERRESTRIC V8.3 — vuosisade + mantereisuus"
)

plt.xlabel(
    "Pituusaste"
)

plt.ylabel(
    "Leveysaste"
)

plt.grid(
    alpha=0.12
)

plt.tight_layout()

plt.show()


# ============================================================
# TUULEN KOSTEUSVAIKUTUS
# ============================================================

wind_rain_effect = (
    calculate_wind_rain_effect(
        wind_u,
        wind_v,
        land_mask
    )
)

plt.figure(
    figsize=(16, 8)
)

plt.imshow(
    wind_rain_effect,
    cmap="RdYlGn",
    extent=extent,
    aspect="auto",
    vmin=-1,
    vmax=1
)

plt.colorbar(
    label="Tuulen meri → maa / maa → meri -vaikutus"
)

plt.contour(
    land_mask,
    levels=[0.5],
    colors="black",
    linewidths=0.5,
    extent=extent, origin="upper"
)

plt.title(
    "TERRESTRIC V8.3 — tuulen vaikutus kosteuden kulkeutumiseen"
)

plt.xlabel(
    "Pituusaste"
)

plt.ylabel(
    "Leveysaste"
)

plt.grid(
    alpha=0.12
)

plt.tight_layout()

plt.show()


# ============================================================
# TUULEN KULJETTAMA KOSTEUS
# ============================================================

wind_moisture = (
    calculate_wind_moisture(
        wind_u,
        wind_v,
        ocean_mask,
        ocean_temperature,
        land_mask
    )
)

plt.figure(
    figsize=(16, 8)
)

plt.imshow(
    wind_moisture,
    cmap="YlGnBu",
    extent=extent,
    aspect="auto"
)

plt.colorbar(
    label="Tuulen kuljettama merikosteus"
)

plt.contour(
    land_mask,
    levels=[0.5],
    colors="black",
    linewidths=0.5,
    extent=extent, origin="upper"
)

plt.title(
    "TERRESTRIC V8.3 — tuulen kuljettama merikosteus"
)

plt.xlabel(
    "Pituusaste"
)

plt.ylabel(
    "Leveysaste"
)

plt.grid(
    alpha=0.12
)

plt.tight_layout()

plt.show()


# ============================================================
# VUORISTON SADEVARJOKARTTA
# ============================================================

plt.figure(
    figsize=(16, 8)
)

shadow_display = (
    rain_shadow
    *
    RAIN_SHADOW_STRENGTH
)

plt.imshow(
    shadow_display,
    cmap="YlOrBr",
    extent=extent,
    aspect="auto"
)

plt.colorbar(
    label="Sadevarjon kuivattava voimakkuus"
)

plt.contour(
    land_mask,
    levels=[0.5],
    colors="black",
    linewidths=0.5,
    extent=extent, origin="upper"
)

plt.title(
    "TERRESTRIC V8.3 — vuoristojen sadevarjot"
)

plt.xlabel(
    "Pituusaste"
)

plt.ylabel(
    "Leveysaste"
)

plt.grid(
    alpha=0.12
)

plt.tight_layout()

plt.show()


# ============================================================
# VUORISTO + SADEVARJO
# ============================================================

plt.figure(
    figsize=(16, 8)
)

plt.imshow(
    elevation,
    cmap="terrain",
    extent=extent,
    aspect="auto"
)

plt.imshow(
    rain_shadow,
    cmap="Purples",
    alpha=0.55,
    extent=extent,
    aspect="auto"
)

plt.contour(
    mountain_ridge,
    levels=[
        0.35,
        0.55,
        0.75
    ],
    colors=[
        "white",
        "orange",
        "red"
    ],
    linewidths=0.6,
    extent=extent, origin="upper"
)

plt.colorbar(
    label="Sadevarjo"
)

plt.title(
    "TERRESTRIC V8.3 — vuoristot ja niiden sadevarjot"
)

plt.xlabel(
    "Pituusaste"
)

plt.ylabel(
    "Leveysaste"
)

plt.grid(
    alpha=0.12
)

plt.tight_layout()

plt.show()


# ============================================================
# PALLOLLISET PINTA-ALAPAINOT
# ============================================================

latitude_weight = np.cos(
    np.radians(LAT)
)

latitude_weight /= (
    latitude_weight.mean()
)


# ============================================================
# GLOBAALI SADEKESKIARVO
# ============================================================

global_mean_rainfall = np.average(
    annual_rainfall,
    weights=latitude_weight
)


# ============================================================
# MAA-ALUEEN SADEKESKIARVO
# ============================================================

land_weights = (
    latitude_weight
    *
    land_mask
)

if np.sum(land_weights) > 0:

    land_mean_rainfall = np.average(
        annual_rainfall,
        weights=land_weights
    )

else:

    land_mean_rainfall = 0.0


# ============================================================
# MERIALUEEN SADEKESKIARVO
# ============================================================

ocean_weights = (
    latitude_weight
    *
    ocean_mask
)

if np.sum(ocean_weights) > 0:

    ocean_mean_rainfall = np.average(
        annual_rainfall,
        weights=ocean_weights
    )

else:

    ocean_mean_rainfall = 0.0


# ============================================================
# MANTEREISUUDEN TILASTOT
# ============================================================

land_continentality = (
    continentality[land_mask]
)

if len(
    land_continentality
) > 0:

    mean_continentality = np.average(
        land_continentality,
        weights=latitude_weight[
            land_mask
        ]
    )

else:

    mean_continentality = 0.0


# ============================================================
# SADETILASTOT
# ============================================================

land_rain = (
    annual_rainfall[
        land_mask
    ]
)

if len(land_rain) > 0:

    land_rain_median = (
        np.median(
            land_rain
        )
    )

    land_rain_min = (
        np.percentile(
            land_rain,
            5
        )
    )

    land_rain_max = (
        np.percentile(
            land_rain,
            95
        )
    )

else:

    land_rain_median = 0
    land_rain_min = 0
    land_rain_max = 0


# ============================================================
# TUULEN KOSTEUSTILASTOT
# ============================================================

land_wind_effect = (
    wind_rain_effect[
        land_mask
    ]
)

if len(
    land_wind_effect
) > 0:

    mean_wind_effect = np.average(
        land_wind_effect,
        weights=latitude_weight[
            land_mask
        ]
    )

    strongest_onshore = (
        np.percentile(
            land_wind_effect,
            95
        )
    )

    strongest_offshore = (
        np.percentile(
            land_wind_effect,
            5
        )
    )

else:

    mean_wind_effect = 0.0
    strongest_onshore = 0.0
    strongest_offshore = 0.0


# ============================================================
# SADEVARJON TILASTOT
# ============================================================

land_shadow = (
    rain_shadow[
        land_mask
    ]
)

if len(
    land_shadow
) > 0:

    mean_rain_shadow = np.average(
        land_shadow,
        weights=latitude_weight[
            land_mask
        ]
    )

    shadow_p95 = np.percentile(
        land_shadow,
        95
    )

    shadow_p99 = np.percentile(
        land_shadow,
        99
    )

else:

    mean_rain_shadow = 0.0
    shadow_p95 = 0.0
    shadow_p99 = 0.0


# ============================================================
# TEKSTIYHTEENVETO
# ============================================================

print()
print("=" * 60)
print("TERRESTRIC V8.3 — YHTEENVETO")
print("=" * 60)

print()

print(
    "Maa-ala:",
    round(
        land_coverage * 100,
        2
    ),
    "%"
)

print(
    "Meriala:",
    round(
        ocean_coverage * 100,
        2
    ),
    "%"
)

print()

print(
    "Keskilämpötila:",
    round(
        np.mean(
            mean_temperature
        ),
        2
    ),
    "°C"
)

print()

print(
    "Globaali keskimääräinen vuosisade:",
    round(
        global_mean_rainfall,
        1
    ),
    "mm/v"
)

print(
    "Maa-alueiden keskimääräinen vuosisade:",
    round(
        land_mean_rainfall,
        1
    ),
    "mm/v"
)

print(
    "Merialueiden keskimääräinen sade:",
    round(
        ocean_mean_rainfall,
        1
    ),
    "mm/v"
)

print()

print(
    "Maa-alueiden mediaanisade:",
    round(
        land_rain_median,
        1
    ),
    "mm/v"
)

print(
    "Maa-alueiden sade P5:",
    round(
        land_rain_min,
        1
    ),
    "mm/v"
)

print(
    "Maa-alueiden sade P95:",
    round(
        land_rain_max,
        1
    ),
    "mm/v"
)

print()

print(
    "Keskimääräinen mantereisuus:",
    round(
        mean_continentality,
        3
    )
)

print(
    "Merien kosteuskerroin:",
    round(
        ocean_humidity_factor,
        3
    )
)

print(
    "Globaali sademäärän skaala:",
    GLOBAL_RAINFALL_SCALE
)

print()

print(
    "Keskimääräinen tuulen kosteussiirtymä:",
    round(
        mean_wind_effect,
        3
    )
)

print(
    "Voimakkain mereltä maalle -alue P95:",
    round(
        strongest_onshore,
        3
    )
)

print(
    "Voimakkain maalta merelle -alue P5:",
    round(
        strongest_offshore,
        3
    )
)

print()

print(
    "Keskimääräinen sadevarjo:",
    round(
        mean_rain_shadow,
        4
    )
)

print(
    "Sadevarjo P95:",
    round(
        shadow_p95,
        4
    )
)

print(
    "Sadevarjo P99:",
    round(
        shadow_p99,
        4
    )
)

print()

print(
    "Sivilisaatiokeskuksia:",
    len(
        centers
    )
)

print(
    "Korkein sivilisaatiopotentiaali:",
    round(
        civilization.max(),
        3
    )
)

print(
    "Korkein kauppapotentiaali:",
    round(
        trade_potential.max(),
        3
    )
)

print(
    "Vuorijonopotentiaali:",
    round(
        mountain_ridge.max(),
        3
    )
)

print()

print(
    "Laattoja:",
    NUM_PLATES
)

print(
    "Laattojen distortion:",
    PLATE_DISTORTION
)

print()

print("=" * 60)


print(" Civilization centers")

#print(centers)


print(" X Y weight")
for n in range(0,len(centers)):
	print(centers[n][0], centers[n][1],centers[n][2])

