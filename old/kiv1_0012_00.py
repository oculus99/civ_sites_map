
#######################################
#
## Create fractal world climate and biomes
##
## estimate from dem and climate mean values
##
## human-like specie birth location
## agriculture birth and spreding 
## primary civilization areas
#
## simple mean t, deltat approach
##
## 04.09.2026 0000.0012.00
##
######################################

import numpy as np
import math

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from scipy.ndimage import (
    gaussian_filter,generic_filter,uniform_filter,
    distance_transform_edt, map_coordinates
)

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


from PIL import Image

from noise import pnoise3

import rasterio

import heapq



##seed1=42
#seed1=333

#seed1=1246
#seed1=32
#seed1=321
#seed1=42
#seed1=88
#seed1=123
#seed1=53
#seed1=64
seed1=73

base_noisescale=0.6
distort_noisescale=0.75
distort_coeff=0.1

will_load_image=False

#imagename='orogen1.png'
imagename='terra.png'

sealevel=0.6
dem_min=-6000
dem_max=5000

width=360*4
height=180*4

#width=360
#height=180

## NOTE theres affects only some climate props!!!

#ecc=0.0167
#tilt=23.44
#mvelp=102.7

## basic parematers of planet, here terra-like composition

#S1=1361*(1/1) ## S1 solar constant W m-2
#ecc=0.0167 ## eccentricity
#tilt=23.44 ## axis tilt or obliquity
#mvelp=102.7 ## position of axis against orbit


distau=1.00
starlum=1
starteff=5778

Sk=starlum/math.sqrt(distau)

S1=1361*Sk*math.pow((starteff/5778), -0.8) ## S1 solar constant W m-2
ecc=0.0167*1 ## eccentricity
tilt=23.44 ## axis tilt or obliquity
mvelp=102.7 ## position of axis against orbit


air_pressure_atm=1
atmos_co2_ppm= 280
planet_mass_me=1
rotation_period_days=1
orbital_period_years=1

planet_radius_re=math.pow(planet_mass_me, 0.27) ## terra-like internal composition 
planet_radius=6371*planet_radius_re #3 km

#mean_temp=15
mean_temp=15
temp_diff=70**math.sqrt(rotation_period_days)*math.sqrt(planet_radius_re)/air_pressure_atm

temp_dev=temp_diff/2
polar_temp=mean_temp-temp_dev
temp_diff=mean_temp+temp_dev

SIGMA = 5.670374419e-8
#quit(-1)


def load_image_and_normalize(imagename):
    try:
        with Image.open(imagename) as img:
            leveys, korkeus = img.size
            print(f"Image '{imagename}' loaded.")
            print(f"Width: {leveys} px")
            print(f"Height: {korkeus} px")
            harmaa_kuva = img.convert('L')
            kuva_taulukko = np.array(img)
            kuva_min=np.min(kuva_taulukko)
            kuva_max=np.max(kuva_taulukko)
            kuva_delta=kuva_max-kuva_min
            normalized_image= (kuva_taulukko-kuva_min)/kuva_delta
            return(normalized_image, width, height)
    except FileNotFoundError:
        print(f"Virhe: file '{imagename}' not found")
        return(None, None, None)

def normalize(taulukko):
    """
    Normalisoi NumPy-taulukon arvot lineaarisesti välille.
    """
    min_arvo = np.min(taulukko)
    max_arvo = np.max(taulukko)
    
    # Estetään nollalla jakaminen, jos kaikki taulukon arvot ovat samoja
    if min_arvo == max_arvo:
        return np.zeros_like(taulukko, dtype=float)
        
    return (taulukko - min_arvo) / (max_arvo - min_arvo)

def sigmoid_meri_manner_jakauma(dem_taulukko, säätökerroin=1.0):
    """
    Muuntaa DEM-datan välille 0-1 sigmoidilla siten, että
    merenpinnan taso (arvo 0.5) asettuu 71. persentiilin kohdalle.
    
    Parametrit:
    - dem_taulukko: NumPy-taulukko korkeusdatasta
    - säätökerroin: Kontrasti. Pieni arvo (esim. 0.05) tekee siirtymästä 
                    loivan, suuri arvo (esim. 2.0) tekee rajasta jyrkän.
    """
    # Etsitään korkeus, jonka alapuolella on 71 % datasta (merenpinta)
    merenpinnan_taso = np.percentile(dem_taulukko, 71)
    
    # Keskitetään data niin, että kuvitteellinen merenpinta on 0
    z = säätökerroin * (dem_taulukko - merenpinnan_taso)
    
    # Sigmoid-muunnos
    return 1 / (1 + np.exp(-z))



def sigmoid_dem(dem_taulukko, saatokerroin=1.0, keskiarvo=None):
    """
    Muuntaa DEM-taulukon arvot välille 0-1 sigmoid-funktiolla.
    
    Parametrit:
    - dem_taulukko: NumPy-taulukko (DEM)
    - säätökerroin: Määrittää kuinka jyrkkä S-käyrä on (suurempi arvo = jyrkempi kontrasti)
    - keskiarvo: Keskipiste, jonka ympärille sigmoid-käyrä asettuu. 
                 Jos None, käytetään taulukon todellista keskiarvoa.
    """
    if keskiarvo is None:
        keskiarvo = np.mean(dem_taulukko)
        
    # Standardoidaan data (keskitetään nollaan) ja kerrotaan säätökertoimella
    z = saatokerroin * (dem_taulukko - keskiarvo)
    
    # Varsinainen sigmoid-kaava: 1 / (1 + e^-z)
    return 1 / (1 + np.exp(-z))



import numpy as np


def ocean_currents(
    dem,
    wind_x,
    wind_y,
    wind_z,
    temperature,
    precipitation,
    planet_radius_km,
    rotation_period_hours,
    rotation_direction=1,
    sea_level=0.0,
    max_depth=6000.0,

    # -----------------------------
    # Säädettävät parametrit
    # -----------------------------
    wind_factor=0.025,
    coriolis_factor=0.35,
    gyre_factor=0.015,
    coast_factor=0.30,
    thermal_factor=0.008,
    rain_factor=0.002,
    upwelling_factor=0.025,
    deep_factor=0.008,
):
    """
    Yksinkertaistettu mutta käyttökelpoinen planeetan
    merivirtamalli.

    Kaikki inputit ovat 2D numpy-arrayta:
        [height, width]

    DEM:
        metriä merenpinnan ylä-/alapuolella

    wind_x/y/z:
        m/s

    temperature:
        °C

    precipitation:
        esim. mm/vuosi

    Palauttaa:
        current_x
        current_y
        current_z
        surface_x
        surface_y
        deep_x
        deep_y
        upwelling
        ocean
        depth
    """

    # =========================================================
    # 1. INPUT
    # =========================================================

    dem = np.asarray(dem, dtype=np.float64)

    wind_x = np.asarray(wind_x, dtype=np.float64)
    wind_y = np.asarray(wind_y, dtype=np.float64)
    wind_z = np.asarray(wind_z, dtype=np.float64)

    temperature = np.asarray(
        temperature,
        dtype=np.float64
    )

    precipitation = np.asarray(
        precipitation,
        dtype=np.float64
    )

    h, w = dem.shape

    # =========================================================
    # 2. MERI / MAA
    # =========================================================

    ocean = dem < sea_level

    depth = np.maximum(
        sea_level - dem,
        0.0
    )

    depth_norm = np.clip(
        depth / max_depth,
        0.0,
        1.0
    )

    # =========================================================
    # 3. LATITUDE
    # =========================================================

    # DEM oletetaan:
    #
    # y=0       = +90°
    # y=h-1     = -90°
    #
    # Jos sinun DEMissäsi etelä on y=0,
    # vaihda tämä toisinpäin.

    latitude = np.linspace(
        90.0,
        -90.0,
        h
    )

    lat = np.radians(latitude)[:, None]

    # =========================================================
    # 4. LONGITUDE
    # =========================================================

    longitude = np.linspace(
        -180.0,
        180.0,
        w,
        endpoint=False
    )

    lon = np.radians(longitude)[None, :]

    # =========================================================
    # 5. PLANEETAN PYÖRIMINEN
    # =========================================================

    radius_m = planet_radius_km * 1000.0

    rotation_period = (
        rotation_period_hours * 3600.0
    )

    omega = (
        2.0 * np.pi
        / rotation_period
    )

    # Coriolis-parametri
    f = (
        2.0
        * rotation_direction
        * omega
        * np.sin(lat)
    )

    # =========================================================
    # 6. TUULEN PERUSVIRTA
    # =========================================================

    surface_x = wind_x * wind_factor
    surface_y = wind_y * wind_factor
    surface_z = wind_z * wind_factor

    # =========================================================
    # 7. CORIOLIS
    # =========================================================

    coriolis_x = (
        -f * surface_y
    )

    coriolis_y = (
        f * surface_x
    )

    surface_x += (
        coriolis_x
        * coriolis_factor
    )

    surface_y += (
        coriolis_y
        * coriolis_factor
    )

    # =========================================================
    # 8. SUURI MITTAKAAVA / GYRE
    # =========================================================
    #
    # Luodaan subtrooppinen suuri kierto.
    #
    # sin(lat) * cos(lat)
    #
    # antaa vastakkaissuuntaista liikettä
    # pohjoisella ja eteläisellä pallonpuoliskolla.
    #

    gyre_strength = (
        np.sin(lat)
        * np.cos(lat)
    )

    # Heikko longitude-riippuvuus estää täysin
    # homogeenisen kentän.

    gyre_x = (
        np.cos(lat)
        * gyre_factor
    )

    gyre_y = (
        -np.sin(lon)
        * gyre_strength
        * gyre_factor
    )

    surface_x += gyre_x
    surface_y += gyre_y

    # =========================================================
    # 9. LÄMPÖTILA
    # =========================================================

    ocean_temperature = temperature[ocean]

    if ocean_temperature.size:
        mean_temperature = np.nanmean(
            ocean_temperature
        )
    else:
        mean_temperature = 0.0

    temperature_anomaly = (
        temperature
        - mean_temperature
    )

    # Lämmin vesi pyrkii kohti matalampaa latitudea
    # ja kylmä kohti korkeampaa latitudea.
    #
    # Tämä on tarkoituksella vain pieni korjaustermi.

    thermal_y = (
        -temperature_anomaly
        * np.cos(lat)
        * thermal_factor
    )

    surface_y += thermal_y

    # =========================================================
    # 10. SADEMÄÄRÄ
    # =========================================================

    ocean_rain = precipitation[ocean]

    if ocean_rain.size:
        mean_rain = np.nanmean(
            ocean_rain
        )
    else:
        mean_rain = 0.0

    rain_anomaly = (
        precipitation
        - mean_rain
    )

    # Runsas sade -> makeampi/kevyempi pintavesi.
    #
    # Vaikutus pidetään pienenä.

    surface_z += (
        -rain_anomaly
        * rain_factor
    )

    # =========================================================
    # 11. MERENPOHJAN VAIKUTUS
    # =========================================================

    # Syvässä vedessä vapaa virtaus.
    # Matalassa vedessä virtaus hidastuu.

    depth_factor = (
        0.25
        + 0.75 * depth_norm
    )

    surface_x *= depth_factor
    surface_y *= depth_factor

    # =========================================================
    # 12. RANNIKON SUUNTAINEN VIRTA
    # =========================================================

    # DEM-gradientti
    #
    # np.gradient palauttaa:
    # gy = y-suunnan gradientti
    # gx = x-suunnan gradientti

    gy, gx = np.gradient(dem)

    slope = np.sqrt(
        gx * gx
        + gy * gy
    )

    slope += 1e-12

    # Gradientin normaali
    normal_x = gx / slope
    normal_y = gy / slope

    # Tangentti rannikolle
    tangent_x = -normal_y
    tangent_y = normal_x

    # Rannikon vaikutusalue.
    #
    # 0 m syvyydessä = vahva
    # syvällä = heikko

    coastal_weight = np.exp(
        -depth / 300.0
    )

    coastal_weight *= ocean

    # Nykyisen virtauksen projektio
    # rannikon suuntaan.

    tangent_velocity = (
        surface_x * tangent_x
        + surface_y * tangent_y
    )

    surface_x += (
        tangent_x
        * tangent_velocity
        * coast_factor
        * coastal_weight
    )

    surface_y += (
        tangent_y
        * tangent_velocity
        * coast_factor
        * coastal_weight
    )

    # =========================================================
    # 13. KUMPUAMINEN
    # =========================================================

    # Tuulen komponentti rannikon normaalin suunnassa.

    offshore_wind = (
        wind_x * normal_x
        + wind_y * normal_y
    )

    # Tuuli pois rannikolta -> kumpuaminen.

    upwelling = (
        offshore_wind
        * coastal_weight
        * upwelling_factor
    )

    upwelling = np.maximum(
        upwelling,
        0.0
    )

    # Coriolis muuttaa hieman kumpuamisen suuntaa.

    surface_z += upwelling

    # =========================================================
    # 14. SYVÄVIRTAUS
    # =========================================================

    # Yksinkertainen tiheysindeksi.
    #
    # Kylmä + runsas sade / suolaisuuden puute
    # -> tässä yksinkertaistettu tiheyspoikkeama.

    temp_density = (
        -temperature_anomaly
    )

    rain_density = (
        -rain_anomaly
        * 0.1
    )

    density_anomaly = (
        temp_density
        + rain_density
    )

    # Tiheysero tuottaa pystysuuntaista liikettä.

    deep_z = (
        density_anomaly
        * deep_factor
    )

    # =========================================================
    # 15. SYVÄN VEDEN HORIZONTALINEN VIRTA
    # =========================================================

    # Syvävirta kulkee vastakkaiseen suuntaan
    # pintakierron osaan.

    deep_x = (
        -surface_x
        * depth_norm
        * 0.20
    )

    deep_y = (
        -surface_y
        * depth_norm
        * 0.20
    )

    # Lämpötila vaikuttaa syvän veden liikkeeseen.

    deep_x += (
        temperature_anomaly
        * 0.002
        * depth_norm
    )

    deep_y += (
        -temperature_anomaly
        * 0.002
        * depth_norm
    )

    # =========================================================
    # 16. YHDISTETTY VIRTA
    # =========================================================

    current_x = (
        surface_x
        + deep_x
    )

    current_y = (
        surface_y
        + deep_y
    )

    current_z = (
        surface_z
        + deep_z
    )

    # =========================================================
    # 17. MAA-ALUEET NOLLAKSI
    # =========================================================

    current_x = np.where(
        ocean,
        current_x,
        0.0
    )

    current_y = np.where(
        ocean,
        current_y,
        0.0
    )

    current_z = np.where(
        ocean,
        current_z,
        0.0
    )

    surface_x = np.where(
        ocean,
        surface_x,
        0.0
    )

    surface_y = np.where(
        ocean,
        surface_y,
        0.0
    )

    deep_x = np.where(
        ocean,
        deep_x,
        0.0
    )

    deep_y = np.where(
        ocean,
        deep_y,
        0.0
    )

    upwelling = np.where(
        ocean,
        upwelling,
        0.0
    )

    # =========================================================
    # 18. NUMEERINEN SIIVOUS
    # =========================================================

    outputs = [
        current_x,
        current_y,
        current_z,
        surface_x,
        surface_y,
        deep_x,
        deep_y,
        upwelling,
    ]

    outputs = [
        np.nan_to_num(
            x,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        for x in outputs
    ]

    (
        current_x,
        current_y,
        current_z,
        surface_x,
        surface_y,
        deep_x,
        deep_y,
        upwelling,
    ) = outputs

    return {
        "current_x": current_x,
        "current_y": current_y,
        "current_z": current_z,

        "surface_x": surface_x,
        "surface_y": surface_y,

        "deep_x": deep_x,
        "deep_y": deep_y,

        "upwelling": upwelling,

        "ocean": ocean,
        "depth": depth,
    }




def calculate_ocean_currents_00(
    dem,
    wind_x,
    wind_y,
    wind_z,
    temperature,
    precipitation,
    planet_radius_km,
    rotation_period_hours,
    rotation_direction=1,
    sea_level=0.0,
    wind_strength=0.03,
    temperature_strength=0.002,
    rain_strength=0.0005,
    coriolis_strength=1.0,
    bathymetry_strength=0.15,
    upwelling_strength=0.02,
):
    """
    Laskee yksinkertaistetun merivirtakentän.

    Kaikki gridit ovat muotoa (height, width).

    dem:
        Maaston korkeus metreinä.
        Merellä arvot < sea_level.

    wind_x, wind_y, wind_z:
        Tuulikenttä m/s.

    temperature:
        Lämpötila, esim. °C.

    precipitation:
        Sademäärä, esim. mm/vuosi.

    planet_radius_km:
        Planeetan säde kilometreinä.

    rotation_period_hours:
        Pyörähdysaika tunteina.

    rotation_direction:
        +1 = sama suunta kuin Maa
        -1 = vastakkainen

    Palauttaa:
        current_x
        current_y
        current_z
        upwelling
    """

    dem = np.asarray(dem, dtype=np.float64)
    wind_x = np.asarray(wind_x, dtype=np.float64)
    wind_y = np.asarray(wind_y, dtype=np.float64)
    wind_z = np.asarray(wind_z, dtype=np.float64)
    temperature = np.asarray(temperature, dtype=np.float64)
    precipitation = np.asarray(precipitation, dtype=np.float64)

    h, w = dem.shape

    # ---------------------------------------------------------
    # 1. MERI / MAA
    # ---------------------------------------------------------

    ocean = dem < sea_level

    # Syvyys metreinä
    depth = np.maximum(sea_level - dem, 0.0)

    # Normalisoitu syvyys
    depth_norm = np.clip(depth / 5000.0, 0.0, 1.0)

    # ---------------------------------------------------------
    # 2. LATITUUDI
    # ---------------------------------------------------------

    # Gridin oletetaan kattavan -90 ... +90
    latitude = np.linspace(-90.0, 90.0, h)
    latitude_rad = np.radians(latitude)

    # 2D latitude
    lat2d = latitude_rad[:, None]

    # ---------------------------------------------------------
    # 3. PLANEETAN PYÖRIMINEN
    # ---------------------------------------------------------

    rotation_period_sec = rotation_period_hours * 3600.0

    omega = 2.0 * np.pi / rotation_period_sec

    # Coriolis-parametri
    f = (
        2.0
        * rotation_direction
        * omega
        * np.sin(lat2d)
    )

    # ---------------------------------------------------------
    # 4. TUULEN AIHEUTTAMA PINTAVIRTA
    # ---------------------------------------------------------

    current_x = wind_x * wind_strength
    current_y = wind_y * wind_strength
    current_z = wind_z * wind_strength

    # ---------------------------------------------------------
    # 5. CORIOLIS
    # ---------------------------------------------------------

    # Pohjoisella pallonpuoliskolla +f
    # Eteläisellä -f.
    #
    # dx/dt = -f * vy
    # dy/dt =  f * vx

    cx = -f * current_y
    cy = f * current_x

    current_x += cx * coriolis_strength
    current_y += cy * coriolis_strength

    # ---------------------------------------------------------
    # 6. LÄMPÖTILAN VAIKUTUS
    # ---------------------------------------------------------

    # Yksinkertainen lämpötila-anomalia.
    #
    # Vähennetään koko planeetan keskilämpötila.
    temp_anomaly = temperature - np.nanmean(
        temperature[ocean]
    )

    # Lämmin vesi pyrkii hieman kohti päiväntasaajaa,
    # kylmä vesi kohti napoja.
    #
    # Tämä ei ole täydellinen fysikaalinen malli,
    # mutta auttaa muodostamaan suuren mittakaavan kiertoa.

    temp_force_y = (
        -temp_anomaly * temperature_strength
        * np.sin(lat2d)
    )

    current_y += temp_force_y

    # ---------------------------------------------------------
    # 7. SADEMÄÄRÄ / SUOLAISUUDEN APPROKSIMAATIO
    # ---------------------------------------------------------

    rain_anomaly = precipitation - np.nanmean(
        precipitation[ocean]
    )

    # Paljon sadetta -> makeampi vesi -> hieman kevyempi.
    #
    # Tehdään tästä pieni pystysuuntainen komponentti.

    current_z += (
        -rain_anomaly
        * rain_strength
    )

    # ---------------------------------------------------------
    # 8. SYVYYDEN VAIKUTUS
    # ---------------------------------------------------------

    # Syvässä meressä virtaus saa hieman suuremman
    # inertian / pienemmän kitkan.

    depth_factor = (
        1.0
        + depth_norm * bathymetry_strength
    )

    current_x *= depth_factor
    current_y *= depth_factor

    # ---------------------------------------------------------
    # 9. RANNIKON SUUNTAINEN VIRTAUS
    # ---------------------------------------------------------

    # Lasketaan DEM-gradientti.
    #
    # Gradientti osoittaa ylöspäin maastoa kohti.
    # Merellä siitä voidaan muodostaa rannikon tangentti.

    gy, gx = np.gradient(dem)

    slope = np.sqrt(gx * gx + gy * gy) + 1e-12

    # Rannikon tangenttivektori
    coast_tx = -gy / slope
    coast_ty = gx / slope

    # Rannikon vaikutus vain matalassa vedessä
    coastal_weight = np.exp(-depth / 500.0)

    # Projektio rannikon suuntaan
    coastal_flow = (
        current_x * coast_tx
        + current_y * coast_ty
    )

    current_x += (
        coast_tx
        * coastal_flow
        * coastal_weight
        * 0.15
    )

    current_y += (
        coast_ty
        * coastal_flow
        * coastal_weight
        * 0.15
    )

    # ---------------------------------------------------------
    # 10. KUMPUAMINEN
    # ---------------------------------------------------------

    # Rannikon tangentti
    # -> määritellään rantaan nähden ulospäin suuntautuva komponentti.

    coast_nx = gx / slope
    coast_ny = gy / slope

    # Tuulen komponentti rannikon normaalin suunnassa
    offshore_wind = (
        wind_x * coast_nx
        + wind_y * coast_ny
    )

    # Kumpuaminen:
    #
    # jos tuuli kuljettaa pintavettä rannasta poispäin,
    # syvempi vesi nousee.

    upwelling = (
        offshore_wind
        * coastal_weight
        * upwelling_strength
    )

    # Vain merellä
    upwelling *= ocean

    # Vertical velocity
    current_z += upwelling

    # ---------------------------------------------------------
    # 11. MAA-ALUEET NOLLAKSI
    # ---------------------------------------------------------

    current_x = np.where(ocean, current_x, 0.0)
    current_y = np.where(ocean, current_y, 0.0)
    current_z = np.where(ocean, current_z, 0.0)

    upwelling = np.where(ocean, upwelling, 0.0)

    # ---------------------------------------------------------
    # 12. NANIT POIS
    # ---------------------------------------------------------

    current_x = np.nan_to_num(current_x)
    current_y = np.nan_to_num(current_y)
    current_z = np.nan_to_num(current_z)
    upwelling = np.nan_to_num(upwelling)

    return (
        current_x,
        current_y,
        current_z,
        upwelling,
    )




def simulate_thermal_erosion(dem, iterations=10, c_repose=1.0, talus_rate=0.1):
    """
    Simuloi termistä eroosiota (rinteiden luhistumista) korkeusmallissa.
    
    Parametrit:
    - dem: 2D numpy-array (korkeudet metreinä)
    - iterations: Kuinka monta kertaa koko maasto käydään läpi
    - c_repose: Kriittinen korkeusero naapurisolujen välillä (lepokulman kynnys metreinä)
    - talus_rate: Kuinka suuri osa ylimääräisestä maasta valuu yhdellä kerralla (0.0 - 0.5)
    """
    height, width = dem.shape
    eroded_dem = dem.copy().astype(np.float32)
    
    # Naapurisolujen siirtymät (8-suuntainen naapurusto)
    # Sisältää suorat ja diagonaaliset naapurit
    neighbors = [
        (-1, -1, 1.414), (-1, 0, 1.0), (-1, 1, 1.414),
        (0, -1, 1.0),                  (0, 1, 1.0),
        (1, -1, 1.414),  (1, 0, 1.0),  (1, 1, 1.414)
    ]
    
    for _ in range(iterations):
        # Tehdään kopio iteraation alussa, jotta laskenta ei ketjuunnu epätasaisesti
        current_dem = eroded_dem.copy()
        
        # Käydään läpi sisäosat (jätetään reunat rauhaan indeksien vuoksi)
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                h_center = current_dem[y, x]
                
                # Merenpinnan alapuolelta ei kuluteta mitään
                if h_center <= 0:
                    continue
                
                max_slope_diff = 0
                target_x, target_y = x, y
                
                # Esitellään muuttujat kriittiselle korkeuserolle huomioiden etäisyys
                # Diagonaalisilla naapureilla on pidempi matka (1.414), joten kynnys on suurempi
                
                # Esitellään naapurit ja etsitään suurin jyrkkyys
                for dy, dx, dist in neighbors:
                    h_neighbor = current_dem[y + dy, x + dx]
                    
                    # Lasketaan korkeusero suhteessa etäisyyteen
                    height_diff = h_center - h_neighbor
                    
                    # Jos rinne on liian jyrkkä (ylittää lepokulman kynnyksen)
                    if height_diff > (c_repose * dist):
                        if height_diff > max_slope_diff:
                            max_slope_diff = height_diff
                            target_x, target_y = x + dx, y + dy
                
                # Jos liian jyrkkä rinne löytyi, siirretään maata
                if max_slope_diff > 0:
                    # Lasketaan siirrettävä määrä (rajataan, ettei kuluteta nollan alle)
                    moved_amount = max_slope_diff * talus_rate
                    
                    if h_center - moved_amount < 0:
                        moved_amount = h_center
                        
                    # Päivitetään korkeudet
                    eroded_dem[y, x] -= moved_amount
                    eroded_dem[target_y, target_x] += moved_amount

    return np.clip(eroded_dem, 0, None)

    



def simulate_erosion(dem, num_droplets=50000, learning_rate=0.1, inertia=0.05, capacity_factor=4.0, deposition_rate=0.1, erosion_rate=0.1, evaporation_rate=0.02):
    """
    Simuloi hydraulista eroosiota digitaalisessa korkeusmallissa (DEM).
    
    Parametrit:
    - dem: 2D numpy-array (korkeudet metreinä)
    - num_droplets: Simuloitavien sadepisaroiden määrä
    """
    height, width = dem.shape
    eroded_dem = dem.copy().astype(np.float32)
    
    for _ in range(num_droplets):
        # 1. Arvotaan pisaralle aloituspaikka (ei reunoille)
        x = np.random.uniform(1, width - 2)
        y = np.random.uniform(1, height - 2)
        
        dir_x, dir_y = 0.0, 0.0
        sediment = 0.0
        water = 1.0
        speed = 1.0
        
        # Pisaran elinkaari (max 30 askelta per pisara)
        for _ in range(30):
            ix, iy = int(x), int(y)
            
            # Tarkistetaan ettei olla merenpinnan tasolla tai sen alla
            if eroded_dem[iy, ix] <= 0:
                eroded_dem[iy, ix] = max(0.0, eroded_dem[iy, ix] + sediment)
                break
                
            # 2. Lasketaan gradientti (kallistussuunta) bilineaarisesti
            # Naapuripisteet
            h00 = eroded_dem[iy, ix]
            h10 = eroded_dem[iy, ix + 1]
            h01 = eroded_dem[iy + 1, ix]
            h11 = eroded_dem[iy + 1, ix + 1]
            
            # Painotukset
            u = x - ix
            v = y - iy
            
            grad_x = (h10 - h00) * (1 - v) + (h11 - h01) * v
            grad_y = (h01 - h00) * (1 - u) + (h11 - h10) * u
            
            # 3. Päivitetään pisaran suunta (inertia mukana)
            dir_x = dir_x * inertia - grad_x * (1 - inertia)
            dir_y = dir_y * inertia - grad_y * (1 - inertia)
            
            # Normalisoidaan suunta
            mag = np.sqrt(dir_x**2 + dir_y**2)
            if mag != 0:
                dir_x /= mag
                dir_y /= mag
                
            # Päivitetään paikka
            x += dir_x
            y += dir_y
            
            # Jos pisara menee ulos kartalta, lopetetaan
            if x < 1 or x >= width - 2 or y < 1 or y >= height - 2:
                break
                
            # Lasketaan korkeusero
            new_ix, new_iy = int(x), int(y)
            delta_h = eroded_dem[new_iy, new_ix] - h00
            
            # 4. Lasketaan sedimenttikapasiteetti
            # Mitä jyrkempi ja nopeampi, sitä enemmän maata mahtuu mukaan
            capacity = max(0.0, -delta_h) * speed * water * capacity_factor
            
            if sediment > capacity or delta_h > 0:
                # Kasataan sedimenttiä (pisara hidastuu tai nousee ylämäkeen)
                deposit = (sediment - capacity) * deposition_rate if delta_h < 0 else min(delta_h, sediment)
                sediment -= deposit
                eroded_dem[iy, ix] += deposit
            else:
                # Kulutetaan maastoa (otetaan sedimenttiä kyytiin)
                erode = min((capacity - sediment) * erosion_rate, -delta_h)
                # Estetään merenpinnan alitus kulutuksessa
                if eroded_dem[iy, ix] - erode < 0:
                    erode = eroded_dem[iy, ix]
                
                sediment += erode
                eroded_dem[iy, ix] -= erode
                
            # Päivitetään nopeus ja vesimäärä
            speed = np.sqrt(max(0.0, speed**2 + delta_h * 9.81))
            water *= (1 - evaporation_rate)
            
            if water < 0.01:
                break
                
    return np.clip(eroded_dem, 0, None) # Varmistetaan vielä lopuksi merenpinta


    

def muotoile_maan_jakauma(kohina_array, sealevel=0.5):
    """
    Muuntaa raa'an kohinan (välillä 0-1) vastaamaan Maan hypsometrista jakaumaa.
    
    Parametrit:
    - kohina_array: Perlin-kohinataulukko, jonka arvot on valmiiksi normalisoitu välille [0, 1]
    - sealevel: Haluttu merenpinnan taso (0.5). Alapuolella meri, yläpuolella manner.
    """
    # 1. Etsitään kynnysarvo raa'asta kohinasta, joka jakaa datan 71/29 suhteessa
    kohina_raja = np.percentile(kohina_array, 71)
    
    # Luodaan tyhjä taulukko tulokselle
    muotoiltu = np.zeros_like(kohina_array)
    
    # 2. MERIALUEET (71% datasta, raa'an kohina-rajan alapuolella)
    # Skaalataan merialueet välille [0, sealevel] eli [0.0, 0.5]
    meri_maski = kohina_array <= kohina_raja
    # Normalisoidaan merikohina välille 0-1
    meri_norm = (kohina_array[meri_maski] - np.min(kohina_array)) / (kohina_raja - np.min(kohina_array))
    
    # Jotta saadaan syvät valtameritasangot (Maan tyypillinen piirre), 
    # ajetaan meridata jyrkän sigmoidin/voimafunktion läpi, joka "pudottaa" pohjan alas
    muotoiltu[meri_maski] = (meri_norm ** 2) * sealevel

    # 3. MANNERALUEET (29% datasta, raa'an kohina-rajan yläpuolella)
    # Skaalataan manneralueet välille [sealevel, 1.0] eli [0.5, 1.0]
    manner_maski = kohina_array > kohina_raja
    manner_norm = (kohina_array[manner_maski] - kohina_raja) / (np.max(kohina_array) - kohina_raja)
    
    # S-käyrä (sigmoid) mantereille, jotta saadaan laajat tasangot rannikon lähelle 
    # ja jyrkät vuoristot vasta aivan korkeimmille kohdille
    manner_sigmoid = 1 / (1 + np.exp(-6 * (manner_norm - 0.3)))
    # Skaalataan takaisin välille [0.5, 1.0]
    manner_scaled = sealevel + (1.0 - sealevel) * manner_sigmoid
    
    muotoiltu[manner_maski] = manner_scaled
    
    return muotoiltu


def spherical_noise_native(width, height, scale, octaves=6, persistence=0.5, lacunarity=2.0, seed_offset=0.0, seed_value=12):
    """Aiemmin luotu funktio pienenä variaationa (seed_offset lisätty erottamaan X/Y kohinat)."""
    lat = np.linspace(-np.pi / 2, np.pi / 2, height)
    lon = np.linspace(-np.pi, np.pi, width)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    
    nx = scale * np.cos(lat_grid) * np.cos(lon_grid)
    ny = scale * np.cos(lat_grid) * np.sin(lon_grid)
    nz = scale * np.sin(lat_grid) + seed_offset # Siirretään 3D-avaruudessa eri kohtaan
    
    v_pnoise3 = np.vectorize(lambda x, y, z: pnoise3(x, y, z, octaves=octaves, 
                                                     persistence=persistence, 
                                                     lacunarity=lacunarity, base=seed_value))
    return normalize(v_pnoise3(nx, ny, nz))

def spherical_noise_offset(input_image, move_amount_max=10.0, scale=1.5, seed=12):
    """
    Siirtää input_image-kuvan pikseleitä pallon pinnalla 3D-kohinan mukaan.
    Säilyttää saumattomuuden reunoilla ja napojen geometrian.
    """
    height, width = input_image.shape[:2]
    seed_value=seed    
    # 1. Luodaan kaksi erillistä kohinakarttaa (yksi pituusasteelle, yksi leveysasteelle)
    # Käytetään siirrosta (seed_offset) eri arvoja, jotta liikkeet eivät ole identtiset
    noise_lon = spherical_noise_native(width, height, scale, seed_offset=0.0, seed_value=seed)
    noise_lat = spherical_noise_native(width, height, scale, seed_offset=100.0, seed_value=seed1)
    
    # 2. Luodaan alkuperäiset pikselikoordinaatit (X, Y)
    y_indices, x_indices = np.indices((height, width), dtype=np.float64)
    
    # 3. Lasketaan siirtymät (offsetit) pikseleinä
    # Kohina on välillä [-1, 1], joten kerrotaan se maksimisiirrolla
    offset_lon = noise_lon * move_amount_max
    offset_lat = noise_lat * move_amount_max
    
    # 4. Sovelletaan siirtymät koordinaatteihin
    new_x = x_indices + offset_lon
    new_y = y_indices + offset_lat
    
    # 5. Korjataan pallopinnan rajat (TÄRKEÄÄ SAUMATTOMUUDELLE):
    # - Pituusaste (X) rullaa ympäri saumattomasti (Wrap-around / Torus)
    new_x = np.mod(new_x, width)
    # - Leveysaste (Y) ei voi mennä yli napojen, joten peilataan tai rajataan se reunoihin
    new_y = np.clip(new_y, 0, height - 1)
    
    # 6. Interpoloidaan uudet pikseliarvot alkuperäisestä kuvasta
    # Jos kyseessä on värikuva (RGB), tehdään siirto jokaiselle kanavalle erikseen
    if len(input_image.shape) == 3:
        output_image = np.zeros_like(input_image)
        for c in range(input_image.shape[2]):
            output_image[..., c] = map_coordinates(input_image[..., c], [new_y, new_x], order=1, mode='wrap')
    else:
        output_image = map_coordinates(input_image, [new_y, new_x], order=1, mode='wrap')
        
    return normalize(output_image)


def generate_spherical_noise(width, height, scale=1.0, octaves=4, persistence=0.5, lacunarity=2.0, seed=1):
    # Luodaan tyhjä taulukko melulle
    noise_map = np.zeros((height, width))
    seed_value=seed1    
    # Luodaan leveys- ja pituusasteiden ruudukko (radiaaneina)
    # lon (pituusaste): -PI ... PI (kiertää pallon ympäri)
    # lat (leveysaste): -PI/2 ... PI/2 (pohjois- ja etelänapa)
    lon = np.linspace(-np.pi, np.pi, width)
    lat = np.linspace(-np.pi / 2, np.pi / 2, height)
    
    # Tehdään koordinaateista 2D-ruudukko
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    
    # Muunnetaan pallo-koordinaatit (lon, lat) 3D-kartesisiksi koordinaateiksi (x, y, z)
    # Kerroin 'scale' määrittää, kuinka "lähellä" tai "kaukana" melu on (vaikuttaa tiheyteen)
    x = scale * np.cos(lat_grid) * np.cos(lon_grid)
    y = scale * np.cos(lat_grid) * np.sin(lon_grid)
    z = scale * np.sin(lat_grid)
    
    # Lasketaan pnoise3-arvo jokaiselle 3D-pisteelle
    # Koska noise-kirjaston funktiot eivät tue suoraan numpy-taulukoita, iteroidaan alkiot
    for i in range(height):
        for j in range(width):
            noise_map[i, j] = pnoise3(
                x[i, j], 
                y[i, j], 
                z[i, j], 
                octaves=octaves, 
                persistence=persistence, 
                lacunarity=lacunarity, base=seed1
            )
            
    return noise_map

def create_dem_from_array(imagee, sealevel, dem_min, dem_max):
    #deltasea=dem_min/sealevel
    #deltaground=dem_max/(1-sealevel)
    dem=np.copy(imagee)
    #dem=np.where(dem<=0, dem*deltasea, dem*deltaground)
    deltadem=dem_max-dem_min
    print(deltadem)
    dem=(dem*deltadem)+dem_min
    landmask=np.copy(dem)
    landmask=np.where(dem<=0,0,1)
    return(dem, landmask)






def leviamis_rasteri(
    alku_lon,
    alku_lat,alku_vaesto,
    kaksink_aika,
    konduktanssi,
    kantokyky,
    aika_kysytty,
    planet_radius=6371.0,
    leviamisvauhti=1.0,
    n0=1.0,
    kapasiteettiosuus=0.99,
):
    """
    Kulttuurin least-cost / cost-distance -leviämismalli
    pallopinnalla.

    Palauttaa
    ----------
    dict:
        saapumisaika
        vaesto
        kantokyky
        aika_kantokykyyn
        aika_kysytyssa
        kasvunopeus_r
    """

    konduktanssi = np.asarray(konduktanssi, dtype=np.float64)
    kantokyky = np.asarray(kantokyky, dtype=np.float64)

    if konduktanssi.ndim != 2:
        raise ValueError("Rasterin pitää olla 2-ulotteinen.")

    if kantokyky.shape != konduktanssi.shape:
        raise ValueError(
            "konduktanssi ja kantokyky pitää olla saman kokoisia."
        )

    if np.any((konduktanssi < 0) | (konduktanssi > 1)):
        raise ValueError("Konduktanssin pitää olla välillä 0...1.")

    if kaksink_aika <= 0:
        raise ValueError("kaksink_aika pitää olla > 0.")

    height, width = konduktanssi.shape
    n = height * width

    # ------------------------------------------------------------
    # Pikselin indeksit
    # ------------------------------------------------------------

    def idx(row, col):
        return row * width + col

    # ------------------------------------------------------------
    # Lähtöpikseli
    #
    # Rasteri:
    # lon = -180 ... 180
    # lat =   90 ... -90
    # ------------------------------------------------------------

    col0 = int((alku_lon + 180.0) / 360.0 * width)
    row0 = int((90.0 - alku_lat) / 180.0 * height)

    col0 = np.clip(col0, 0, width - 1)
    row0 = np.clip(row0, 0, height - 1)

    source = idx(row0, col0)

    # ------------------------------------------------------------
    # Pikselikeskusten koordinaatit
    # ------------------------------------------------------------

    lon = -180.0 + (np.arange(width) + 0.5) * 360.0 / width
    lat = 90.0 - (np.arange(height) + 0.5) * 180.0 / height

    lon2d, lat2d = np.meshgrid(lon, lat)

    # ------------------------------------------------------------
    # Pallopinnan etäisyys
    # ------------------------------------------------------------

    def distance(lat1, lon1, lat2, lon2):

        lat1 = np.radians(lat1)
        lat2 = np.radians(lat2)
        lon1 = np.radians(lon1)
        lon2 = np.radians(lon2)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(lat1)
            * np.cos(lat2)
            * np.sin(dlon / 2.0) ** 2
        )

        return (
            2.0
            * planet_radius
            * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        )

    # ------------------------------------------------------------
    # Rakennetaan sparse-graafin kaaret
    #
    # 8-naapuria.
    # ------------------------------------------------------------

    rows = []
    cols = []
    costs = []

    naapuri = [
        (-1, -1),
        (-1,  0),
        (-1,  1),
        ( 0, -1),
        ( 0,  1),
        ( 1, -1),
        ( 1,  0),
        ( 1,  1),
    ]

    for r in range(height):

        for c in range(width):

            if konduktanssi[r, c] <= 0:
                continue

            source_idx = idx(r, c)

            for dr, dc in naapuri:

                rr = r + dr
                cc = (c + dc) % width

                # Pohjois- ja etelänavat eivät mene ympäri.
                if rr < 0 or rr >= height:
                    continue

                if konduktanssi[rr, cc] <= 0:
                    continue

                target_idx = idx(rr, cc)

                dist = distance(
                    lat2d[r, c],
                    lon2d[r, c],
                    lat2d[rr, cc],
                    lon2d[rr, cc],
                )

                # Reunan konduktanssi.
                cond = (
                    konduktanssi[r, c]
                    + konduktanssi[rr, cc]
                ) / 2.0

                # km / (km/v) = vuotta
                cost = dist / (
                    leviamisvauhti * cond
                )

                rows.append(source_idx)
                cols.append(target_idx)
                costs.append(cost)

    # ------------------------------------------------------------
    # Sparse adjacency matrix
    # ------------------------------------------------------------

    graph = csr_matrix(
        (costs, (rows, cols)),
        shape=(n, n),
    )

    # ------------------------------------------------------------
    # Dijkstra: yhdestä lähteestä kaikkiin
    # ------------------------------------------------------------

    distances = dijkstra(
        graph,
        directed=True,
        indices=source,
    )

    saapumisaika = distances.reshape(
        height,
        width,
    )

    # ------------------------------------------------------------
    # Väestö
    # ------------------------------------------------------------

    r_growth = np.log(2.0) / kaksink_aika

    saavutettu = np.isfinite(saapumisaika)

    aika_pikselissa = np.zeros_like(
        saapumisaika
    )

    aika_pikselissa[saavutettu] = np.maximum(
        0.0,
        aika_kysytty - saapumisaika[saavutettu],
    )

    K = np.maximum(
        kantokyky,
        0.0,
    )

    vaesto = np.zeros_like(K)

    # ------------------------------------------------------------
    # Logistinen kasvu
    # ------------------------------------------------------------

    mask = (
        saavutettu
        & (K > n0)
    )

    vaesto[mask] = (
        K[mask]
        /
        (
            1.0
            +
            (
                (K[mask] - n0) / n0
            )
            *
            np.exp(
                -r_growth
                * aika_pikselissa[mask]
            )
        )
    )

    # Jos K <= n0
    mask = (
        saavutettu
        & (K <= n0)
    )

    vaesto[mask] = K[mask]

    # ------------------------------------------------------------
    # Aika 99 %:iin kantokyvystä
    # ------------------------------------------------------------

    aika_kantokykyyn = np.full_like(
        K,
        np.inf,
    )

    mask = (
        saavutettu
        & (K > n0)
    )

    tavoite = kapasiteettiosuus * K[mask]

    t = (
        -np.log(
            (
                (K[mask] - tavoite)
                / tavoite
            )
            *
            (
                n0
                / (K[mask] - n0)
            )
        )
        / r_growth
    )

    aika_kantokykyyn[mask] = t

    return {
        "saapumisaika": saapumisaika,
        "vaesto": vaesto,
        "kantokyky": K,
        "aika_kantokykyyn": aika_kantokykyyn,
        "aika_pikselissa": aika_pikselissa,
        "kasvunopeus_r": r_growth,
        "alku_pixel": (row0, col0),
    }



def leviamis_rasteri_hidas(
    alku_lon,
    alku_lat,
    kaksink_aika,
    konduktanssi,
    kantokyky,
    aika_kysytty,
    planet_radius=6371.0,
    leviamisvauhti=1.0,
    n0=1.0,
    kasvufunktio="logistinen",
    kapasiteettiosuus=0.99,
):
    """
    Arvioi kulttuurin leviämistä pallopinnalla.

    Parametrit
    ----------
    alku_lon : float
        Lähtöpisteen pituusaste [-180, 180].
    alku_lat : float
        Lähtöpisteen leveysaste [-90, 90].

    kaksink_aika : float
        Väestön kaksinkertaistumisaika vuosina.
        Kasvunopeudeksi muunnetaan r = ln(2) / kaksink_aika.

    konduktanssi : np.ndarray, shape=(height, width)
        Leviämisen konduktanssi välillä 0...1.
        0 = alueelle ei voi levitä.
        1 = normaali leviämisnopeus.

    kantokyky : np.ndarray, shape=(height, width)
        Pikselikohtainen kantokyky K.

    aika_kysytty : float
        Aika vuosina leviämisen alkamisesta.

    planet_radius : float
        Planeetan säde kilometreinä.

    leviamisvauhti : float
        Leviämisnopeus, km/vuosi, kun konduktanssi = 1.
        Oletus 1 km/vuosi.

    n0 : float
        Väestö, jolla kulttuuri aloittaa uudessa pikselissä.
        Oletus 1.

    kasvufunktio : str
        "logistinen" tai "eksponentiaalinen".

    kapasiteettiosuus : float
        Kuinka lähellä K:ta lasketaan "kantokyvyn saavuttamisen"
        tapahtuvan. Esimerkiksi 0.99 = 99 % K:sta.

    Palauttaa
    ----------
    dict, jossa:
        saapumisaika
        vaesto
        kantokyky
        aika_kantokykyyn
        aika_kysytyssa
        kasvunopeus
    """

    konduktanssi = np.asarray(konduktanssi, dtype=float)
    kantokyky = np.asarray(kantokyky, dtype=float)

    if konduktanssi.ndim != 2:
        raise ValueError("konduktanssi pitää olla 2D-taulukko")

    if kantokyky.shape != konduktanssi.shape:
        raise ValueError("konduktanssi ja kantokyky pitää olla saman kokoisia")

    if np.any(konduktanssi < 0) or np.any(konduktanssi > 1):
        raise ValueError("konduktanssin pitää olla välillä 0...1")

    height, width = konduktanssi.shape

    # ------------------------------------------------------------
    # 1. Muunna alkuperäinen lon/lat pikseliksi
    # ------------------------------------------------------------

    col0 = int((alku_lon + 180.0) / 360.0 * width)
    row0 = int((90.0 - alku_lat) / 180.0 * height)

    col0 = np.clip(col0, 0, width - 1)
    row0 = np.clip(row0, 0, height - 1)

    # ------------------------------------------------------------
    # 2. Pikselikeskusten koordinaatit
    # ------------------------------------------------------------

    lons = -180.0 + (np.arange(width) + 0.5) * 360.0 / width
    lats = 90.0 - (np.arange(height) + 0.5) * 180.0 / height

    lat_rad = np.radians(lats)

    # ------------------------------------------------------------
    # 3. Pallopinnan etäisyys kahden pisteen välillä
    # ------------------------------------------------------------

    def haversine(lat1, lon1, lat2, lon2):
        lat1 = np.radians(lat1)
        lat2 = np.radians(lat2)
        lon1 = np.radians(lon1)
        lon2 = np.radians(lon2)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(lat1)
            * np.cos(lat2)
            * np.sin(dlon / 2.0) ** 2
        )

        return 2.0 * planet_radius * np.arcsin(
            np.sqrt(np.clip(a, 0.0, 1.0))
        )

    # ------------------------------------------------------------
    # 4. Etäisyydet viereisiin pikseleihin
    #
    # Käytetään 8-naapuria.
    # ------------------------------------------------------------

    dlat = 180.0 / height
    dlon = 360.0 / width

    # ------------------------------------------------------------
    # 5. Dijkstra: kulttuurin saapumisaika
    # ------------------------------------------------------------

    inf = np.inf

    saapumisaika = np.full((height, width), inf, dtype=float)

    saapumisaika[row0, col0] = 0.0

    heap = [(0.0, row0, col0)]

    # 8-naapurusto.
    naapurit = [
        (-1, -1),
        (-1,  0),
        (-1,  1),
        ( 0, -1),
        ( 0,  1),
        ( 1, -1),
        ( 1,  0),
        ( 1,  1),
    ]

    while heap:

        aika, r, c = heapq.heappop(heap)

        if aika != saapumisaika[r, c]:
            continue

        # Pallon pohjois-etelä -reunat eivät yhdisty.
        # Pituusaste kuitenkin kiertyy ympäri.
        for dr, dc in naapurit:

            rr = r + dr
            cc = (c + dc) % width

            if rr < 0 or rr >= height:
                continue

            kond1 = konduktanssi[r, c]
            kond2 = konduktanssi[rr, cc]

            if kond1 <= 0 or kond2 <= 0:
                continue

            # Käytetään reunalla keskimääräistä konduktanssia.
            kond = 0.5 * (kond1 + kond2)

            if kond <= 0:
                continue

            # Naapuripikselien keskikohtien välinen pallopintaetäisyys.
            dist = haversine(
                lats[r],
                lons[c],
                lats[rr],
                lons[cc],
            )

            # Nopeus km/v.
            nopeus = leviamisvauhti * kond

            # Aika tämän reunan ylittämiseen.
            dt = dist / nopeus

            uusi_aika = aika + dt

            if uusi_aika < saapumisaika[rr, cc]:
                saapumisaika[rr, cc] = uusi_aika
                heapq.heappush(
                    heap,
                    (uusi_aika, rr, cc),
                )

    # ------------------------------------------------------------
    # 6. Väestönkasvun parametri
    # ------------------------------------------------------------

    if kaksink_aika <= 0:
        raise ValueError("kaksink_aika pitää olla > 0")

    r = np.log(2.0) / kaksink_aika

    # ------------------------------------------------------------
    # 7. Kuinka kauan kulttuuri on ollut pikselissä?
    # ------------------------------------------------------------

    kulunut_aika = np.maximum(
        0.0,
        aika_kysytty - saapumisaika
    )

    saavutettu = np.isfinite(saapumisaika)

    # ------------------------------------------------------------
    # 8. Väestö pikseli kerrallaan
    # ------------------------------------------------------------

    vaesto = np.zeros_like(kantokyky, dtype=float)

    K = np.maximum(kantokyky, 0.0)

    if kasvufunktio == "eksponentiaalinen":

        vaesto[saavutettu] = (
            n0 * np.exp(r * kulunut_aika[saavutettu])
        )

        # Kantokyky toimii ylärajana.
        vaesto[saavutettu] = np.minimum(
            vaesto[saavutettu],
            K[saavutettu],
        )

    elif kasvufunktio == "logistinen":

        # Logistic:
        #
        # N(t) = K / (1 + ((K-N0)/N0) exp(-rt))
        #
        # Jokainen pikseli aloittaa oman kasvunsa
        # hetkellä saapumisaika.

        mask = saavutettu & (K > n0)

        vaesto[mask] = (
            K[mask]
            /
            (
                1.0
                + ((K[mask] - n0) / n0)
                * np.exp(-r * kulunut_aika[mask])
            )
        )

        # Jos K <= n0, lähtöpopulaatio ei voi olla suurempi
        # kuin kantokyky.
        mask_pieni_K = saavutettu & (K <= n0)

        vaesto[mask_pieni_K] = K[mask_pieni_K]

    else:
        raise ValueError(
            "kasvufunktio pitää olla 'logistinen' tai 'eksponentiaalinen'"
        )

    # ------------------------------------------------------------
    # 9. Aika n0 -> kapasiteettiosuus * K
    #
    # Logistisessa mallissa K:ta ei saavuteta täsmälleen
    # äärellisessä ajassa, joten käytetään esim. 99 % K.
    # ------------------------------------------------------------

    aika_kantokykyyn = np.full_like(
        K,
        np.inf,
        dtype=float,
    )

    if kasvufunktio == "logistinen":

        mask = saavutettu & (K > n0)

        tavoite = kapasiteettiosuus * K[mask]

        # Logisticin käänteinen ratkaisu:
        #
        # t = -1/r * ln(
        #       ((K-N) / N) * (N0 / (K-N0))
        #     )

        kelvollinen = (
            (tavoite > n0)
            & (tavoite < K[mask])
        )

        t = np.full_like(tavoite, np.inf)

        t[kelvollinen] = (
            -1.0 / r
            * np.log(
                (
                    (K[mask][kelvollinen] - tavoite[kelvollinen])
                    / tavoite[kelvollinen]
                )
                *
                (
                    n0
                    / (K[mask][kelvollinen] - n0)
                )
            )
        )

        aika_kantokykyyn[mask] = t

    else:
        # Eksponentiaalinen kasvu:
        #
        # N = N0 exp(rt)
        #
        # N = osuus*K
        #
        # t = ln(osuus*K/N0)/r

        mask = saavutettu & (K > n0)

        tavoite = kapasiteettiosuus * K[mask]

        kelvollinen = tavoite > n0

        t = np.full_like(tavoite, np.inf)

        t[kelvollinen] = (
            np.log(tavoite[kelvollinen] / n0) / r
        )

        aika_kantokykyyn[mask] = t

    return {
        "saapumisaika": saapumisaika,
        "vaesto": vaesto,
        "kantokyky": K,
        "aika_kantokykyyn": aika_kantokykyyn,
        "aika_kysytyssa": kulunut_aika,
        "kasvunopeus_r": r,
        "alku_pixel": (row0, col0),
    }


def etsi_paikka_ja_levia(
    target_alt,
    target_temp,
    target_rain,
    relief,
    temp,
    rain,
    ajo_vuodet=500,

    # =========================================================
    # LEVIÄMINEN
    # =========================================================

    sisamaanopeus_km_v=1.0,
    rannikkokerroin=10.0,

    saarinopeus_km_v=20.0,
    max_saar_hyppy_km=50.0,

    tasankokerroin=3.0,
    vuoristokerroin=0.30,

    min_suitability=0.05,

    # =========================================================
    # VÄESTÖ
    # =========================================================

    alku_vaesto=100.0,

    # vuosittainen luonnollinen kasvu
    vaesto_kasvu=0.015,

    # kuinka paljon yksi suitability-yksikkö
    # pystyy ylläpitämään väestöä
    kantokyky=10000.0,

    # kuinka paljon väestöä tarvitaan, jotta
    # uusi leviämisrintama syntyy tehokkaasti
    levitys_vaesto=100.0,

    # väestön vaikutus leviämisnopeuteen
    vaesto_leviamisbonus=2.0
):

    """
    Çayönüstä alkavan maatalous-/karjanhoitokulttuurin
    leviämismalli.

    Mallissa on:

    - ilmasto- ja maastosoveltuvuus
    - sisämaan leviäminen
    - nopeampi rannikkoleviäminen
    - lähisaarten hyppääminen
    - tasankobonus
    - vuoristohidaste
    - väestönkasvu
    - aluekohtainen kantokyky
    - väestön vaikutus leviämisnopeuteen

    Leviäminen ratkaistaan prioriteettijonolla,
    joten algoritmi ei käy turhaan koko rasteria
    uudelleen jokaisella kierroksella.
    """

    # ============================================================
    # 1. PERUSTIEDOT
    # ============================================================

    shape = relief.shape
    height, width = shape

    R_earth = 6371.0

    lat_step = 180.0 / height
    lon_step = 360.0 / width

    # ============================================================
    # 2. MAA / MERI
    # ============================================================

    land_mask = np.where(
        relief >= 1,
        1,
        0
    )

    # ============================================================
    # 3. ÇAYÖNÜ-FINGERPRINT
    # ============================================================

    score_alt = np.exp(
        -((relief - target_alt) ** 2)
        / (2 * 150 ** 2)
    ) * land_mask

    score_temp = np.exp(
        -((temp - target_temp) ** 2)
        / (2 * 1.5 ** 2)
    ) * land_mask

    score_rain = np.exp(
        -((rain - target_rain) ** 2)
        / (2 * 70 ** 2)
    ) * land_mask

    cayonu_index = (
        score_alt *
        score_temp *
        score_rain
    )

    # ============================================================
    # 4. MAANVILJELYN SOVELTUVUUS
    # ============================================================

    agriculture_temp = np.exp(
        -((temp - 15.0) ** 2)
        / (2 * 5.0 ** 2)
    )

    agriculture_rain = np.exp(
        -((rain - 700.0) ** 2)
        / (2 * 350.0 ** 2)
    )

    agriculture_alt = np.exp(
        -((relief - 500.0) ** 2)
        / (2 * 700.0 ** 2)
    )

    agriculture_index = (
        agriculture_temp *
        agriculture_rain *
        agriculture_alt
    )

    agriculture_index *= land_mask

    # ============================================================
    # 5. KARJANHOIDON SOVELTUVUUS
    # ============================================================

    pastoral_temp = np.exp(
        -((temp - 13.0) ** 2)
        / (2 * 7.0 ** 2)
    )

    pastoral_rain = np.exp(
        -((rain - 500.0) ** 2)
        / (2 * 400.0 ** 2)
    )

    pastoral_alt = np.exp(
        -((relief - 800.0) ** 2)
        / (2 * 1000.0 ** 2)
    )

    pastoral_index = (
        pastoral_temp *
        pastoral_rain *
        pastoral_alt
    )

    pastoral_index *= land_mask

    # ============================================================
    # 6. YHDISTETTY SOVELTUVUUS
    # ============================================================

    suitability = np.maximum(
        agriculture_index,
        pastoral_index
    )

    # ============================================================
    # 7. KARTAT
    # ============================================================

    kulttuuri = np.zeros(
        shape,
        dtype=np.uint8
    )

    leviamisaika = np.full(
        shape,
        np.inf,
        dtype=float
    )

    vaesto = np.zeros(
        shape,
        dtype=float
    )

    # ============================================================
    # 8. KANTOKYKY
    # ============================================================

    carrying_capacity = (
        kantokyky *
        suitability
    )

    # ============================================================
    # 9. RANNIKKO
    # ============================================================

    # Lasketaan KERRAN koko rasterille.
    # Ei kutsuta funktiota miljoonia kertoja.

    rannikko = np.zeros(
        shape,
        dtype=bool
    )

    for dy, dx in [
        (-1, -1),
        (-1,  0),
        (-1,  1),
        ( 0, -1),
        ( 0,  1),
        ( 1, -1),
        ( 1,  0),
        ( 1,  1)
    ]:

        shifted = np.roll(
            land_mask,
            shift=(dy, dx),
            axis=(0, 1)
        )

        rannikko |= (
            (land_mask == 1)
            &
            (shifted == 0)
        )

    # Napojen väärät wrapit pois
    rannikko[0, :] = False
    rannikko[-1, :] = False

    # ============================================================
    # 10. ETÄISYYDET
    # ============================================================

    def solumatka_km(y, x, ny, nx):

        lat1 = -90.0 + (y + 0.5) * lat_step
        lat2 = -90.0 + (ny + 0.5) * lat_step

        dlat = np.radians(
            lat2 - lat1
        )

        dlon_deg = (
            nx - x
        ) * lon_step

        if dlon_deg > 180:
            dlon_deg -= 360

        if dlon_deg < -180:
            dlon_deg += 360

        dlon = np.radians(
            dlon_deg
        )

        lat1r = np.radians(lat1)
        lat2r = np.radians(lat2)

        a = (
            np.sin(dlat / 2) ** 2
            +
            np.cos(lat1r)
            * np.cos(lat2r)
            * np.sin(dlon / 2) ** 2
        )

        c = 2 * np.arctan2(
            np.sqrt(a),
            np.sqrt(1 - a)
        )

        return R_earth * c

    # ============================================================
    # 11. LÄHTÖPISTE
    # ============================================================

    alku_y, alku_x = np.unravel_index(
        np.argmax(cayonu_index),
        shape
    )

    kulttuuri[
        alku_y,
        alku_x
    ] = 1

    leviamisaika[
        alku_y,
        alku_x
    ] = 0.0

    vaesto[
        alku_y,
        alku_x
    ] = alku_vaesto

    print(
        f"Kulttuuri-lähtöpiste: "
        f"({alku_y}, {alku_x})"
    )

    print(
        f"Kulttuuri-indeksi: "
        f"{cayonu_index[alku_y, alku_x]:.3f}"
    )

    # ============================================================
    # 12. PRIORITEETTIJONO
    # ============================================================

    # (aika, y, x)

    heap = [
        (
            0.0,
            alku_y,
            alku_x
        )
    ]

    # ============================================================
    # 13. NAAPURIT
    # ============================================================

    suunnat = [
        (-1, -1),
        (-1,  0),
        (-1,  1),
        ( 0, -1),
        ( 0,  1),
        ( 1, -1),
        ( 1,  0),
        ( 1,  1)
    ]

    # ============================================================
    # 14. LEVIÄMINEN
    # ============================================================

    while heap:

        aika_nyt, y, x = heapq.heappop(heap)

        # Vanha heap-merkintä
        if aika_nyt != leviamisaika[y, x]:
            continue

        if aika_nyt > ajo_vuodet:
            break

        # ========================================================
        # VÄESTÖN KASVU
        # ========================================================

        P = vaesto[y, x]

        K = carrying_capacity[y, x]

        if K > 0 and P > 0:

            # Kuinka monta vuotta solu on ollut asutettuna
            t = aika_nyt

            # Logistinen kasvu
            #
            # P(t) = K / (1 + ((K-P0)/P0)e^-rt)

            if K > alku_vaesto:

                P0 = alku_vaesto

                P = K / (
                    1.0
                    +
                    ((K - P0) / P0)
                    *
                    np.exp(
                        -vaesto_kasvu * t
                    )
                )

            else:

                P = K

            vaesto[y, x] = P

        # ========================================================
        # VÄESTÖN LEVIÄMISBONUS
        # ========================================================

        if levitys_vaesto > 0:

            vaesto_bonus = min(
                vaesto_leviamisbonus,
                1.0
                +
                vaesto[y, x]
                /
                levitys_vaesto
            )

        else:

            vaesto_bonus = 1.0

        # ========================================================
        # NAAPURIT
        # ========================================================

        for dy, dx in suunnat:

            ny = y + dy
            nx = x + dx

            # Longitude wrap
            if nx < 0:
                nx = width - 1

            if nx >= width:
                nx = 0

            if ny < 0 or ny >= height:
                continue

            # ----------------------------------------------------
            # Meriä ei täytetä kulttuurilla
            # ----------------------------------------------------

            if land_mask[ny, nx] == 0:
                continue

            # ----------------------------------------------------
            # JO SAAVUTETTU
            # ----------------------------------------------------

            if np.isfinite(
                leviamisaika[ny, nx]
            ):
                continue

            # ----------------------------------------------------
            # SOVELTUVUUS
            # ----------------------------------------------------

            p = suitability[
                ny,
                nx
            ]

            if p < min_suitability:
                continue

            # ----------------------------------------------------
            # ETÄISYYS
            # ----------------------------------------------------

            matka = solumatka_km(
                y,
                x,
                ny,
                nx
            )

            # ----------------------------------------------------
            # PERUSNOPEUS
            # ----------------------------------------------------

            nopeus = (
                sisamaanopeus_km_v
            )

            # ----------------------------------------------------
            # RANNIKKO
            # ----------------------------------------------------

            if (
                rannikko[y, x]
                and rannikko[ny, nx]
            ):

                nopeus *= rannikkokerroin

            elif rannikko[ny, nx]:

                nopeus *= np.sqrt(
                    rannikkokerroin
                )

            # ----------------------------------------------------
            # TASANKO
            # ----------------------------------------------------

            korkeus = relief[
                ny,
                nx
            ]

            if korkeus < 300:

                bonus = (
                    1.0
                    +
                    (tasankokerroin - 1.0)
                    *
                    (1.0 - korkeus / 300.0)
                )

                nopeus *= bonus

            # ----------------------------------------------------
            # VUORISTO
            # ----------------------------------------------------

            if korkeus > 1000:

                nopeus *= vuoristokerroin

            # ----------------------------------------------------
            # SOVELTUVUUS
            # ----------------------------------------------------

            suitability_kerroin = (
                0.25
                +
                0.75 * p
            )

            nopeus *= suitability_kerroin

            # ----------------------------------------------------
            # VÄESTÖ
            # ----------------------------------------------------

            nopeus *= vaesto_bonus

            # ----------------------------------------------------
            # AIKA
            # ----------------------------------------------------

            kulunut_aika = (
                matka /
                nopeus
            )

            uusi_aika = (
                aika_nyt
                +
                kulunut_aika
            )

            if uusi_aika > ajo_vuodet:
                continue

            # ----------------------------------------------------
            # SAAVUTUS
            # ----------------------------------------------------

            kulttuuri[
                ny,
                nx
            ] = 1

            leviamisaika[
                ny,
                nx
            ] = uusi_aika

            # Alkuväestö skaalataan hieman
            # alueen soveltuvuudella.
            vaesto[
                ny,
                nx
            ] = (
                alku_vaesto
                *
                (0.5 + 0.5 * p)
            )

            heapq.heappush(
                heap,
                (
                    uusi_aika,
                    ny,
                    nx
                )
            )

    # ============================================================
    # 15. TULOSTUS
    # ============================================================

    saavutetut = np.isfinite(
        leviamisaika
    )

    maara = np.sum(
        saavutetut
    )

    kokonaisvaesto = np.sum(
        vaesto
    )

    print(
        f"Saavutettuja soluja: "
        f"{maara} / {height * width}"
    )

    print(
        f"Arvioitu kokonaisväestö: "
        f"{kokonaisvaesto:,.0f}"
    )

    # ============================================================
    # 16. KULTTUURI
    # ============================================================

    plt.figure(
        figsize=(12, 6)
    )

    plt.imshow(
        kulttuuri,
        cmap="Greens"
    )

    plt.title(
        f"Kulttuuri {ajo_vuodet} vuoden jälkeen"
    )

    plt.show()

    # ============================================================
    # 17. LEVIÄMISAIKA
    # ============================================================

    aika_plot = (
        leviamisaika.copy()
    )

    aika_plot[
        np.isinf(aika_plot)
    ] = np.nan

    plt.figure(
        figsize=(12, 6)
    )

    plt.imshow(
        aika_plot,
        cmap="turbo"
    )

    plt.title(
        "Kulttuurin saavuttamisvuosi"
    )

    plt.colorbar(
        label="Vuosi"
    )

    plt.show()

    # ============================================================
    # 18. VÄESTÖKARTTA
    # ============================================================

    vaesto_plot = (
        vaesto.copy()
    )

    vaesto_plot[
        vaesto_plot <= 0
    ] = np.nan

    plt.figure(
        figsize=(12, 6)
    )

    plt.imshow(
        np.log10(vaesto_plot),
        cmap="magma"
    )

    plt.title(
        "Arvioitu väestö "
        "(log10)"
    )

    plt.colorbar(
        label="log10(väestö)"
    )

    plt.show()

    # ============================================================
    # 19. PALAUTUS
    # ============================================================

    return (
        cayonu_index,
        agriculture_index,
        pastoral_index,
        suitability,
        kulttuuri,
        leviamisaika,
        vaesto,
        carrying_capacity
    )












def calculate_land_sea_percentage(landmask):
    """
    Laskee maan ja meren todellisen pinta-alaosuuden ottaen huomioon leveysasteet.
    
    Parametrit:
    landmask : list tai np.ndarray
        2D-taulukko (esim. shape 360x720), jossa 0 = meri ja 1 = maa.
        Oletetaan, että rivit (korkeus) jakautuvat tasaisesti pohjoisnavalta etelänavalle.
    
    Palauttaa:
    dict: {'maa_prosentti': float, 'meri_prosentti': float}
    """
    # Muunnetaan numpy-taulukoksi, jos se on normaali Python-lista
    mask = np.array(landmask)
    height, width = mask.shape
    
    # Luodaan leveysasteet kullekin riville (+90 astetta pohjoista ... -90 astetta etelää)
    # Jos taulukkosi alkaa etelänavalta, vaihda 90 ja -90 paikkaa (tulos on silti sama)
    latitudes = np.linspace(90, -90, height)
    
    # Muunnetaan leveysasteet radiaaneiksi kosinin laskemista varten
    lat_rad = np.radians(latitudes)
    
    # Lasketaan painokerroin kullekin riville (leveyspiirin kosini)
    # Päiväntasaajalla cos(0) = 1 (suurin pinta-ala), navoilla cos(90) = 0 (pienin pinta-ala)
    row_weights = np.cos(lat_rad)
    
    # Monistetaan painokertoimet koko taulukon leveydelle, jotta saadaan 2D-painomatriisi
    weights_matrix = np.repeat(row_weights[:, np.newaxis], width, axis=1)
    
    # Lasketaan painotettu summa maalle (missä landmask == 1)
    # Koska meri on 0 ja maa on 1, mask * weights_matrix jättää jäljelle vain maan painot
    total_land_weight = np.sum(mask * weights_matrix)
    
    # Lasketaan koko maapallon kaikkien solujen kokonaispaino
    total_world_weight = np.sum(weights_matrix)
    
    # Lasketaan prosenttiosuudet
    land_percentage = (total_land_weight / total_world_weight) * 100
    sea_percentage = 100 - land_percentage
    
    return {
        'maa_prosentti': round(land_percentage, 2),
        'meri_prosentti': round(sea_percentage, 2)
    }


import numpy as np
from scipy.ndimage import distance_transform_edt


def merisuunnan_laskenta(seamask):
    """
    Laskee jokaiselle gridipisteelle suunnan kohti lähintä meripistettä.

    seamask:
        True  = meri
        False = maa

    Palauttaa:
        merisuunta_x
        merisuunta_y
    """

    _, indices = distance_transform_edt(
        ~seamask,
        return_distances=True,
        return_indices=True
    )

    sea_y = indices[0]
    sea_x = indices[1]

    y, x = np.indices(seamask.shape)

    dx = sea_x - x
    dy = sea_y - y

    norm = np.sqrt(dx**2 + dy**2)

    merisuunta_x = dx / (norm + 1e-12)
    merisuunta_y = dy / (norm + 1e-12)

    # Meressä ei tarvita suuntaa
    merisuunta_x[seamask] = 0
    merisuunta_y[seamask] = 0

    return merisuunta_x, merisuunta_y



def calculate_dem_statistics(dem_matrix):
    """
    Laskee leveysastepainotetut tilastot korkeusmallista (DEM).
    
    Parametrit:
    dem_matrix : list tai np.ndarray
        2D-taulukko, jossa arvot ovat korkeuksia/syvyyksiä metreinä.
        Merenpinta on 0. Maa > 0, Meri < 0.
        Oletetaan, että rivit jakautuvat tasaisesti pohjoisnavalta (+90) etelänavalle (-90).
    """
    # Muunnetaan numpy-taulukoksi tarvittaessa
    dem = np.array(dem_matrix, dtype=float)
    height, width = dem.shape
    
    # 1. Luodaan leveysastepainot (kosini-painotus)
    latitudes = np.linspace(90, -90, height)
    row_weights = np.cos(np.radians(latitudes))
    # Tehdään 2D-painomatriisi, joka vastaa DEM-taulukon kokoa
    weights = np.repeat(row_weights[:, np.newaxis], width, axis=1)
    
    # 2. Luodaan maskit maalle ja merelle
    # Huom: Tasainen 0 lasketaan tässä merenpinnaksi (meri)
    land_mask = dem > 0
    sea_mask = dem <= 0
    
    # 3. Maksimikorkeus ja minimisyvyys (absoluuttiset ääriarvot eivät vaadi painotusta)
    # Jos maata tai merta ei ole laisinkaan, asetetaan arvoksi 0
    max_land_height = np.max(dem[land_mask]) if np.any(land_mask) else 0.0
    min_sea_depth = np.min(dem[sea_mask]) if np.any(sea_mask) else 0.0
    
    # 4. Lasketaan leveysastepainotetut keskikorkeudet ja -syvyydet
    # Keskikorkeus maalle (vain ne pisteet, joissa land_mask on True)
    if np.any(land_mask):
        mean_land_height = np.sum(dem[land_mask] * weights[land_mask]) / np.sum(weights[land_mask])
    else:
        mean_land_height = 0.0
        
    # Keskisyvyys merelle (vain ne pisteet, joissa sea_mask on True)
    if np.any(sea_mask):
        mean_sea_depth = np.sum(dem[sea_mask] * weights[sea_mask]) / np.sum(weights[sea_mask])
    else:
        mean_sea_depth = 0.0

    return {
        'min_sea_depth_m': round(min_sea_depth, 1),
        'max_land_height_m': round(max_land_height, 1),
        'mean_sea_depth_m': round(mean_sea_depth, 1),
        'mean_land_height_m': round(mean_land_height, 1)
    }




def calculate_twi(dem, cell_size):
    """
    Laskee Topographic Wetness Indexin (TWI) DEMistä.

    Parameters
    ----------
    dem : np.ndarray
        Korkeusmatriisi muodossa (height, width), metreinä.
    cell_size : float
        Rasterisolun koko metreinä.

    Returns
    -------
    np.ndarray
        TWI-matriisi, sama koko kuin dem.

    Kaava:
        TWI = ln(a / tan(beta))

    missä:
        a    = upslope contributing area / solun leveys
        beta = rinnekulma radiaaneina
    """

    dem = np.asarray(dem, dtype=float)

    # Gradientti
    dz_dy, dz_dx = np.gradient(dem, cell_size, cell_size)

    # Rinnekulma
    slope = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))

    # Vältetään tan(0) = 0
    tan_slope = np.tan(slope)
    tan_slope = np.maximum(tan_slope, 1e-6)

    # Yksinkertainen contributing area:
    # alustavasti jokainen solu vastaa omaa pinta-alaansa
    contributing_area = np.full(
        dem.shape,
        cell_size,
        dtype=float
    )

    # TWI
    twi = np.log(contributing_area / tan_slope)

    return twi



def calculate_tpi(dem, window_size=5):
    dem = np.asarray(dem, dtype=float)

    mean_elevation = uniform_filter(
        dem,
        size=window_size,
        mode="nearest"
    )

    # Poistetaan keskimmäisen solun vaikutus keskiarvoon
    n = window_size * window_size
    mean_neighbors = (
        mean_elevation * n - dem
    ) / (n - 1)

    return dem - mean_neighbors



def calculate_lakes(
    relief,
    sateet,
    pet,
    flow_to,
    accumulation,
    cell_size,
    min_inflow=500_000,
    min_depth=2.0,
    lake_strength=1.0
):
    """
    Laskee järvet nopeasti D8-valuntaverkosta.

    Parameters
    ----------
    relief : np.ndarray
        DEM. Meri = 0, maa > 0.

    sateet : np.ndarray
        Vuotuinen sademäärä mm/vuosi.

    pet : np.ndarray
        Potentiaalinen haihdunta mm/vuosi.

    flow_to : np.ndarray
        D8-virtaussuunta calculate_rivers()-funktiosta.

    accumulation : np.ndarray
        D8-valunnan kertymä calculate_rivers()-funktiosta.

    cell_size : float
        Rasterisolun koko metreinä.

    min_inflow : float
        Pienin valuma-alueen vuosittainen vesimäärä,
        jolla järvi voi syntyä.

    min_depth : float
        Järven minimisyvyys metreinä.

    lake_strength : float
        Säätää vesitaseen vaikutusta järven kokoon.

    Returns
    -------
    lakes : np.ndarray
        Boolean-matriisi. True = järvi.

    lake_depth : np.ndarray
        Arvioitu järven syvyys metreinä.

    lake_id : np.ndarray
        Jokaiselle järvisolulle järven tunniste.
        0 = ei järveä.
    """

    relief = np.asarray(relief, dtype=float)
    sateet = np.asarray(sateet, dtype=float)
    pet = np.asarray(pet, dtype=float)
    accumulation = np.asarray(accumulation, dtype=float)

    height, width = relief.shape
    size = height * width

    # ---------------------------------------------------------
    # 1. MAA / MERI
    # ---------------------------------------------------------

    landmask = relief > 0

    # ---------------------------------------------------------
    # 2. NETTOVESI
    # ---------------------------------------------------------
    #
    # Sade - PET.
    #
    # Tämä ei ole vielä täydellinen vesitase, mutta antaa
    # järville tärkeän kosteuden / kuivuuden vaikutuksen.

    netto_vesi = np.maximum(
        sateet - pet,
        0.0
    )

    # ---------------------------------------------------------
    # 3. JOKAISEN SOLUN D8-PÄÄTE
    # ---------------------------------------------------------
    #
    # Jokainen solu seuraa flow_to-ketjuaan alaspäin.
    #
    # Käytetään korkeussuuntaista järjestystä:
    # matalampi solu käsitellään ensin.
    #
    # Tällöin jokainen solu voi periä oman päätepisteensä
    # suoraan seuraavalta solulta.

    order = np.argsort(
        relief.ravel()
    )

    sink_id = np.full(
        size,
        -1,
        dtype=np.int32
    )

    next_id = 0

    for index in order:

        y = index // width
        x = index % width

        if not landmask[y, x]:
            continue

        ny, nx = flow_to[y, x]

        # Ei virtaussuuntaa -> sinkki
        if ny < 0:
            sink_id[index] = next_id
            next_id += 1

        else:
            downstream = ny * width + nx

            # Koska downstream on matalammalla, sen ID
            # on jo määritetty.
            sink_id[index] = sink_id[downstream]

    sink_id_2d = sink_id.reshape(
        height,
        width
    )

    sink_count = next_id

    if sink_count == 0:
        return (
            np.zeros_like(relief, dtype=bool),
            np.zeros_like(relief, dtype=float),
            np.zeros_like(relief, dtype=np.int32)
        )

    # ---------------------------------------------------------
    # 4. VALUMA-ALUEIDEN VESIMÄÄRÄ
    # ---------------------------------------------------------
    #
    # accumulation on tässä mm/vuosi.
    #
    # Muutetaan solun vesimäärä m³/vuosi:
    #
    # mm / 1000 * m²

    cell_area = cell_size ** 2

    vesimaara = (
        netto_vesi / 1000.0
    ) * cell_area

    # Summaa jokaisen sinkin valuma-alueen vedet.
    #
    # np.bincount on tähän huomattavasti nopeampi kuin
    # rasterin toistuva läpikäynti.

    valid = sink_id >= 0

    basin_water = np.bincount(
        sink_id[valid],
        weights=vesimaara.ravel()[valid],
        minlength=sink_count
    )

    # ---------------------------------------------------------
    # 5. SINKIEN SIJAINNIT
    # ---------------------------------------------------------

    sink_positions = np.full(
        (sink_count, 2),
        -1,
        dtype=np.int32
    )

    for index in np.flatnonzero(valid):

        sid = sink_id[index]

        if sink_positions[sid, 0] < 0:
            sink_positions[sid] = (
                index // width,
                index % width
            )

    # ---------------------------------------------------------
    # 6. SPILL-KORKEUDET
    # ---------------------------------------------------------
    #
    # Etsitään jokaiselle valuma-altaalle alin raja,
    # josta vesi voisi poistua.
    #
    # Tämä tehdään ilman että jokaista allasta käydään
    # erikseen läpi.

    spill_height = np.full(
        sink_count,
        np.inf,
        dtype=float
    )

    # Tarkastellaan kaikki 8 naapuria.
    neighbors = [
        (-1, -1),
        (-1,  0),
        (-1,  1),
        ( 0, -1),
        ( 0,  1),
        ( 1, -1),
        ( 1,  0),
        ( 1,  1)
    ]

    for dy, dx in neighbors:

        y0 = max(0, -dy)
        y1 = min(height, height - dy)

        x0 = max(0, -dx)
        x1 = min(width, width - dx)

        src = sink_id_2d[y0:y1, x0:x1]
        nbr = relief[
            y0 + dy:y1 + dy,
            x0 + dx:x1 + dx
        ]

        nbr_sink = sink_id_2d[
            y0 + dy:y1 + dy,
            x0 + dx:x1 + dx
        ]

        # Vain rajat, joissa naapuri kuuluu eri
        # valuma-altaaseen.
        boundary = (
            (src >= 0) &
            (nbr_sink != src)
        )

        ids = src[boundary]

        if ids.size:
            heights = nbr[boundary]

            # Pienin poistumiskorkeus.
            np.minimum.at(
                spill_height,
                ids,
                heights
            )

    # ---------------------------------------------------------
    # 7. JÄRVEN EDELLYTYS
    # ---------------------------------------------------------

    lake_candidates = (
        (basin_water >= min_inflow) &
        np.isfinite(spill_height)
    )

    # ---------------------------------------------------------
    # 8. JÄRVEN VEDENPINTA
    # ---------------------------------------------------------
    #
    # Arvioidaan altaan saama vesimäärä suhteessa
    # sen pinta-alaan.
    #
    # Tätä ei tulkita täydellisenä vuosittaisena
    # hydrologisena tasapainona, vaan järven koon
    # muodostavana pelimallina.

    basin_area_cells = np.bincount(
        sink_id[valid],
        minlength=sink_count
    )

    basin_area_m2 = (
        basin_area_cells *
        cell_area
    )

    # Keskimääräinen vesikerros vuodessa.
    water_depth = np.zeros(
        sink_count,
        dtype=float
    )

    good_area = basin_area_m2 > 0

    water_depth[good_area] = (
        basin_water[good_area] /
        basin_area_m2[good_area]
    )

    # Vähintään min_depth, mutta ei yli spill-korkeuden.
    #
    # Koska water_depth on vuosittainen vesimäärä,
    # lake_strength toimii tässä säätökertoimena.

    water_depth *= lake_strength

    # ---------------------------------------------------------
    # 9. MUODOSTETAAN JÄRVI
    # ---------------------------------------------------------

    lake_id = np.zeros(
        (height, width),
        dtype=np.int32
    )

    lakes = np.zeros(
        (height, width),
        dtype=bool
    )

    lake_depth = np.zeros(
        (height, width),
        dtype=float
    )

    # Käydään läpi vain järvikandidaatit.
    # Tämä on yleensä huomattavasti pienempi joukko
    # kuin koko rasteri.

    lake_sinks = np.flatnonzero(
        lake_candidates
    )

    for sid in lake_sinks:

        sy, sx = sink_positions[sid]

        if sy < 0:
            continue

        depth = water_depth[sid]

        if depth < min_depth:
            continue

        # Järven vedenpinta.
        water_level = min(
            spill_height[sid],
            relief[sy, sx] + depth
        )

        # Valuma-alueen solut, jotka jäävät veden alle.
        mask = (
            (sink_id_2d == sid) &
            (relief <= water_level)
        )

        lakes[mask] = True
        lake_id[mask] = sid + 1

        lake_depth[mask] = (
            water_level -
            relief[mask]
        )

    return lakes, lake_depth, lake_id



def calculate_rivers(relief, sateet, pet, cell_size, threshold):
    """
    Laskee valunnan ja jokiverkoston D8-menetelmällä.

    Parameters
    ----------
    relief : np.ndarray
        DEM / relief. Meri = 0, maa > 0.
    sateet : np.ndarray
        Vuotuinen sademäärä, mm/vuosi.
    pet : np.ndarray
        Potentiaalinen haihdunta, mm/vuosi.
    cell_size : float
        Rasterisolun koko metreinä.
    threshold : float
        Minimivalunta, jolla solu luokitellaan joeksi.

    Returns
    -------
    rivers : np.ndarray
        Boolean-matriisi, True = joki.

    accumulation : np.ndarray
        Soluun kertyvä vuotuinen vesimäärä.

    flow_to : np.ndarray
        D8-virtaussuunta.
    """

    relief = np.asarray(relief, dtype=float)
    sateet = np.asarray(sateet, dtype=float)
    pet = np.asarray(pet, dtype=float)

    height, width = relief.shape

    # ---------------------------------------------------------
    # 1. MAA / MERI
    # ---------------------------------------------------------

    landmask = relief > 0

    # ---------------------------------------------------------
    # 2. NETTOVESI
    # ---------------------------------------------------------
    #
    # Sade - potentiaalinen haihdunta.
    #
    # Negatiivinen tulos tarkoittaa, ettei pintavaluntaa
    # synny tästä solusta.

    netto_vesi = np.maximum(sateet - pet, 0.0)

    # ---------------------------------------------------------
    # 3. D8-NAAPURIT
    # ---------------------------------------------------------

    neighbors = [
        (-1, -1),
        (-1,  0),
        (-1,  1),
        ( 0, -1),
        ( 0,  1),
        ( 1, -1),
        ( 1,  0),
        ( 1,  1)
    ]

    # Virtaussuunta.
    # -1 = ei virtaussuuntaa.

    flow_to = np.full(
        (height, width, 2),
        -1,
        dtype=np.int32
    )

    # ---------------------------------------------------------
    # 4. D8-VIRTAUS
    # ---------------------------------------------------------

    for y in range(height):
        for x in range(width):

            # Meri ei tarvitse D8-valuntaa.
            if not landmask[y, x]:
                continue

            best_slope = 0.0
            best_neighbor = None

            for dy, dx in neighbors:

                ny = y + dy
                nx = x + dx

                if not (0 <= ny < height and 0 <= nx < width):
                    continue

                # Merta kohti saa virrata.
                # Meri toimii lopullisena poistona.

                distance = cell_size

                if dy != 0 and dx != 0:
                    distance = cell_size * np.sqrt(2)

                slope = (
                    relief[y, x] -
                    relief[ny, nx]
                ) / distance

                if slope > best_slope:
                    best_slope = slope
                    best_neighbor = (ny, nx)

            if best_neighbor is not None:
                flow_to[y, x] = best_neighbor

    # ---------------------------------------------------------
    # 5. VALUNNAN KERTYMINEN
    # ---------------------------------------------------------
    #
    # Jokainen solu tuo verkostoon oman nettovesimääränsä.
    #
    # Yksikkö on tässä suhteessa mm/vuosi per rasterisolu.
    # Varsinainen pinta-ala voidaan ottaa huomioon myöhemmin
    # muuntamalla mm -> m³.

    accumulation = netto_vesi.copy()

    # Korkeimmasta matalimpaan.
    order = np.argsort(relief.ravel())[::-1]

    for index in order:

        y, x = np.unravel_index(
            index,
            relief.shape
        )

        ny, nx = flow_to[y, x]

        if ny >= 0:
            accumulation[ny, nx] += accumulation[y, x]

    # ---------------------------------------------------------
    # 6. JOKIMASKI
    # ---------------------------------------------------------

    rivers = (
        landmask &
        (accumulation >= threshold)
    )

    return rivers, accumulation, flow_to



def distance_to_sea(relief, PLANET_R):
    """
    Laskee etäisyyden lähimpään mereen kilometreinä.

    Parameters
    ----------
    relief : np.ndarray
        DEM-matriisi muodossa (height, width).
        relief <= 0 = meri
        relief > 0  = maa

    PLANET_R : float
        Planeetan säde kilometreinä.

    Returns
    -------
    np.ndarray
        Matriisi (height, width), jossa jokaisen maapikselin
        etäisyys lähimpään mereen kilometreinä.
        Meripikseleillä arvo on 0.
    """
    height, width = relief.shape

    land = relief > 0

    # Koko planeetan oletus: 180° x 360°
    dlat = np.pi / height
    dlon = 2 * np.pi / width

    # Pikselin keskipisteiden leveysasteet
    lat = (
        -np.pi / 2
        + (np.arange(height) + 0.5) * dlat
    )

    # Etäisyys lähimpään meripikseliin rasterikoordinaateissa
    pixel_distance = distance_transform_edt(land)

    # Pikselin fyysinen koko km
    pixel_height_km = PLANET_R * dlat
    pixel_width_km = PLANET_R * dlon * np.cos(lat)

    # Käytetään paikallista pikselikokoa
    pixel_size_km = np.minimum(
        pixel_height_km,
        pixel_width_km
    )[:, None]

    distance_to_sea_km = pixel_distance * pixel_size_km

    # Meri = 0 km
    distance_to_sea_km[~land] = 0

    return distance_to_sea_km


def distance_to_someheight(relief, PLANET_R, hei):
    """
    Laskee etäisyyden lähimpään kohtaan, jonka korkeus
    on vähintään hei.

    relief <= 0 = meri
    relief > 0  = maa

    hei : korkeus, esim. 2000 metriä

    Palauttaa etäisyyden kilometreinä.
    """

    height, width = relief.shape

    # Kohteet, joihin etäisyys halutaan laskea
    target = relief >= hei

    dlat = np.pi / height
    dlon = 2 * np.pi / width

    lat = (
        -np.pi / 2
        + (np.arange(height) + 0.5) * dlat
    )

    # Etäisyys lähimpään target-pikseliin
    pixel_distance = distance_transform_edt(~target)

    # Pikselin fyysinen koko km
    pixel_height_km = PLANET_R * dlat
    pixel_width_km = PLANET_R * dlon * np.cos(lat)

    pixel_size_km = np.minimum(
        pixel_height_km,
        pixel_width_km
    )[:, None]

    distance_to_km = pixel_distance * pixel_size_km

    return distance_to_km






def laske_ray_trace_shadows(dem, light_dir, sample_spacing=1.0):
    """
    Laskee varjot DEM-korkeusmallille säteenseurannalla (Ray Tracing).
    
    Parametrit:
    - dem: 2D NumPy-taulukko (korkeusarvot)
    - light_dir: Valon suuntavektori [x, y, z] (z:n tulee olla positiivinen)
    - sample_spacing: Pikselien välinen etäisyys metreinä (resoluutio)
    """
    ny, nx = dem.shape
    shadow_map = np.ones_like(dem, dtype=float) # 1.0 = valossa, 0.0 = varjossa
    
    # Normalisoidaan valon suuntavektori
    light_dir = np.array(light_dir, dtype=float)
    light_dir /= np.linalg.norm(light_dir)
    
    dx, dy, dz = light_dir
    
    # Jos valo tulee suoraan ylhäältä, varjoja ei synny tasaisella maastolla
    if dz >= 0.99:
        return shadow_map

    # Askelpituus x-y tasossa (yksi pikseli kerrallaan voimakkaimman akselin mukaan)
    max_step = max(abs(dx), abs(dy))
    step_x = dx / max_step
    step_y = dy / max_step
    step_z = dz / max_step * sample_spacing
    
    # Luodaan koordinaattiverkko
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny))
    
    # Iteroidaan säteitä valoa kohti max_steps-verran (riippuu maaston koosta)
    # Mitä pidempi etäisyys, sitä pidemmät varjot (esim. ilta-aurinko)
    max_steps = int(max(nx, ny) * 0.2) 
    
    # Tehdään kopio säteiden lähtöpisteistä
    curr_x = X.astype(float)
    curr_y = Y.astype(float)
    curr_z = dem.copy()
    
    for _ in range(1, max_steps):
        # Siirretään sädettä askel valonlähdettä kohti
        curr_x += step_x
        curr_y += step_y
        curr_z += step_z
        
        # Pyöristetään lähimpään pikseli-indeksiin
        ix = np.round(curr_x).astype(int)
        iy = np.round(curr_y).astype(int)
        
        # Maski pisteistä, jotka ovat vielä kartan sisällä
        valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        
        if not np.any(valid):
            break # Kaikki säteet karkasivat kartalta
            
        # Jos säteen korkeus (curr_z) on matalampi kuin maaston korkeus kyseisessä pisteessä,
        # kyseinen alkupiste (X, Y) on varjossa.
        maaston_korkeus = dem[iy[valid], ix[valid]]
        varjossa = curr_z[valid] < maaston_korkeus
        
        # Päivitetään alkuperäiset koordinaatit varjoon
        shadow_map[valid] = np.where(varjossa, 0.3, shadow_map[valid])
        
    return shadow_map

def laske_hillshade(dem, azimuth=315, angle_altitude=45):
    """
    Laskee maaston valovarjostuksen (Hillshade/Emboss) nopeasti NumPy-gradientilla.
    
    Parametrit:
    - dem: 2D NumPy-taulukko (korkeusmalli)
    - azimuth: Valon tulosuunta asteina (0=Pohjoinen, 90=Itä, 180=Etelä, 270=Länsi).
               Standardi kartografinen suunta on 315 (luode).
    - angle_altitude: Valon korkeuskulma horisontista asteina (0-90).
    """
    # Muutetaan kulmat radiaaneiksi
    azimuth_rad = np.radians(azimuth)
    altitude_rad = np.radians(angle_altitude)
    
    # Lasketaan maaston gradientti (muutosnopeus x- ja y-suunnassa)
    # Jos pikselikoko ei ole 1:1 korkeuden kanssa, jaa tulos pikselivälillä (esim. / dx)
    grad_y, grad_x = np.gradient(dem)
    
    # Lasketaan rinteiden kaltevuus (slope) ja suunta (aspect)
    slope = np.pi/2.0 - np.arctan(np.sqrt(grad_x**2 + grad_y**2))
    aspect = np.arctan2(-grad_x, grad_y)
    
    # Lasketaan valon ja pinnan kohtaamiskulma (Lambertin sääntö)
    shaded = np.sin(altitude_rad) * np.sin(slope) + \
             np.cos(altitude_rad) * np.cos(slope) * np.cos(azimuth_rad - aspect)
             
    # Normalisoidaan tulos välille 0-1
    shaded = (shaded + 1.0) / 2.0
    return shaded



def laske_vuosittainen_lampotila(dema, polar_temp, temp_diff, sealevel=0.5, max_korkeus_m=6000):
    """
    Laskee vuoden keskilämpötilan DEM-taulukon (relief) pohjalta.
    
    Parametrit:
    - dem: 2D NumPy-taulukko, arvoalue 0.0 - 1.0 (0.5 on merenpinta)
    - sealevel: Merenpinnan kynnysarvo taulukossa (oletus 0.5)
    - max_korkeus_m: Mikä on taulukon arvoa 1.0 vastaava todellinen korkeus metreinä
    """
    height, width = dema.shape
    
    # 1. LASKETAAN MERENPINNAN LÄMPÖTILA (Leveyspiirin mukaan)
    # Luodaan Y-akselille arvot -1 (Etelänapa) ... 0 (Päiväntasaaja) ... 1 (Pohjoisnapa)
    y_lin = np.linspace(-1, 1, height)
    # Monistetaan tämä koko taulukon leveydelle (vektoroitu Y-koordinaatisto)
    _, Y_leveyspiirit = np.meshgrid(np.arange(width), y_lin)
    
    # Merenpinnan oletuslämpötila: Päiväntasaajalla +28 °C, Navoilla -20 °C
    # Käytetään kosinifunktiota, jotta lämpötila laskee tasaisesti napa-alueita kohti
    #merenpinta_temp = -20 + 48 * np.cos(Y_leveyspiirit * np.pi / 2)
    merenpinta_temp = polar_temp + temp_diff * np.cos(Y_leveyspiirit * np.pi / 2)    
    # 2. LASKETAAN KORKEUDEN VAIKUTUS (Vain mantereille, eli merenpinnan yläpuolelle)
    # Muunnetaan taulukon 0.5 - 1.0 arvot todellisiksi metreiksi (0m - max_korkeus)
    korkeus_m = np.zeros_like(dema, dtype=float)
    manner_maski = dema > sealevel
    
    # Skaalataan mantereiden korkeus metreiksi
    #korkeus_m[manner_maski] = ((dem[manner_maski] - sealevel) / (1.0 - sealevel)) * max_korkeus_m
    korkeus_m=dema
    # Lämpötila laskee ~0.0065 °C per metri (Lapse rate)
    korkeus_vaikutus = korkeus_m * 0.0065
    
    # Lopullinen lämpötila: Merenpinnan lämpötila - korkeuden tuoma kylmeneminen
    lopullinen_temp = merenpinta_temp - korkeus_vaikutus
    #lopullinen_temp = merenpinta_temp   
    #lopullinen_temp = korkeus_vaikutus
    return lopullinen_temp


def laske_perus_ilmasto(
    relief, t_mean, delta_t, 
    tilt=23.5,
    ecc=0.0167,
    mvelp=102.9,
    P=1.0,
    kiertosolut=3
):
    """
    Laskee planeetan yksinkertaistetun ilmaston 12 kuukaudelle.

    Parametrit
    ----------
    relief : 2D numpy-taulukko
        Maaston korkeus metreinä.
        Muoto: [height, width]

    tilt : float
        Akselikallistuma asteina.
        Maa: 23.5

    ecc : float
        Radan eksentrisyys.
        Maa: 0.0167

    mvelp : float
        Perihelin sijainti asteina.

    P : float
        Vuoden pituus maavuosina.
        Vaikuttaa sademäärään.

    kiertosolut : int
        Ilmakehän kiertosolujen määrä per pallonpuolisko.
        Maa-tyyppinen oletus: 3.

    Palauttaa
    ----------
    kk_temp : ndarray
        Kuukausittaiset lämpötilat °C.
        Muoto: [12, height, width]

    kk_tuuli_x : ndarray
        Tuulen X-komponentti.
        Muoto: [12, height, width]

    kk_tuuli_y : ndarray
        Tuulen Y-komponentti.
        Muoto: [12, height, width]

    kk_tuuli_z : ndarray
        Tuulen Z-komponentti.
        Muoto: [12, height, width]

    kk_sade : ndarray
        Kuukausittainen sademäärä.
        Muoto: [12, height, width]
    """
    t_min=t_mean-(delta_t/2)
    t_max=t_mean+(delta_t/2)    
    monsoon_kerroin=0.5

    height, width = relief.shape

    distsea=distance_to_sea(relief, planet_radius)
    seamask=np.copy(relief)
    seamask=np.where(seamask<=0,1,0 )
    merisuunta_x, merisuunta_y = merisuunnan_laskenta(seamask)

	
    # ============================================================
    # KOORDINAATIT
    # ============================================================

    # +1 = pohjoisnapa
    #  0 = päiväntasaaja
    # -1 = etelänapa

    y_lin = np.linspace(
        1.0,
        -1.0,
        height
    )

    Y = np.repeat(
        y_lin[:, np.newaxis],
        width,
        axis=1
    )

    abs_y = np.abs(Y)

    # Leveysaste asteina

    latitude = (
        Y * 90.0
    )
    #plt.imshow(Y)
    #plt.show()
    #quit(-1)
    # ============================================================
    # TULOSTAULUKOT
    # ============================================================

    kk_temp = np.zeros(
        (12, height, width),
        dtype=float
    )

    kk_tuuli_x = np.zeros(
        (12, height, width),
        dtype=float
    )

    kk_tuuli_y = np.zeros(
        (12, height, width),
        dtype=float
    )

    kk_tuuli_z = np.zeros(
        (12, height, width),
        dtype=float
    )

    kk_sade = np.zeros(
        (12, height, width),
        dtype=float
    )

    # ============================================================
    # DEM:N GRADIENTIT
    # ============================================================
    #
    # Lasketaan kerran, koska itse DEM ei muutu kuukausien välillä.
    #
    # gradient palauttaa:
    #
    # grad_y = pystysuunnan muutos
    # grad_x = vaakasuunnan muutos

    grad_y, grad_x = np.gradient(
        relief.astype(float)
    )

    # ============================================================
    # ASTEET -> RADIAANIT
    # ============================================================

    tilt_rad = np.radians(
        tilt
    )

    mvelp_rad = np.radians(
        mvelp
    )

    # ============================================================
    # 12 KUUKAUTTA
    # ============================================================

    for kk in range(12):

        # ========================================================
        # AURINGON DEKLIINAATIO
        # ========================================================

        vuoden_kulma = 2.0 * np.pi * (kk / 12.0)

        aurinko_deklinaatio = np.degrees(
            tilt_rad * np.cos(vuoden_kulma)
        )
        #aurinko_y = (
        #aurinko_deklinaatio /
        #(np.pi / 2.0)
        #)
        aurinko_y=(aurinko_deklinaatio/90)
        # ========================================================
        # AURINGON ETÄISYYS
        # ========================================================

        rata_kulma = vuoden_kulma - mvelp_rad

        aurinkoetaisyys = (
            (1.0 - ecc**2) /
            (1.0 + ecc * np.cos(rata_kulma))
        )
        print("vuoden_kulma, rata_kulma, aurinko_deklinaatio, aurinko_y, aurinkoetaisyys")
        print(vuoden_kulma, rata_kulma, aurinko_deklinaatio, aurinko_y, aurinkoetaisyys)

        # ========================================================
        # AURINGON LÄMMITTÄVÄ VAIKUTUS
        # ========================================================

        latitude_rad = np.radians(latitude)
        deklinaatio_rad = np.radians(aurinko_deklinaatio)

        aurinko = (
            np.cos(latitude_rad - deklinaatio_rad)
            /
            aurinkoetaisyys**2
        )

        # Normalisointi
        aurinko = np.clip(aurinko, 0.0, 1.0)


        # ========================================================
        # LÄMPÖTILA
        # ========================================================

        t_pohja = (
            t_min +
            (t_max - t_min) * aurinko
        )
 
 
 
        # --------------------------------------------------------
        # 5. KORKEUDEN VAIKUTUS
        # --------------------------------------------------------

        korkeus_lampo = (
            relief /
            1000.0
        ) * 6.5  *1.0

        kk_temp[kk] = (
            t_pohja -
            korkeus_lampo
        )

        # ========================================================
        # TUULI
        # ========================================================
        #
        # TÄRKEÄ:
        #
        # Tuuli lasketaan nyt kokonaan laske_tuuli()-funktiossa.
        #
        # Jokaiselle kuukaudelle annetaan vuoden kulma, joten
        # ilmakehän kiertosolut voivat siirtyä vuodenaikojen mukana.

        (
            tuuli_x,
            tuuli_y,
            tuuli_z
        ) = laske_tuuli(
            relief,
            vuoden_kulma=vuoden_kulma,
            tilt=tilt,
            kiertosolut=kiertosolut
        )

        # ========================================================
        # PIENI LÄMPÖTILAN AIHEUTTAMA KONVEKTIO
        # ========================================================
        #
        # Kuumempi ilma -> hieman voimakkaampi nouseva liike.
        #
        # Tämä lisätään tässä, koska lämpötila on juuri laskettu.

        lampo_norm = np.clip(
            (
                kk_temp[kk] -
                20.0
            ) / 30.0,
            -1.0,
            1.0
        )

        tuuli_z = (
            tuuli_z +
            lampo_norm * 0.03
        )

        # ========================================================
        # NORMALISOINTI
        # ========================================================

        nopeus = np.sqrt(
            tuuli_x ** 2 +
            tuuli_y ** 2 +
            tuuli_z ** 2
        )

        nopeus = np.maximum(
            nopeus,
            1e-8
        )

        tuuli_x /= nopeus
        tuuli_y /= nopeus
        tuuli_z /= nopeus

        # ========================================================
        # TALLENNETAAN TUULI
        # ========================================================

        kk_tuuli_x[kk] = tuuli_x
        kk_tuuli_y[kk] = tuuli_y
        kk_tuuli_z[kk] = tuuli_z

        # ========================================================
        # SADE
        # ========================================================

        # --------------------------------------------------------
        # 1. Trooppinen sade
        # --------------------------------------------------------

        trooppinen_sade = (
            2400.0 *
            np.exp(
                -120.0 *
                (
                    Y -
                    aurinko_y*monsoon_kerroin ## ei nouse ihan kääntöpiirille 
                ) ** 2
            )
        )

        # --------------------------------------------------------
        # 2. Lauhkeiden alueiden sade
        # --------------------------------------------------------

        lauhkea_sade = (
            800.0 *
            np.exp(
                -25.0 *
                (
                    abs_y -
                    0.60
                ) ** 2
            )
        )

        # --------------------------------------------------------
        # 3. Kylmien alueiden sade
        # --------------------------------------------------------

        kylma_sade = (
            150.0 *
            np.exp(
                -5.0 *
                abs_y ** 2
            )
        )

        # --------------------------------------------------------
        # 4. Lämpötilan vaikutus kosteuteen
        # --------------------------------------------------------

        kosteus = np.clip(
            (
                kk_temp[kk] +
                15.0
            ) / 30.0,
            0.05,
            2.0
        )

        # ========================================================
        # 5. TUULEN MUKAINEN OROGRAFIA
        # ========================================================
        #
        # Nyt käytetään sekä X- että Y-tuulta.
        #
        # Positiivinen nousu:
        #     ilma liikkuu ylämäkeen
        #
        # Negatiivinen nousu:
        #     ilma liikkuu alamäkeen
        #
        # Tämä on paljon parempi kuin vanha pelkkä rinne_x.

        nousu = (
            tuuli_x * grad_x +
            tuuli_y * grad_y
        )

        # --------------------------------------------------------
        # Orogrfinen sadevaikutus
        # --------------------------------------------------------

        orografinen_efekti = (
            1.0 +
            1.2 *
            np.tanh(
                nousu /
                40.0
            )
        )

        orografinen_efekti = np.clip(
            orografinen_efekti,
            0.05,
            3.0
        )

        # ========================================================
        # 6. VUORISTON KORKEUS
        # ========================================================

        korkeus = np.maximum(
            relief,
            0.0
        )

        korkeus_efekti = (
            1.0 +
            0.15 *
            np.clip(
                korkeus / 3000.0,
                0.0,
                1.0
            )
            *
            np.clip(
                nousu / 20.0,
                0.0,
                1.0
            )
        )

        # ========================================================
        # 7. SADEKATVE
        # ========================================================

        sadekatve = np.ones_like(
            relief,
            dtype=float
        )

        alamaki = (
            nousu < 0
        )

        sadekatve[alamaki] = (
            1.0 +
            0.35 *
            np.tanh(
                nousu[alamaki] /
                40.0
            )
        )

        meriefekti=np.copy(distsea)
        #meriefekti=(1+np.exp(-distsea/2000))/2
        # tuulen nopeus vaakatasossa
        tuulinopeus = np.sqrt(tuuli_x**2 + tuuli_y**2)
        merituuli = -(tuuli_x * merisuunta_x + tuuli_y * merisuunta_y)
        tuulikerroin = np.clip(merituuli / (tuulinopeus + 1e-6), 0, 1)
        etaisyyskerroin = np.exp(-distsea / 20000)
        #meriefekti = etaisyyskerroin * (0.3 + 0.7 * tuulikerroin)
        tuulinopeus = np.sqrt(tuuli_x**2 + tuuli_y**2)
        tuulivoima = 1 - np.exp(-tuulinopeus / 5)

        meriefekti = (
        np.exp(-distsea / 20000)
        * (0.3 + 0.7 * tuulikerroin)
        * (0.5 + 0.5 * tuulivoima)
        )
        
        ## lämpötilan meriefekti
        T_etaisyysvaikutus = np.exp(-distsea / 20000)

        T_tuulensuunta = np.clip(
        merituuli / (tuulinopeus + 1e-6),
        0,
        1
        )

        T_meriefekti = T_etaisyysvaikutus * (
        0.3 + 0.7 * T_tuulensuunta
        )
        
        # ========================================================
        # 8. YHDISTETÄÄN SADE
        # ========================================================
        sade_leveysaste=(trooppinen_sade+lauhkea_sade+kylma_sade)/12

        sade_pohja = (
            kosteus *
            sade_leveysaste
            #* orografinen_efekti
            *
            korkeus_efekti
            *
            sadekatve *meriefekti
            +
            1 ## 40.0
        )

        # ========================================================
        # 9. KUUKAUSISADE
        # ========================================================

        ## kk_sade[kk] = np.clip(sade_pohja /12.0 *P,0.0,None)
        #kk_sade[kk] = sade_leveysaste*kosteus*korkeus_efekti*sadekatve
        kk_sade[kk] = sade_pohja*1.0
    # ============================================================
    # PALAUTUS
    # ============================================================

    return (
        kk_temp,
        kk_tuuli_x,
        kk_tuuli_y,
        kk_tuuli_z,
        kk_sade
    )





def laske_tuuli(
    dem,
    vuoden_kulma=0.0,
    tilt=23.5,
    kiertosolut=3
):
    """
    Laskee planeetan yksinkertaistetun 3D-tuulikentän.

    Parametrit
    ----------
    dem : np.ndarray
        Maaston korkeusdata.
        Meri = 0, maa > 0.

    vuoden_kulma : float
        Planeetan vuodenaika radiaaneina.
        0 ... 2*pi.

    tilt : float
        Akselikallistuma asteina.

    kiertosolut : int
        Ilmakehän kiertosolujen määrä per pallonpuolisko.
        Oletus 3 vastaa Hadley/Ferrel/polaarirakennetta.

    Palauttaa
    ----------
    tuuli_x, tuuli_y, tuuli_z : np.ndarray
        Normalisoidun tuulivektorin komponentit.
    """

    height, width = dem.shape

    # ============================================================
    # 1. LEVEYSPIIRIT
    # ============================================================

    y_lin = np.linspace(-1.0, 1.0, height)

    _, Y = np.meshgrid(
        np.arange(width),
        y_lin
    )

    # ============================================================
    # 2. AURINGON SIJAINTI
    # ============================================================

    tilt_rad = np.radians(tilt)

    aurinko_leveysaste = (
        tilt_rad *
        np.sin(vuoden_kulma)
    )

    aurinko_y = (
        aurinko_leveysaste /
        (np.pi / 2.0)
    )

    # ============================================================
    # 3. TUULIKENTÄN ALUSTUS
    # ============================================================

    tuuli_x = np.zeros_like(
        Y,
        dtype=float
    )

    tuuli_y = np.zeros_like(
        Y,
        dtype=float
    )

    tuuli_z = np.zeros_like(
        Y,
        dtype=float
    )

    # ============================================================
    # 4. MAA / MERI
    # ============================================================

    landmask = dem > 0

    # ============================================================
    # 5. ILMAKEHÄN KIERTOSOLUT
    # ============================================================
    #
    # Solut määritellään suhteessa siirtyneeseen päiväntasaajaan.
    #
    # aurinko_y:
    #
    #     pohjoisen kesä -> positiivinen
    #     etelän kesä    -> negatiivinen
    #
    # Siirtymä pidetään maltillisena, jotta koko ilmakehän
    # rakenne ei vaeltaisi epärealistisesti navalta toiselle.

    solu_siirtyma = (
        aurinko_y * 0.35
    )

    siirretty_y = (
        Y - solu_siirtyma
    )

    abs_siirretty_y = np.abs(
        siirretty_y
    )

    solu_leveys = (
        1.0 / kiertosolut
    )

    # ------------------------------------------------------------
    # Pohjoinen ja eteläinen pallonpuolisko
    # ------------------------------------------------------------

    for solu in range(kiertosolut):

        y_min = (
            solu *
            solu_leveys
        )

        y_max = (
            (solu + 1) *
            solu_leveys
        )

        maski = (
            (abs_siirretty_y >= y_min) &
            (abs_siirretty_y < y_max)
        )

        # --------------------------------------------------------
        # Solun pääasiallinen itä-länsisuuntainen virtaus
        # --------------------------------------------------------

        if solu % 2 == 0:
            suunta = -1.0
        else:
            suunta = 1.0

        tuuli_x[maski] = suunta

        # --------------------------------------------------------
        # Pieni pohjois-eteläsuuntainen komponentti
        # --------------------------------------------------------
        #
        # Ilmakehä ei todellisuudessa liiku täysin vaakasuorissa
        # kaistoissa. Lisätään siis heikko meridionaalinen virtaus.
        #
        # Solun keskikohta määrää suunnan.

        solu_keskikohta = (
            y_min +
            solu_leveys * 0.5
        )

        etaisyys_solun_keskelta = (
            abs_siirretty_y[maski] -
            solu_keskikohta
        )

        # Suunta vaihtuu solun sisällä.
        #
        # Kerroin pidetään pienenä, koska pääasiallinen tuuli
        # tulee edelleen X-suunnasta.

        meridionaalinen = (
            -np.sign(
                siirretty_y[maski]
            )
            *
            np.clip(
                np.abs(
                    etaisyys_solun_keskelta
                )
                /
                (solu_leveys * 0.5),
                0.0,
                1.0
            )
            *
            0.15
        )

        tuuli_y[maski] += (
            meridionaalinen
        )

    # ============================================================
    # 6. MAASTON KALTEVUUS
    # ============================================================

    dem_y, dem_x = np.gradient(
        dem.astype(float)
    )

    kaltevuus = np.hypot(
        dem_x,
        dem_y
    )

    # ============================================================
    # 7. MAASTON VAIKUTUS VAAKATUULEEN
    # ============================================================
    #
    # Tuuli pyrkii hieman ohjautumaan maaston muotojen mukaan.
    #
    # -dem_x:
    #   rinteen korkein kohta vasemmalla/oikealla
    #
    # -dem_y:
    #   vastaava pohjois-eteläsuunnassa.
    #
    # Vaikutus pidetään pienenä verrattuna globaaliin
    # kiertosolujärjestelmään.

    tuuli_x[landmask] += (
        -dem_x[landmask] *
        0.10
    )

    tuuli_y[landmask] += (
        -dem_y[landmask] *
        0.10
    )

    # ============================================================
    # 8. VUORISTON NOUSUVIRTA
    # ============================================================

    maan_kaltevuus = (
        kaltevuus[landmask]
    )

    if maan_kaltevuus.size > 0:

        vuoristo_raja = np.percentile(
            maan_kaltevuus,
            75
        )

        tasamaa_raja = np.percentile(
            maan_kaltevuus,
            50
        )

    else:

        vuoristo_raja = np.inf
        tasamaa_raja = 0.0

    vuoristo = (
        landmask &
        (kaltevuus >= vuoristo_raja)
    )

    # Mitä jyrkempi rinne,
    # sitä voimakkaampi nousukomponentti.

    tuuli_z[vuoristo] += (
        kaltevuus[vuoristo] *
        0.10
    )

    # ============================================================
    # 9. TASAISEN MAA-ALUEEN LASKEVA VIRTA
    # ============================================================

    tasainen_maa = (
        landmask &
        (kaltevuus <= tasamaa_raja)
    )

    tuuli_z[tasainen_maa] -= 0.02

    # ============================================================
    # 10. TERMinen KONVEKTIO
    # ============================================================
    #
    # Auringon alla oleva alue saa hieman nousevaa virtausta.
    #
    # Tämä tekee tropiikin tuulesta dynaamisemman ja yhdistää
    # tuulikentän vuodenaikaan.

    aurinko_ero = (
        np.abs(
            Y - aurinko_y
        )
    )

    konvektio = np.exp(
        -(aurinko_ero / 0.25) ** 2
    )

    tuuli_z += (
        konvektio *
        0.04
    )

    # ============================================================
    # 11. NORMALISOINTI
    # ============================================================

    nopeus = np.sqrt(
        tuuli_x**2 +
        tuuli_y**2 +
        tuuli_z**2
    )

    nopeus = np.maximum(
        nopeus,
        1e-8
    )

    tuuli_x /= nopeus
    tuuli_y /= nopeus
    tuuli_z /= nopeus

    return (
        tuuli_x,
        tuuli_y,
        tuuli_z
    )



def laske_tuuli_year_only(dem, kiertosolut=3):
    """
    Laskee planeetan yksinkertaistetun tuulikentän.

    Parametrit
    ----------
    dem : np.ndarray
        Maaston korkeusdata. Meri = 0, maa > 0.
    kiertosolut : int
        Ilmakehän kiertosolujen määrä per pallonpuolisko.
        Oletus 3 vastaa Maan kaltaista rakennetta.

    Palauttaa
    ----------
    tuuli_x, tuuli_y, tuuli_z : np.ndarray
        Tuulen komponentit.
    """

    # ---------------------------------------------------------
    # 1. LEVEYSPIIRIT
    # ---------------------------------------------------------

    y_lin = np.linspace(-1, 1, height)
    _, Y = np.meshgrid(np.arange(width), y_lin)

    tuuli_x = np.zeros_like(Y, dtype=float)
    tuuli_y = np.zeros_like(Y, dtype=float)
    tuuli_z = np.zeros_like(Y, dtype=float)

    # ---------------------------------------------------------
    # 2. MAA / MERI DEM:stä
    # ---------------------------------------------------------

    landmask = dem > 0

    # ---------------------------------------------------------
    # 3. GLOBAALIT ILMAKEHÄN KIERTOSOLUT
    # ---------------------------------------------------------

    solu_leveys = 1.0 / kiertosolut

    for solu in range(kiertosolut):

        y_min = solu * solu_leveys
        y_max = (solu + 1) * solu_leveys

        # Sama solurakenne molemmilla pallonpuoliskoilla.
        maski = (
            (np.abs(Y) >= y_min) &
            (np.abs(Y) < y_max)
        )

        # Vuorotteleva pääsuunta:
        # 0 = itätuuli
        # 1 = länsituuli
        # 2 = itätuuli
        # jne.

        if solu % 2 == 0:
            tuuli_x[maski] = -1.0
        else:
            tuuli_x[maski] = 1.0

    # ---------------------------------------------------------
    # 4. MAASTON KALTEVUUS
    # ---------------------------------------------------------

    dem_y, dem_x = np.gradient(dem)

    kaltevuus = np.hypot(dem_x, dem_y)

    # Prosenttipisteet lasketaan VAIN MAALTA.
    # Näin meren nollakorkeus ei vääristä kynnysarvoja.

    maan_kaltevuus = kaltevuus[landmask]

    if maan_kaltevuus.size > 0:
        vuoristo_raja = np.percentile(maan_kaltevuus, 75)
        tasamaa_raja = np.percentile(maan_kaltevuus, 50)
    else:
        # Planeetalla ei ole maata.
        vuoristo_raja = np.inf
        tasamaa_raja = 0.0

    # ---------------------------------------------------------
    # 5. MAASTON PAIKALLINEN VAIKUTUS TUULEEN
    # ---------------------------------------------------------

    # Pieni vaakasuuntainen vaikutus.
    # Globaali kiertosolujen vaikutus säilyy hallitsevana.

    tuuli_x[landmask] += -dem_x[landmask] * 0.10
    tuuli_y[landmask] += -dem_y[landmask] * 0.10

    # ---------------------------------------------------------
    # 6. VUORISTOJEN NOUSUVIRTA
    # ---------------------------------------------------------

    vuoristo = landmask & (kaltevuus >= vuoristo_raja)

    tuuli_z[vuoristo] += kaltevuus[vuoristo] * 0.10

    # ---------------------------------------------------------
    # 7. TASAISEN SISÄMAAN LASKEVA VIRTA
    # ---------------------------------------------------------

    # Tämä on kevyt approksimaatio mantereen sisäosien
    # korkeapainevaikutuksesta.
    #
    # Tässä vaiheessa emme vielä yritä laskea varsinaista
    # ilmanpainetta.

    tasainen_maa = (
        landmask &
        (kaltevuus <= tasamaa_raja)
    )

    tuuli_z[tasainen_maa] -= 0.02

    # ---------------------------------------------------------
    # 8. NORMALISOINTI
    # ---------------------------------------------------------

    nopeus = np.sqrt(
        tuuli_x**2 +
        tuuli_y**2 +
        tuuli_z**2
    )

    nopeus = np.maximum(nopeus, 1e-8)

    tuuli_x /= nopeus
    tuuli_y /= nopeus
    tuuli_z /= nopeus

    return tuuli_x, tuuli_y, tuuli_z

def laske_sademaara_soluilla_00(
    dem,
    etaisyys_mereen_km,
    sealevel=0.5,
    moisture_coeff=1.0
):
    """
    Laskee vuotuisen sademäärän (mm/vuosi).

    Sademäärään vaikuttavat:
    - leveyspiirin globaali sade
    - päiväntasaajan voimakas konvektio
    - etäisyys merestä
    - tuulen suunta
    - maaston nousu tuulen suunnassa
    - vuoriston korkeus
    - tuulenpuoleinen / suojanpuoleinen rinne

    DEM:
        meri = 0
        maa > 0
    """

    height, width = dem.shape

    # ---------------------------------------------------------
    # 1. LEVEYSPIIRIT
    # ---------------------------------------------------------

    y_lin = np.linspace(-1, 1, height)
    _, Y = np.meshgrid(np.arange(width), y_lin)

    abs_y = np.abs(Y)

    # ---------------------------------------------------------
    # 2. GLOBAALI PERUSSATEEN MÄÄRÄ
    # ---------------------------------------------------------
    #
    # Päiväntasaaja:
    # voimakas konvektio ja nouseva kostea ilma.
    #
    # Lauhkeat leveysasteet:
    # rintamien ja matalapaineiden aiheuttama sade.
    #
    # Subtrooppiset alueet:
    # kuivempia laskevan ilman vuoksi.

    trooppinen_sade = (
        2400 *
        np.exp(-60 * Y**2)
    )

    lauhkea_sade = (
        800 *
        np.exp(-25 * (abs_y - 0.60)**2)
    )

    perussade = (
        moisture_coeff *
        (
            trooppinen_sade +
            lauhkea_sade
        )
        + 80
    )

    # ---------------------------------------------------------
    # 3. TUULI
    # ---------------------------------------------------------

    tuuli_x, tuuli_y, tuuli_z = laske_tuuli(dem)

    # ---------------------------------------------------------
    # 4. MAA / MERI
    # ---------------------------------------------------------

    landmask = dem > sealevel

    # ---------------------------------------------------------
    # 5. MEREN KOSTEUS
    # ---------------------------------------------------------
    #
    # Mereltä kauemmas mentäessä kosteus vähenee.
    #
    # Tropiikissa vaikutus on kuitenkin paljon heikompi,
    # koska voimakas konvektio ja vesikierto voivat ylläpitää
    # kosteutta myös mantereen sisällä.
    #
    # Tämä on tärkeää Amazonin ja Kongon kaltaisille alueille.

    meri_kosteus = np.exp(
        -etaisyys_mereen_km / 700.0
    )

    # Trooppisuus:
    # 1 = päiväntasaajalla
    # 0 = kaukana päiväntasaajasta

    trooppisuus = np.exp(
        -12 * Y**2
    )

    # Tavallisilla leveysasteilla meri on tärkeä kosteuden lähde.
    #
    # Tropiikissa merietäisyyden vaikutusta lievennetään.

    meri_kerroin = (
        0.40 +
        0.60 * meri_kosteus
    )

    meri_kerroin = (
        meri_kerroin * (1.0 - 0.70 * trooppisuus)
        +
        trooppisuus
    )

    # ---------------------------------------------------------
    # 6. MAASTON KALTEVUUS
    # ---------------------------------------------------------

    grad_y, grad_x = np.gradient(dem)

    # Maaston gradientti tuulen suunnassa.
    #
    # Positiivinen:
    # tuuli liikkuu ylämäkeen
    #
    # Negatiivinen:
    # tuuli liikkuu alamäkeen

    nousu = (
        tuuli_x * grad_x +
        tuuli_y * grad_y
    )

    # ---------------------------------------------------------
    # 7. OROGRAFIA
    # ---------------------------------------------------------
    #
    # Nousu lisää sadetta.
    # Alamäkeen virtaaminen vähentää.
    #
    # tanh estää yksittäisiä valtavia DEM-arvoja
    # räjäyttämästä sademäärää.

    orografinen_efekti = (
        1.0 +
        1.2 * np.tanh(nousu / 40.0)
    )

    # ---------------------------------------------------------
    # 8. VUORISTON KORKEUS
    # ---------------------------------------------------------
    #
    # Korkeus EI suoraan tuota sadetta.
    #
    # Sen vaikutus on pieni lisä vuoriston tuulenpuoleiselle
    # alueelle, jossa ilma jo muutenkin pakotetaan kohoamaan.
    #
    # Korkeuden vaikutus siis riippuu noususta.

    korkeus = np.maximum(dem - sealevel, 0)

    korkeus_efekti = (
        1.0 +
        0.15 *
        np.clip(korkeus / 3000.0, 0, 1) *
        np.clip(nousu / 20.0, 0, 1)
    )

    # ---------------------------------------------------------
    # 9. SADEKATVE
    # ---------------------------------------------------------
    #
    # Kun ilma liikkuu alamäkeen, sateen määrä vähenee.
    #
    # Erityisesti vuoriston suojanpuolella syntyy kuiva alue.

    sadekatve = np.ones_like(dem)

    alamaki = nousu < 0

    sadekatve[alamaki] = (
        1.0 +
        0.35 * np.tanh(nousu[alamaki] / 40.0)
    )

    # ---------------------------------------------------------
    # 10. YHDISTETÄÄN TEKIJÄT
    # ---------------------------------------------------------

    sademaara = (
        perussade
        * meri_kerroin
        * orografinen_efekti
        * korkeus_efekti
        * sadekatve
    )

    # ---------------------------------------------------------
    # 11. MERI / MANNER
    # ---------------------------------------------------------
    #
    # Merellä sade määräytyy pääasiassa globaalin kosteuden
    # ja trooppisen konvektion perusteella.
    #
    # Mantereella mereltä tulevan kosteuden merkitys korostuu.

    # Ei vielä erillistä merisadetta:
    # meri saa luonnollisesti korkean trooppisen sateen,
    # mutta mantereen sisämaa kuivuu merikosteuden vähentyessä.

    # ---------------------------------------------------------
    # 12. RAJOITETAAN TULOS
    # ---------------------------------------------------------

    sademaara = np.clip(
        sademaara,
        40,
        5000
    )

    return sademaara, tuuli_x



def laske_sademaara_soluilla(
    dem,
    etaisyys_mereen_km,
    sealevel=0.5,
    moisture_coeff=1.0
):
    """
    Laskee vuotuisen sademäärän (mm/vuosi).

    Sademäärään vaikuttavat:
    - leveyspiirin globaali sade
    - päiväntasaajan voimakas konvektio
    - etäisyys merestä
    - tuulen suunta
    - maaston nousu tuulen suunnassa
    - vuoriston korkeus
    - tuulenpuoleinen / suojanpuoleinen rinne

    DEM:
        meri = 0
        maa > 0
    """

    height, width = dem.shape

    # ---------------------------------------------------------
    # 1. LEVEYSPIIRIT
    # ---------------------------------------------------------

    y_lin = np.linspace(-1, 1, height)
    _, Y = np.meshgrid(np.arange(width), y_lin)

    abs_y = np.abs(Y)

    # ---------------------------------------------------------
    # 2. GLOBAALI PERUSSATEEN MÄÄRÄ
    # ---------------------------------------------------------
    #
    # Päiväntasaaja:
    # voimakas konvektio ja nouseva kostea ilma.
    #
    # Lauhkeat leveysasteet:
    # rintamien ja matalapaineiden aiheuttama sade.
    #
    # Subtrooppiset alueet:
    # kuivempia laskevan ilman vuoksi.

    trooppinen_sade = (
        2400 *
        np.exp(-60 * Y**2)
    )

    lauhkea_sade = (
        800 *
        np.exp(-25 * (abs_y - 0.60)**2)
    )

    perussade = (
        moisture_coeff *
        (
            trooppinen_sade +
            lauhkea_sade
        )
        + 80
    )

    # ---------------------------------------------------------
    # 3. TUULI
    # ---------------------------------------------------------

    tuuli_x, tuuli_y, tuuli_z = laske_tuuli(dem)

    # ---------------------------------------------------------
    # 4. MAA / MERI
    # ---------------------------------------------------------

    landmask = dem > sealevel

    # ---------------------------------------------------------
    # 5. MEREN KOSTEUS
    # ---------------------------------------------------------
    #
    # Mereltä kauemmas mentäessä kosteus vähenee.
    #
    # Tropiikissa vaikutus on kuitenkin paljon heikompi,
    # koska voimakas konvektio ja vesikierto voivat ylläpitää
    # kosteutta myös mantereen sisällä.
    #
    # Tämä on tärkeää Amazonin ja Kongon kaltaisille alueille.

    meri_kosteus = np.exp(
        -etaisyys_mereen_km / 700.0
    )

    # Trooppisuus:
    # 1 = päiväntasaajalla
    # 0 = kaukana päiväntasaajasta

    trooppisuus = np.exp(
        -12 * Y**2
    )

    # Tavallisilla leveysasteilla meri on tärkeä kosteuden lähde.
    #
    # Tropiikissa merietäisyyden vaikutusta lievennetään.

    meri_kerroin = (
        0.40 +
        0.60 * meri_kosteus
    )

    meri_kerroin = (
        meri_kerroin * (1.0 - 0.70 * trooppisuus)
        +
        trooppisuus
    )

    # ---------------------------------------------------------
    # 6. MAASTON KALTEVUUS
    # ---------------------------------------------------------

    grad_y, grad_x = np.gradient(dem)

    # Maaston gradientti tuulen suunnassa.
    #
    # Positiivinen:
    # tuuli liikkuu ylämäkeen
    #
    # Negatiivinen:
    # tuuli liikkuu alamäkeen

    nousu = (
        tuuli_x * grad_x +
        tuuli_y * grad_y
    )

    # ---------------------------------------------------------
    # 7. OROGRAFIA
    # ---------------------------------------------------------
    #
    # Nousu lisää sadetta.
    # Alamäkeen virtaaminen vähentää.
    #
    # tanh estää yksittäisiä valtavia DEM-arvoja
    # räjäyttämästä sademäärää.

    orografinen_efekti = (
        1.0 +
        1.2 * np.tanh(nousu / 40.0)
    )

    # ---------------------------------------------------------
    # 8. VUORISTON KORKEUS
    # ---------------------------------------------------------
    #
    # Korkeus EI suoraan tuota sadetta.
    #
    # Sen vaikutus on pieni lisä vuoriston tuulenpuoleiselle
    # alueelle, jossa ilma jo muutenkin pakotetaan kohoamaan.
    #
    # Korkeuden vaikutus siis riippuu noususta.

    korkeus = np.maximum(dem - sealevel, 0)

    korkeus_efekti = (
        1.0 +
        0.15 *
        np.clip(korkeus / 3000.0, 0, 1) *
        np.clip(nousu / 20.0, 0, 1)
    )

    # ---------------------------------------------------------
    # 9. SADEKATVE
    # ---------------------------------------------------------
    #
    # Kun ilma liikkuu alamäkeen, sateen määrä vähenee.
    #
    # Erityisesti vuoriston suojanpuolella syntyy kuiva alue.

    #sadekatve = np.ones_like(dem)

    #alamaki = nousu < 0

    #sadekatve[alamaki] = (
    #    1.0 +
    #    0.35 * np.tanh(nousu[alamaki] / 40.0)
    #)

    # ========================================================
    # 7. SADEKATVE
    # ========================================================

    sadekatve, nousu = laske_sadevarjo(
    relief=relief,
    tuuli_x=tuuli_x,
    tuuli_y=tuuli_y,
    tuuli_z=tuuli_z,

    # Esimerkiksi 1 km rasteri
    solukoko=1000.0,

    # Etsitään vuoria 200 km tuulen yläpuolelta
    max_etaisyys=200_000.0,

    # Tarkistus 2 km välein
    askel=2000.0,

    # Sadevarjon herkkyys
    kulma_asteikko=8.0,

    # Kuinka nopeasti varjo häviää
    etaisyys_asteikko=100_000.0,

    # Maksimissaan 85 % vähennys
    varjon_voimakkuus=0.85,
    )

    # ---------------------------------------------------------
    # 10. YHDISTETÄÄN TEKIJÄT
    # ---------------------------------------------------------

    sademaara = (
        perussade
        * meri_kerroin
        * korkeus_efekti
        * sadekatve
        *(0.7+ 0.3*orografinen_efekti)
    )

    # ---------------------------------------------------------
    # 11. MERI / MANNER
    # ---------------------------------------------------------
    #
    # Merellä sade määräytyy pääasiassa globaalin kosteuden
    # ja trooppisen konvektion perusteella.
    #
    # Mantereella mereltä tulevan kosteuden merkitys korostuu.

    # Ei vielä erillistä merisadetta:
    # meri saa luonnollisesti korkean trooppisen sateen,
    # mutta mantereen sisämaa kuivuu merikosteuden vähentyessä.

    # ---------------------------------------------------------
    # 12. RAJOITETAAN TULOS
    # ---------------------------------------------------------

    sademaara = np.clip(
        sademaara,
        40,
        5000
    )

    return sademaara, tuuli_x


def luo_biomikartta(dem, lampotilat, sateet, sealevel=0.5):
    """
    Luokittelee maaston biomeihin lämpötilan ja sademäärän perusteella.
    Palauttaa indeksikartan ja värikartan visualisointia varten.
    """
    height, width = dem.shape
    
    # Määritetään biomien numeeriset id-tunnukset
    BIOMIT = {
        'MERI': 0,
        'AAVIKKO': 1,
        'SAVANNI_RUOHOKKO': 2,
        'SADEMETSÄ': 3,
        'LAUHKEA_METSÄ': 4,
        'HAVUMETSÄ': 5,
        'TUNDRA': 6,
        'IKIJÄÄ': 7
    }
    
    # Luodaan tyhjä kartta, joka täytetään oletuksena merellä
    biomi_kartta = np.zeros((height, width), dtype=int)
    
    # Maski mantereelle (merenpinnan yläpuolella oleva maasto)
    manner = dem > sealevel
    
    # Haetaan lämpötila ja sade vain mannerpisteistä helpompaa hakua varten
    T = lampotilat
    P = sateet
    
    # --- LUOKITTELUSÄÄNNÖT (Whittakerin malli mukautettuna) ---
    
    # 1. Ikijäät ja kylmimmät alueet
    ikijaa = manner & (T < -10)
    biomi_kartta[ikijaa] = BIOMIT['IKIJÄÄ']
    
    # 2. Tundra (kylmä, vähän sadetta)
    tundra = manner & (T >= -10) & (T < 0)
    biomi_kartta[tundra] = BIOMIT['TUNDRA']
    
    # 3. Havumetsä / Taiga (viileä ilmasto)
    havumetsa = manner & (T >= 0) & (T < 8) & (P >= 200)
    biomi_kartta[havumetsa] = BIOMIT['HAVUMETSÄ']
    
    # 4. Aavikko (kuivat alueet lämpötilasta riippumatta, paitsi arktiset)
    aavikko = manner & (T >= 0) & (P < 250)
    biomi_kartta[aavikko] = BIOMIT['AAVIKKO']
    
    # 5. Lauhkea metsä (lehtimetsät ja seka-alueet)
    lauhkea = manner & (T >= 8) & (T < 18) & (P >= 250)
    biomi_kartta[lauhkea] = BIOMIT['LAUHKEA_METSÄ']
    
    # 6. Savanni ja trooppinen ruohikko (lämmin, keskiverto tai kausittainen sade)
    savanni = manner & (T >= 18) & (P >= 250) & (P < 1500)
    biomi_kartta[savanni] = BIOMIT['SAVANNI_RUOHOKKO']
    
    # 7. Trooppinen sademetsä (kuuma ja erittäin sateinen)
    sademetsa = manner & (T >= 18) & (P >= 1500)
    biomi_kartta[sademetsa] = BIOMIT['SADEMETSÄ']
    
    # Korjataan mahdolliset manneralueet, jotka jäivät rajojen väliin (oletus ruohikkoon/aavikkoon)
    nolla_manner = manner & (biomi_kartta == 0)
    biomi_kartta[nolla_manner] = BIOMIT['SAVANNI_RUOHOKKO']

    return biomi_kartta

import numpy as np
from scipy.ndimage import label

def laske_mantereet(relief, planet_radius=6371.0):
    """
    Laskee mantereiden pinta-alat ja tunnistaa ne suuruusjärjestyksessä.
    
    Parametrit:
    - relief: 2D numpy-taulukko (Y-akseli = latitude -90..90, X-akseli = longitude -180..180)
    - planet_radius: Planeetan säde (oletus 6371.0 km)
    
    Palauttaa:
    - osuus_taulukko: Taulukko, jossa maa-pikselin arvo on sen edustaman mantereen 
                      prosenttiosuus koko planeetan pinta-alasta (0.0 jos merta)
    - jarjestys_taulukko: Taulukko, jossa meren arvo on 0, suurin manner 1, toiseksi suurin 2 jne.
    """
    ny, nx = relief.shape
    
    # 1. Määritetään maa-alueet (kaikki merenpinnan yläpuolella > 0)
    maa_maski = relief > 0
    
    # 2. Lasketaan pikselien pinta-alat pallon pinnalla
    # Jaetaan leveysasteet (-90 to 90) tasan pikselirivien kesken
    lat_edges = np.linspace(-90, 90, ny + 1)
    # Muunnetaan radiaaneiksi obliquity/integrointia varten
    lat_edges_rad = np.radians(lat_edges)
    
    # Lasketaan jokaisen rivin (leveysastekaistan) pinta-alaosuus koko pallosta.
    # Pallon vyöhykkeen ala: A = 2 * pi * R^2 * (sin(lat2) - sin(lat1))
    # Koko pallon ala: A_total = 4 * pi * R^2
    # Osuus koko pallosta: (sin(lat2) - sin(lat1)) / 2
    rivien_osuudet = (np.sin(lat_edges_rad[1:]) - np.sin(lat_edges_rad[:-1])) / 2.0
    
    # Koska rivi jakautuu nx-määrään pituusastepikseleitä, yhden pikselin osuus on:
    pikselien_osuudet_rivilla = rivien_osuudet / nx
    
    # Luodaan koko taulukon kokoinen pinta-alaosuusmatriisi (broadcasting)
    # numpy.newaxis tekee 1D-vektorista pystysuuntaisen (ny, 1), joka monistuu nx-leveyteen
    pikseli_osuudet = pikselien_osuudet_rivilla[:, np.newaxis] * np.ones((1, nx))
    
    # Lasketaan todelliset neliökilometrit (Koko ala = 4 * pi * R^2)
    koko_pinta_ala = 4 * np.pi * (planet_radius ** 2)
    pikseli_alat_km2 = pikseli_osuudet * koko_pinta_ala
    
    # 3. Tunnistetaan yhtenäiset maa-alueet (mantereet)
    # scipy.ndimage.label ryhmittelee vierekkäiset True-arvot omiksi saarekkeikseen.
    # HUOM: Tämä ei ota huomioon itä-länsi-suuntaista pallo-kiertoa (-180 ja 180 rajan yli).
    muokatut_mantereet, n_manteretta = label(maa_maski)
    
    # Lasketaan jokaisen löydetyn mantereen kokonaispinta-ala ja sen osuus planeetasta
    manner_alat_km2 = {}
    manner_osuudet = {}
    
    for i in range(1, n_manteretta + 1):
        manner_maski = (muokatut_mantereet == i)
        alan_osuus = np.sum(pikseli_osuudet[manner_maski])
        ala_km2 = np.sum(pikseli_alat_km2[manner_maski])
        
        manner_osuudet[i] = alan_osuus
        manner_alat_km2[i] = ala_km2

    # 4. Järjestetään mantereet suuruusjärjestykseen pinta-alan mukaan (suurin ensin)
    jarjestetyt_id_parit = sorted(manner_osuudet.items(), key=lambda x: x[1], reverse=True)
    
    # Luodaan tyhjät tulostaulukot
    osuus_taulukko = np.zeros_like(relief, dtype=float)
    jarjestys_taulukko = np.zeros_like(relief, dtype=int)
    
    # Täytetään taulukot uudella järjestyksellä (1 = suurin, 2 = toiseksi suurin...)
    for uusi_indeksi, (vanha_id, osuus) in enumerate(jarjestetyt_id_parit, start=1):
        manner_maski = (muokatut_mantereet == vanha_id)
        
        osuus_taulukko[manner_maski] = osuus
        jarjestys_taulukko[manner_maski] = uusi_indeksi
        
        # Tulostetaan vähän lisätietoa top-mantereista konsoliin
        if uusi_indeksi <= 5:  # Näytetään esim. 5 suurinta
            print(f"Manner {uusi_indeksi}: Pinta-ala = {manner_alat_km2[vanha_id]:,.1f} km², Osuus planeetasta = {osuus*100:.2f}%")

    return osuus_taulukko, jarjestys_taulukko


import numpy as np

def laske_globaalit_pikselialat(height, width, planet_radius_km):
    """
    Laskee koko planeetan kattavan rasterin pikselikohtaiset pinta-alat (km²).
    
    height: Rasterin rivien määrä (pohjoisesta etelään, +90 -> -90)
    width:  Rasterin sarakkeiden määrä (lännestä itään, -180 -> 180)
    planet_radius_km: Planeetan säde kilometreinä
    """
    # 1. Luodaan leveysasteiden rajat (height + 1 kpl linjoja) pohjoisesta etelään
    lat_edges = np.linspace(90, -90, height + 1)
    lat_edges_rad = np.radians(lat_edges)
    
    # 2. Yhden pikselin pituusasteen leveys radiaaneina (koko pallo = 360 astetta)
    dlon_rad = np.radians(360.0 / width)
    
    # 3. Lasketaan jokaisen rivin pinta-ala (sinipintojen erotus)
    sin_lat = np.sin(lat_edges_rad)
    # sin_lat[:-1] on pikselin yläreuna, sin_lat[1:] on alareuna
    rivikohtaiset_alat = (planet_radius_km**2) * dlon_rad * (sin_lat[:-1] - sin_lat[1:])
    
    # 4. Monistetaan rivien alat kaikille sarakkeille -> muotoon (height, width)
    pikseli_alat = np.repeat(rivikohtaiset_alat[:, np.newaxis], width, axis=1)
    
    return pikseli_alat



def laske_albedoluokat(landmask, lampotilat, sateet):

    # Alustetaan albedorasteri NaN-arvoilla (float64, jotta tukee NaN-arvoja)
    #albedo = np.full_like(landmask.shape, np.nan, dtype=np.float64)
    albedo = np.copy(landmask)
    albedo = np.where(albedo==0,np.nan,0)   

    # Tehdään maski vain maa-alueille, joissa on dataa
    maa_maski = np.copy(landmask) == 1

    # Haetaan helpompaa käsittelyä varten vain maa-alueiden arvot
    T = lampotilat[maa_maski]
    P = sateet[maa_maski]

    # Luodaan taulukko, johon lasketaan kunkin solun albedo
    maan_albedo = np.zeros_like(T, dtype=np.float64)

    # --- ALBEDOLAJIEN MÄÄRITTELY (10 eri luokkaa ilmaston mukaan) ---

    # 1. Tuore pysyvä lumi / Jäätikkö (Erittäin kylmä ja sateinen/luminen)
    lumi_maski = (T <= -10) & (P >= 500)
    maan_albedo[lumi_maski] = 0.85

    # 2. Vanha lumi / Kulunut jää (Erittäin kylmä ja kuiva)
    vanha_lumi_maski = (T <= -10) & (P < 500)
    maan_albedo[vanha_lumi_maski] = 0.65

    # 3. Tundra / Kylmä kasvillisuus (Kylmä, vähän sadetta)
    tundra_maski = (T > -10) & (T <= 0) & (P < 400)
    maan_albedo[tundra_maski] = 0.25

    # 4. Havumetsä (Taiga) (Viileä, kohtalainen sademäärä)
    havumetsa_maski = (T > -5) & (T <= 5) & (P >= 400)
    maan_albedo[havumetsa_maski] = 0.12

    # 5. Lehtimetsä (Lauhkea ja sateinen)
    lehtimetsa_maski = (T > 5) & (T <= 15) & (P >= 600)
    maan_albedo[lehtimetsa_maski] = 0.18

    # 6. Ruohikko / Preeria (Lauhkea ja kuivahko)
    ruohikko_maski = (T > 5) & (T <= 15) & (P < 600)
    maan_albedo[ruohikko_maski] = 0.20

    # 7. Hiekka-aavikko (Kuuma ja erittäin kuiva)
    hiekka_aavikko_maski = (T > 15) & (P < 150)
    maan_albedo[hiekka_aavikko_maski] = 0.40

    # 8. Puoliaavikko / Kuiva pensasto (Kuuma ja kuiva)
    puoliaavikko_maski = (T > 15) & (P >= 150) & (P < 400)
    maan_albedo[puoliaavikko_maski] = 0.28

    # 9. Savanni (Kuuma, selkeä kuiva- ja sadekausi)
    savanni_maski = (T > 15) & (P >= 400) & (P < 1200)
    maan_albedo[savanni_maski] = 0.15

    # 10. Trooppinen sademetsä (Erittäin kuuma ja erittäin sateinen)
    sademetsa_maski = (T > 18) & (P >= 1200)
    maan_albedo[sademetsa_maski] = 0.10
    
    # Sijoitetaan lasketut arvot takaisin alkuperäiseen rasterimuotoon
    albedo[maa_maski] = maan_albedo

    return albedo

import numpy as np


def find_max_points(
    raster,
    planet_radius_km,
    min_distance_km,
    n_points=10,
):
    """
    Etsii rasterin suurimmat pisteet siten, että valittujen pisteiden
    välinen pallopinnalla laskettu etäisyys on vähintään min_distance_km.

    Parametrit
    ----------
    raster : np.ndarray
        2D-taulukko muodossa (height, width).

    planet_radius_km : float
        Planeetan säde kilometreinä.

    min_distance_km : float
        Valittujen pisteiden pienin sallittu etäisyys kilometreinä.

    n_points : int
        Kuinka monta pistettä enintään palautetaan.

    Palauttaa
    ----------
    list of dict
        Jokaiselle pisteelle:
        {
            "row": int,
            "col": int,
            "lon": float,
            "lat": float,
            "value": float
        }

    Rasterin koordinaatisto
    -----------------------
    Rasterin oletetaan kattavan koko pallon:

        longitude: -180 ... +180 astetta
        latitude:   +90 ... -90 astetta

    Pikselin keskipisteet määritetään näin:

        lon = -180 + (col + 0.5) * 360 / width
        lat =  +90 - (row + 0.5) * 180 / height
    """

    raster = np.asarray(raster)

    if raster.ndim != 2:
        raise ValueError("rasterin pitää olla 2D-taulukko")

    if planet_radius_km <= 0:
        raise ValueError("planet_radius_km pitää olla > 0")

    if min_distance_km < 0:
        raise ValueError("min_distance_km pitää olla >= 0")

    if n_points <= 0:
        return []

    height, width = raster.shape

    # Kaikki rasteripikselit
    rows, cols = np.indices((height, width))

    # Pikselien keskipisteiden lon/lat
    lons = -180.0 + (cols + 0.5) * 360.0 / width
    lats = 90.0 - (rows + 0.5) * 180.0 / height

    # Järjestetään pikselit suurimman arvon mukaan.
    # NaN-arvot jätetään pois.
    values = raster.ravel()

    valid = np.isfinite(values)

    flat_indices = np.flatnonzero(valid)

    # Suurin ensin
    flat_indices = flat_indices[
        np.argsort(values[flat_indices])[::-1]
    ]

    selected = []

    # Muunnetaan pisteet radiaaneiksi
    lat_rad = np.deg2rad(lats.ravel())
    lon_rad = np.deg2rad(lons.ravel())

    min_angle = min_distance_km / planet_radius_km

    # Haversinen avulla lasketaan kulma kahden pisteen välillä.
    def angular_distance(i, j):
        dlat = lat_rad[j] - lat_rad[i]
        dlon = lon_rad[j] - lon_rad[i]

        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(lat_rad[i])
            * np.cos(lat_rad[j])
            * np.sin(dlon / 2.0) ** 2
        )

        return 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

    # Käydään suurimmat pisteet läpi.
    for idx in flat_indices:

        # Tarkistetaan etäisyys jo valittuihin pisteisiin
        too_close = False

        for selected_idx in selected:
            angle = angular_distance(idx, selected_idx)

            if angle < min_angle:
                too_close = True
                break

        if too_close:
            continue

        row = idx // width
        col = idx % width

        selected.append(idx)

        if len(selected) >= n_points:
            break

    # Muodostetaan tulos
    result = []

    for idx in selected:
        row = idx // width
        col = idx % width

        result.append({
            "row": int(row),
            "col": int(col),
            "lon": float(lons[row, col]),
            "lat": float(lats[row, col]),
            "value": float(raster[row, col]),
        })

    return result

def pixel_to_lonlat(x, y, width, height):
    # Määritellään annetut rajat
    lon_min, lon_max = -180.0, 180.0
    lat_min, lat_max = -90.0, 90.0
    
    # Lasketaan yhden pikselin koko asteina (resoluutio)
    lon_res = (lon_max - lon_min) / width
    lat_res = (lat_max - lat_min) / height
    
    # Lasketaan koordinaatit pikselin keskipisteelle (+ 0.5)
    lon = float(lon_min + (x + 0.5) * lon_res)
    lat = float(lat_max - (y + 0.5) * lat_res)  # Huom: lat_maxista vähennetään alaspäin
    
    return lon, lat

def onko_rannikko(y, x):
    if land_mask[y, x] == 0:
        return False

    for dy, dx in suunnat:
        ny = y + dy
        nx = (x + dx) % width

        if ny < 0 or ny >= height:
            continue

        if land_mask[ny, nx] == 0:
            return True

    return False



def arvioi_merijaa(landmask, lampotila_c, sademaara_mm):
    sea_ice_threshold=-1.8
    is_land=np.copy(landmask)
    sea_ice=np.copy(lampotila_c)
    sea_ice=np.where(sea_ice<sea_ice_threshold,1,0)
    seamask=np.copy(landmask)
    seamask=np.where(seamask<1,1,0)
    sea_ice=sea_ice*seamask        
    return (sea_ice)


def laske_holdridge_luokat(lampotila_matriisi, sade_matriisi, nodata_arvo=None):
    """
    Laskee Holdridgen elämänmuotoluokat rasterimatriiseista.
    
    Parametrit:
    - lampotila_matriisi: numpy.ndarray (Vuoden keskilämpötila °C)
    - sade_matriisi: numpy.ndarray (Vuotuinen sademäärä mm)
    - nodata_arvo: Alkuperäinen NoData-arvo (korvataan laskennan ajaksi NaN-arvolla)
    
    Palauttaa:
    - luokiteltu_matriisi: numpy.ndarray (Holdridge-luokat 1-5, NoData on -1)
    """
    # Kopioidaan matriisit, jotta alkuperäinen data ei muutu
    t = lampotila_matriisi.copy().astype(float)
    p = sade_matriisi.copy().astype(float)
    
    # NoData-arvojen käsittely
    if nodata_arvo is not None:
        t[t == nodata_arvo] = np.nan
        p[p == nodata_arvo] = np.nan
        
    # 1. Biolämpötilan approksimaatio (Holdridge: rajataan välille 0°C - 30°C)
    biolampotila = np.where(t < 0, 0, t)
    biolampotila = np.where(biolampotila > 30, 30, biolampotila)
    
    # 2. Alustetaan tyhjä matriisi luokitukselle
    luokat = np.zeros_like(biolampotila, dtype=float)
    
    # 3. Luokitussäännöt matriisioperaatioina
    luokat = np.where(biolampotila < 1.5, 5, luokat)  # Tundra / Alpiininen
    luokat = np.where((biolampotila >= 1.5) & (p < 250), 1, luokat)      # Aavikko
    luokat = np.where((biolampotila >= 1.5) & (p >= 250) & (p < 500), 2, luokat)  # Steppe
    luokat = np.where((biolampotila >= 1.5) & (p >= 500) & (p < 1000), 3, luokat) # Kuiva metsä
    luokat = np.where((biolampotila >= 1.5) & (p >= 1000), 4, luokat)   # Kostea / Sademetsä
    
    # Palautetaan NoData-arvot takaisin lukuna -1
    luokat[np.isnan(t) | np.isnan(p)] = -1
    
    return luokat.astype(int)



def calculate_pet(
    temperature,
    dem,
    landmask,
    lat_min=-90.0,
    lat_max=90.0,
    lapse_rate=0.0065,
    temp_amplitude=12.0
):
    """
    Laskee vuosittaisen potentiaalisen haihdunnan (PET)
    Thornthwaiten menetelmällä.

    Parametrit
    ----------
    temperature : np.ndarray
        Vuoden keskilämpötila, °C.
        Muoto (height, width).

    dem : np.ndarray
        Korkeus metreinä.
        Muoto (height, width).

    landmask : np.ndarray
        Maa/meri-mask:
            1 = maa
            0 = meri

        Muoto (height, width).

    lat_min : float
        Rasterin eteläisin leveysaste.
        Oletus -90°.

    lat_max : float
        Rasterin pohjoisin leveysaste.
        Oletus 90°.

    lapse_rate : float
        Lämpötilan pystysuuntainen gradientti °C/m.
        Oletus 0.0065 °C/m = 6.5 °C/km.

    temp_amplitude : float
        Vuotuinen lämpötilavaihtelun amplitudi °C.

    Palauttaa
    ----------
    np.ndarray
        Vuosittainen PET, mm/v.
    """

    # =========================================================
    # Tarkistukset
    # =========================================================

    if temperature.shape != dem.shape:
        raise ValueError(
            "temperature ja dem eivät ole saman kokoisia."
        )

    if temperature.shape != landmask.shape:
        raise ValueError(
            "temperature ja landmask eivät ole saman kokoisia."
        )

    height, width = temperature.shape

    # =========================================================
    # Leveysaste jokaiselle rasteririville
    # =========================================================

    # Rasterin pikselikeskusten leveysasteet.
    #
    # Ensimmäinen rivi = pohjoisin
    # Viimeinen rivi = eteläisin
    #
    # Jos rasterisi on järjestetty toisinpäin, tämä voidaan
    # helposti kääntää.

    latitudes = np.linspace(
        lat_max,
        lat_min,
        height
    )

    # Muutetaan muotoon (height, 1), jotta NumPy broadcasting
    # antaa saman leveysasteen koko riville.

    latitude = latitudes[:, np.newaxis]

    # =========================================================
    # Lämpötilan korkeussäätö
    # =========================================================

    temp = (
        temperature
        - lapse_rate * dem
    )

    # =========================================================
    # Kuukaudet
    # =========================================================

    month_day = np.array([
        15,  # Jan
        46,  # Feb
        74,  # Mar
        105, # Apr
        135, # May
        166, # Jun
        196, # Jul
        227, # Aug
        258, # Sep
        288, # Oct
        319, # Nov
        349  # Dec
    ])

    days = np.array([
        31, 28, 31, 30, 31, 30,
        31, 31, 30, 31, 30, 31
    ])

    # =========================================================
    # Vuodenaikainen lämpötilavaihtelu
    # =========================================================

    season = -np.cos(
        2 * np.pi
        * (month_day - 15)
        / 365.0
    )

    # =========================================================
    # Kuukausilämpötilat
    # =========================================================

    monthly_temp = []

    for i in range(12):

        T = (
            temp
            + temp_amplitude * season[i]
        )

        monthly_temp.append(T)

    # =========================================================
    # Thornthwaite lämpöindeksi
    # =========================================================

    I = np.zeros_like(
        temp,
        dtype=np.float64
    )

    for T in monthly_temp:

        positive = T > 0

        I += np.where(
            positive,
            (T / 5.0) ** 1.514,
            0.0
        )

    # =========================================================
    # Thornthwaiten a-kerroin
    # =========================================================

    a = (
        6.75e-7 * I**3
        - 7.71e-5 * I**2
        + 1.792e-2 * I
        + 0.49239
    )

    # =========================================================
    # PET
    # =========================================================

    pet_annual = np.zeros_like(
        temp,
        dtype=np.float64
    )

    # I = 0 aiheuttaisi division by zero
    safe_I = np.where(
        I > 0,
        I,
        1.0
    )

    # Leveysaste radiaaneiksi
    phi = np.radians(latitude)

    # =========================================================
    # Kuukausittainen laskenta
    # =========================================================

    for i in range(12):

        T = monthly_temp[i]

        positive = T > 0

        pet_month = np.zeros_like(
            temp,
            dtype=np.float64
        )

        # Thornthwaite
        pet_month[positive] = (
            16.0
            * (
                10.0
                * T[positive]
                / safe_I[positive]
            ) ** a[positive]
        )

        # -----------------------------------------------------
        # Auringon deklinaatio
        # -----------------------------------------------------

        declination = np.radians(
            -23.44
            * np.cos(
                np.radians(
                    (360.0 / 365.0)
                    * (month_day[i] + 10)
                )
            )
        )

        # -----------------------------------------------------
        # Auringonlaskukulma
        # -----------------------------------------------------

        x = (
            -np.tan(phi)
            * np.tan(declination)
        )

        # Polar day/night
        polar_day = x < -1
        polar_night = x > 1

        x_clipped = np.clip(
            x,
            -1.0,
            1.0
        )

        sunset_angle = np.arccos(
            x_clipped
        )

        # -----------------------------------------------------
        # Päivän pituus
        # -----------------------------------------------------

        daylight_hours = (
            24.0
            / np.pi
            * sunset_angle
        )

        # 24 h polar day
        daylight_hours = np.where(
            polar_day,
            24.0,
            daylight_hours
        )

        # 0 h polar night
        daylight_hours = np.where(
            polar_night,
            0.0,
            daylight_hours
        )

        # -----------------------------------------------------
        # Päivänpituuskorjaus
        # -----------------------------------------------------

        day_length_factor = (
            daylight_hours / 12.0
        )

        # Kuukauden pituuskorjaus
        month_factor = (
            days[i] / 30.0
        )

        pet_month *= (
            day_length_factor
            * month_factor
        )

        pet_annual += pet_month

    # =========================================================
    # Landmask
    # =========================================================

    pet_annual = np.where(
        landmask == 1,
        pet_annual,
        np.nan
    )

    return pet_annual

import numpy as np
from scipy.ndimage import map_coordinates


def laske_sadevarjo(
    relief,
    tuuli_x,
    tuuli_y,
    tuuli_z=None,
    solukoko=1000.0,
    max_etaisyys=200_000.0,
    askel=2000.0,
    kulma_asteikko=8.0,
    etaisyys_asteikko=100_000.0,
    varjon_voimakkuus=0.85,
    tuulen_nopeus_min=0.1,
):
    """
    Laskee topografisen sadevarjon tuulivektorikentästä.

    Parameters
    ----------
    relief : 2D ndarray
        Maaston korkeus metreinä merenpinnasta.

    tuuli_x : 2D ndarray
        Tuulen itä-länsisuuntainen komponentti.
        +x = tuuli kohti rasterin +x-suuntaa.

    tuuli_y : 2D ndarray
        Tuulen pohjois-eteläsuuntainen komponentti.
        +y = tuuli kohti rasterin +y-suuntaa.

    tuuli_z : 2D ndarray, optional
        Tuulen pystysuuntainen komponentti.

    solukoko : float
        Rasterisolun koko metreinä.

    max_etaisyys : float
        Kuinka kauas tuulen yläpuolelle tarkastetaan.

    askel : float
        Ray marching -askel metreinä.

    kulma_asteikko : float
        Kuinka nopeasti sadevarjo voimistuu
        maaston korkeuskulman kasvaessa.

    etaisyys_asteikko : float
        Kuinka nopeasti sadevarjo heikkenee etäisyyden kasvaessa.

    varjon_voimakkuus : float
        Maksimivaikutus 0...1.

    tuulen_nopeus_min : float
        Tätä pienemmillä tuulilla sadevarjoa ei lasketa.

    Returns
    -------
    sadevarjo : ndarray
        Kerroin 0...1.

        1.0 = ei sadevarjoa
        0.0 = erittäin voimakas sadevarjo

    nousu : ndarray
        Maaston aiheuttama tehokas nousukulma asteina.

    """

    relief = np.asarray(relief, dtype=float)
    tuuli_x = np.asarray(tuuli_x, dtype=float)
    tuuli_y = np.asarray(tuuli_y, dtype=float)

    if tuuli_z is None:
        tuuli_z = np.zeros_like(relief)
    else:
        tuuli_z = np.asarray(tuuli_z, dtype=float)

    if relief.shape != tuuli_x.shape:
        raise ValueError("relief ja tuuli_x eivät ole saman kokoisia")

    if relief.shape != tuuli_y.shape:
        raise ValueError("relief ja tuuli_y eivät ole saman kokoisia")

    if relief.shape != tuuli_z.shape:
        raise ValueError("relief ja tuuli_z eivät ole saman kokoisia")

    if solukoko <= 0:
        raise ValueError("solukoko pitää olla > 0")

    if askel <= 0:
        raise ValueError("askel pitää olla > 0")

    # ------------------------------------------------------------
    # 1. Tuulen vaakasuuntainen nopeus
    # ------------------------------------------------------------

    tuuli_vaaka = np.hypot(
        tuuli_x,
        tuuli_y
    )

    # Tuulen yksikkövektori
    ux = np.divide(
        tuuli_x,
        tuuli_vaaka,
        out=np.zeros_like(tuuli_x),
        where=tuuli_vaaka > tuulen_nopeus_min
    )

    uy = np.divide(
        tuuli_y,
        tuuli_vaaka,
        out=np.zeros_like(tuuli_y),
        where=tuuli_vaaka > tuulen_nopeus_min
    )

    # ------------------------------------------------------------
    # 2. Tuulen pystysuuntainen komponentti
    #
    # Tätä ei käytetä suoraan vuoren geometriseen varjoon,
    # vaan se vaikuttaa siihen, kuinka voimakas varjo on.
    # ------------------------------------------------------------

    tuulen_kulma = np.degrees(
        np.arctan2(
            tuuli_z,
            np.maximum(tuuli_vaaka, 1e-12)
        )
    )

    # ------------------------------------------------------------
    # 3. Alustukset
    # ------------------------------------------------------------

    max_kulma = np.full_like(
        relief,
        -90.0,
        dtype=float
    )

    max_kulma_etaisyys = np.zeros_like(
        relief,
        dtype=float
    )

    # ------------------------------------------------------------
    # 4. Kuljetaan tuulen VASTAISEEN suuntaan
    #
    # Jos tuuli kulkee:
    #
    #       ------>
    #
    # tarkastellaan:
    #
    #       <------
    #
    # koska siellä sijaitsevat tuulen yläpuoliset vuoret.
    # ------------------------------------------------------------

    askeleet = int(
        max_etaisyys / askel
    )

    rivit, sarakkeet = relief.shape

    yy, xx = np.indices(
        relief.shape,
        dtype=float
    )

    kelvollinen_tuuli = (
        tuuli_vaaka > tuulen_nopeus_min
    )

    for i in range(1, askeleet + 1):

        etaisyys = i * askel

        # ----------------------------------------
        # Piste tuulen yläpuolella
        # ----------------------------------------

        sample_x = (
            xx -
            ux * etaisyys / solukoko
        )

        sample_y = (
            yy -
            uy * etaisyys / solukoko
        )

        # ----------------------------------------
        # Tarkistetaan kartan rajat
        # ----------------------------------------

        sisalla = (
            (sample_x >= 0) &
            (sample_x <= sarakkeet - 1) &
            (sample_y >= 0) &
            (sample_y <= rivit - 1) &
            kelvollinen_tuuli
        )

        # ----------------------------------------
        # Bilineaarinen interpolointi maastosta
        # ----------------------------------------

        sample_korkeus = map_coordinates(
            relief,
            [
                sample_y.ravel(),
                sample_x.ravel()
            ],
            order=1,
            mode="nearest"
        ).reshape(relief.shape)

        # ----------------------------------------
        # Korkeusero
        # ----------------------------------------

        korkeusero = (
            sample_korkeus -
            relief
        )

        # ----------------------------------------
        # Esteen korkeuskulma
        # ----------------------------------------

        kulma = np.degrees(
            np.arctan2(
                korkeusero,
                etaisyys
            )
        )

        kulma[~sisalla] = -90.0

        # ----------------------------------------
        # Etsitään suurin horisontin ylittävä kulma
        # ----------------------------------------

        parempi = kulma > max_kulma

        max_kulma[parempi] = kulma[parempi]

        max_kulma_etaisyys[parempi] = etaisyys

    # ------------------------------------------------------------
    # 5. Vain horisontin yläpuolella oleva maasto kiinnostaa
    # ------------------------------------------------------------

    nousu = np.maximum(
        max_kulma,
        0.0
    )

    # ------------------------------------------------------------
    # 6. Muutetaan nousukulma varjon voimakkuudeksi
    #
    # 0°      -> ei varjoa
    # 5°      -> jonkin verran
    # 15°     -> voimakas
    # 30°+    -> erittäin voimakas
    # ------------------------------------------------------------

    kulmavaikutus = (
        1.0 -
        np.exp(
            -nousu /
            kulma_asteikko
        )
    )

    # ------------------------------------------------------------
    # 7. Etäisyysvaimennus
    #
    # Vuoren välitön takapuoli saa voimakkaamman varjon.
    # Kauempana varjo alkaa täyttyä.
    # ------------------------------------------------------------

    etaisyysvaikutus = np.exp(
        -max_kulma_etaisyys /
        etaisyys_asteikko
    )

    # ------------------------------------------------------------
    # 8. Tuulen pystysuuntainen liike
    #
    # Positiivinen tuuli_z tarkoittaa nousevaa ilmaa.
    # Nouseva ilma voi kasvattaa kosteuden tiivistymistä,
    # joten pienennämme tällöin varjon voimakkuutta.
    #
    # Laskeva ilma tekee varjosta hieman voimakkaamman.
    # ------------------------------------------------------------

    z_vaikutus = np.clip(
        1.0 -
        tuulen_kulma / 20.0,
        0.5,
        1.5
    )

    # ------------------------------------------------------------
    # 9. Lopullinen varjon voimakkuus
    # ------------------------------------------------------------

    varjon_voimakkuus_kentta = (
        varjon_voimakkuus *
        kulmavaikutus *
        etaisyysvaikutus *
        z_vaikutus
    )

    # ------------------------------------------------------------
    # 10. Muutetaan voimakkuus kertoimeksi
    #
    # 1.0 = normaali sade
    # 0.5 = puolet
    # 0.1 = hyvin kuiva
    # ------------------------------------------------------------

    sadevarjo = (
        1.0 -
        varjon_voimakkuus_kentta
    )

    sadevarjo = np.clip(
        sadevarjo,
        0.05,
        1.0
    )

    # Heikkotuulisilla alueilla ei tehdä sadevarjoa
    sadevarjo[~kelvollinen_tuuli] = 1.0

    return sadevarjo, nousu

def laske_npp_miami(lampotila_matriisi, sade_matriisi, nodata_arvo=None):
    """
    Laskee nettoprimaarituotannon (NPP) Miamin mallilla (Lieth, 1975).
    
    Parametrit:
    - lampotila_matriisi: numpy.ndarray (Vuoden keskilämpötila °C)
    - sade_matriisi: numpy.ndarray (Vuotuinen sademäärä mm)
    - nodata_arvo: Alkuperäinen NoData-arvo (korvataan laskennan ajaksi NaN-arvolla)
    
    Palauttaa:
    - npp: numpy.ndarray (NPP yksikössä g/m²/vuosi, kuiva-aineena. NoData on -1)
    """
    # Kopioidaan matriisit, jotta alkuperäistä dataa ei muuteta
    t = lampotila_matriisi.copy().astype(float)
    p = sade_matriisi.copy().astype(float)
    
    # NoData-arvojen käsittely
    if nodata_arvo is not None:
        t[t == nodata_arvo] = np.nan
        p[p == nodata_arvo] = np.nan
        
    # 1. NPP:n laskenta lämpötilan perusteella
    # NPP_t = 3000 / (1 + exp(1.315 - 0.119 * T))
    npp_t = 3000 / (1 + np.exp(1.315 - 0.119 * t))
    
    # 2. NPP:n laskenta sademäärän perusteella
    # NPP_p = 3000 * (1 - exp(-0.000664 * P))
    npp_p = 3000 * (1 - np.exp(-0.000664 * p))
    
    # 3. Minimitekijän soveltaminen (Liebigin minimilaki)
    # Valitaan jokaiselle pikselille pienempi arvo lämpötilan ja sateen rajoitteista
    npp = np.minimum(npp_t, npp_p)
    
    # Palautetaan NoData-arvot takaisin lukuna -1
    npp[np.isnan(t) | np.isnan(p)] = -1
    
    return npp



def laske_mannerten_keski_npp(npp_matriisi, landmask_matriisi, height, width):
    """
    Laskee mantereiden pinta-alapainotetun keskimääräisen NPP:n 
    käyttäen leveysasteiden kosinipainotusta.
    
    Parametrit:
    - npp_matriisi: numpy.ndarray (NPP-arvot)
    - landmask_matriisi: numpy.ndarray (1 = manner, 0 = meri/vesi)
    - height: rasterin korkeus (rivien määrä)
    - width: rasterin leveys (sarakkeiden määrä)
    
    Palauttaa:
    - painotettu_keskiarvo: float (Mantereiden keskimääräinen NPP)
    """
    # 1. Luodaan leveysastevektori välille 90°N ... -90°S rasterin korkeuden mukaan
    leveysasteet = np.linspace(90, -90, height)
    
    # 2. Lasketaan kosinipainot (muutetaan asteet ensin radiaaneiksi)
    kosini_painot = np.cos(np.radians(leveysasteet))
    
    # 3. Laajennetaan 1D-painovektori 2D-matriisiksi (height, width)
    # np.newaxis lisää akselin, ja np.repeat monistaa sen leveyden verran
    paino_matriisi = np.repeat(kosini_painot[:, np.newaxis], width, axis=1)
    
    # 4. Luodaan maski, joka hyväksyy vain mannerpikselit ja poistaa NoData-arvot (esim. -1 tai NaN)
    # landmask_matriisi == 1 tarkoittaa maata
    validi_maski = (landmask_matriisi == 1) & (npp_matriisi >= 0) & (~np.isnan(npp_matriisi))
    
    # Tarkistetaan, että maapikseleitä löytyy virheiden välttämiseksi
    if not np.any(validi_maski):
        print("Varoitus: Hyväksyttäviä mannerpikseleitä ei löytynyt!")
        return 0.0
        
    # 5. Lasketaan painotettu summa ja painojen kokonaissumma vain hyväksytyiltä alueilta
    painotettu_summa = np.sum(npp_matriisi[validi_maski] * paino_matriisi[validi_maski])
    painojen_summa = np.sum(paino_matriisi[validi_maski])
    
    # 6. Lopullinen pinta-alakorjattu keskiarvo
    painotettu_keskiarvo = painotettu_summa / painojen_summa
    
    return float(painotettu_keskiarvo)


def laske_human_habitability_index(temp, rain):
    """
    Laskee ihmisen asuttavuusindeksin (0 - 100%) perustuen
    Human Climate Niche -malliin (Lämpötila ja Sademäärä).
    """
    # --- 1. LÄMPÖTILAN SOPIVUUS (Kaksi huippua: Lauhkea ~13°C ja Trooppinen ~22°C)
    # Lauhkean vyöhykkeen huippu
    temp_temperate = np.exp(-((temp - 13) ** 2) / (2 * 4 ** 2))
    # Trooppisen vyöhykkeen huippu
    temp_tropical = np.exp(-((temp - 22) ** 2) / (2 * 3 ** 2))
    
    # Yhdistetään lämpötila-optimit (otetaan maksimi tai painotettu summa)
    temp_score = np.maximum(temp_temperate, temp_tropical * 0.8)
    
    # Ehdoton biologinen raja: jos keskilämpötila on yli 29°C tai alle -5°C, asuttavuus romahtaa
    temp_score = np.where((temp > 29) | (temp < -5), temp_score * 0.1, temp_score)

    # --- 2. SADEMÄÄRÄN SOPIVUUS (Ihanne 600mm - 1500mm maataloudelle)
    # Jos sademäärä on alle 400mm (aavikko), arvo putoaa nollaa kohti rajusti
    rain_score = np.exp(-((rain - 1000) ** 2) / (2 * 450 ** 2))
    
    # Korjataan ääripäät: liian kuiva (aavikko) tai liian märkä (jatkuva tulva/suo)
    rain_score = np.where(rain < 300, rain_score * 0.2, rain_score)
    
    # --- 3. LOPULLINEN INDEKSI
    # Kertolasku varmistaa, että jos toinen tekijä on 0 (esim. kuuma aavikko), indeksi on 0
    habitability = temp_score * rain_score
    
    # Skaalataan välille 0 - 100
    return np.clip(habitability * 100, 0, 100)

## älykkään lajin syntypaikka
def laske_synnyinsija_todennakoisyys(relief, temp, rain):
    """
    Laskee älykkään lajin syntypaikan todennäköisyyden (0.0 - 1.0)
    perustuen topografiaan, lämpötilaan ja sademäärään.
    """
    
    # 1. LÄMPÖTILA-ANALYYSINI (Ihanne 18°C, hajonta 8°C)
    # Mitä lähempänä 18 astetta, sitä korkeampi arvo
    temp_score = np.exp(-((temp - 18) ** 2) / (2 * 8 ** 2))
    
    # 2. SADEMÄÄRÄ-ANALYYSI (Ihanne 1200mm, hajonta 400mm)
    rain_score = np.exp(-((rain - 1200) ** 2) / (2 * 400 ** 2))
    
    # 3. TOPOGRAFIA-ANALYYSI (Ihanne: Kumpuileva kukkulamaasto, esim. 200-600m)
    # Tasangot (0m) ja vuoret (>1500m) saavat matalammat pisteet
    topo_score = np.exp(-((relief - 400) ** 2) / (2 * 300 ** 2))
    
    # YHDISTETÄÄN TEKIJÄT (Kertolasku varmistaa, että jos jokin arvo on nolla, 
    # kokonaistodennäköisyys putoaa nollaan – esim. kiehuva vesi tuhoaa mahdollisuudet)
    synty_todennakoisyys = temp_score * rain_score * topo_score
    #mannerkoot, mannerjarjestys=laske_mantereet(relief, planet_radius=planet_radius)
    mannerkoot, mannerjarjestys=laske_mantereet(relief-120, planet_radius=planet_radius) ## ICE AGE! taken account for spreading
    mannerkoot=normalize(mannerkoot)
    synty_todennakoisyys=synty_todennakoisyys*mannerkoot
    return synty_todennakoisyys


def laske_evoluutiopaine(relief, perus_temp, perus_rain, vuosisadat=100):
    """
    Laskee alueet, joissa ympäristön muutos (ilmaston dynaamisuus) 
    luo voimakkaimman paineen älyn kehittymiselle savannivyöhykkeellä.
    """
    shape = relief.shape
    muutos_matriisi = np.zeros(shape)
    
    # Simuloidaan ilmaston syklejä (esim. jääkaudet, kuivat kaudet) ajan yli
    np.random.seed(123)
    for t in range(vuosisadat):
        # Ilmasto heilahtelee globaalisti ajan funktiona
        globaali_sadeheilahtelu = np.sin(t / 5.0) * 150  # +/- 150mm sadetta
        globaali_lampoheilahtelu = np.cos(t / 7.0) * 2.0  # +/- 2 astetta
        
        # Tämän ajanhetken sääkartat
        tämän_hetken_rain = perus_rain + globaali_sadeheilahtelu + np.random.randn(*shape) * 20
        tämän_hetken_temp = perus_temp + globaali_lampoheilahtelu
        
        # Savannin kriittinen kynnys: sademäärä 500mm - 1000mm.
        # Jos ollaan tällä rajalla, jokainen heilahtelu muuttaa ympäristöä rajusti.
        on_savannia = (tämän_hetken_rain >= 500) & (tämän_hetken_rain <= 1000)
        on_leuto = (tämän_hetken_temp >= 12) & (tämän_hetken_temp <= 26)
        
        # Jos alue on dynaamisella vyöhykkeellä, lisätään pisteitä
        muutos_matriisi += (on_savannia & on_leuto).astype(float)
        
    # Lasketaan alueen jyrkkyys (gradientti). Kumpuileva maasto antaa suojaa 
    # ilmastonmuutokselta (mikroilmastot), mikä auttaa lajia pysymään hengissä muutoksen yli.
    dy, dx = np.gradient(relief)
    jyrkkyys = np.sqrt(dx**2 + dy**2)
    # Suositaan kukkuloita (gradientti > 5 ja < 30), ei tasankoja tai pystysuoria seinämiä
    maaston_suoja = np.exp(-((jyrkkyys - 15) ** 2) / (2 * 10 ** 2))
    
    # Lopullinen indeksi: Korkeimmat pisteet saavat alueet, joissa ympäristö 
    # muuttui useimmin SAST-vyöhykkeellä, mutta maasto tarjosi selviytymispaikkoja.
    evoluutiopaine = muutos_matriisi * maaston_suoja
    
    # Normalisoidaan välille 0 - 100
    evoluutiopaine = (evoluutiopaine / np.max(evoluutiopaine)) * 100
    return evoluutiopaine


def laske_sivilisaatiopisteet(sade, lampo, joki, meri, vuori, relief):
    """
    Arvioi varhaisen / primaarin sivilisaation syntypotentiaalia.

    Kaikki syötteet ovat NumPy-taulukoita, joiden tulee olla samanmuotoisia.

    Palauttaa:
        NumPy-taulukon välillä 0...1.

    Ajatus:
        - Lämmin ilmasto suosii sivilisaation syntyä.
        - Kuiva tai puolikuiva ympäristö + joki on erittäin hyvä yhdistelmä:
          kasteluviljely on mahdollista ja siitä syntyy painetta organisoitua.
        - Joki on tärkein yksittäinen tekijä.
        - Meri ja vuoret lisäävät kaupankäynnin, kulkureittien ja resurssien arvoa.
    """

    # ---------------------------------------------------------
    # 1. LÄMPÖTILA
    # ---------------------------------------------------------
    # Optimi noin 18 °C.
    # Liukuva Gauss-tyyppinen funktio.
    #
    # 18 °C -> 1.0
    # 12 °C -> ~0.61
    # 25 °C -> ~0.61
    # 30 °C -> ~0.24
    #
    # Lämpö on tärkeä, mutta ei tee liian jyrkkää rajaa.

    pisteet_lampo = np.exp(-((lampo - 18.0) / 9.0) ** 2)
    #plt.imshow(pisteet_lampo)
    #plt.show()

    # ---------------------------------------------------------
    # 2. SADEMÄÄRÄ
    # ---------------------------------------------------------
    # Primaarin sivilisaation kannalta emme halua yksinkertaisesti
    # "mahdollisimman paljon sadetta".
    #
    # Kuiva / puolikuiva ympäristö on kiinnostava, JOS siellä on joki.
    #
    # Paras alue tässä mallissa:
    # noin 200–700 mm/vuosi.
    #
    # Liian kuiva -> maatalous vaikeutuu.
    # Liian märkä -> kastelun synnyttämä paine pienenee.

    sade_kuivuus = np.exp(-((sade - 400.0) / 350.0) ** 2)
    sade_kuivuus=normalize(sade_kuivuus)
    #plt.imshow(sade_kuivuus)
    #plt.show()
    # ---------------------------------------------------------
    # 3. JOKI
    # ---------------------------------------------------------
    # Etäisyys jokeen on erittäin tärkeä.
    #
    # 0 km -> 1
    # 5 km -> edelleen erittäin hyvä
    # 20 km -> kohtuullinen
    # 40+ km -> heikko
    #
    # Käytetään pehmeää eksponentiaalista laskua.

    pisteet_joki = np.exp(-joki / 12.0)
    pisteet_joki=normalize(pisteet_joki)
    #plt.imshow(pisteet_joki)
    #plt.show()
    # ---------------------------------------------------------
    # 4. KASTELUVILJELYN POTENTIAALI
    # ---------------------------------------------------------
    # Tämä on mallin tärkein uusi osa.
    #
    # Pelkkä kuivuus ei ole hyvä.
    # Pelkkä joki ei ole hyvä.
    #
    # KUIVA + JOKI = erittäin hyvä.
    #
    # Eli sademäärä ja joki ovat vuorovaikutuksessa.

    kastelupotentiaali = (
        pisteet_joki
        * np.exp(-((sade - 300.0) / 450.0) ** 2)
    )
    kastelupotentiaali = normalize(kastelupotentiaali)
    #plt.imshow(kastelupotentiaali)
    #plt.show()
    # ---------------------------------------------------------
    # 5. MERI
    # ---------------------------------------------------------
    # Meri on kaupalle hyödyllinen, mutta aivan rantaviivassa
    # ei välttämättä ole optimaalinen paikka.
    #
    # Paras esimerkiksi noin 10–100 km merestä.
    #
    # Tämä on tarkoituksella melko heikko paino.

    pisteet_meri = (
        np.exp(-meri / 100.0)
        * (1.0 - np.exp(-meri / 8.0))
    )

    # Normalisoidaan niin, että maksimi on ~1.
    pisteet_meri =normalize(pisteet_meri)
    #if np.size(pisteet_meri) else 1.0
    #plt.imshow(pisteet_meri)
    #plt.show()

    # ---------------------------------------------------------
    # 6. VUORET
    # ---------------------------------------------------------
    # Vuoret ovat kiinnostavia:
    # - luonnonresurssit
    # - kulkureittien hallinta
    # - ilmaston / jokien muodostuminen
    # - kauppa
    #
    # Mutta aivan vuoren juurella ei välttämättä ole paras
    # viljelyalue.
    #
    # Optimi noin 20–100 km.

    #pisteet_vuori = (
    #    np.exp(-((vuori - 45.0) / 70.0) ** 2)
    #)
    pisteet_vuori = (
        np.exp(-((vuori - 100) / 700.0) ** 2)
    )
    pisteet_relief = (
        np.exp(-relief/200)
    )

    pisteet_vuori=normalize(pisteet_vuori*pisteet_relief)

    #plt.imshow(pisteet_vuori)
    #plt.show()
    # ---------------------------------------------------------
    # 7. KAUPANKÄYNTIYMPÄRISTÖ
    # ---------------------------------------------------------
    # Meri + vuoret muodostavat yhdessä hieman vahvemman
    # kaupankäynti-/strategiaedun.
    #
    # Ei tehdä tästä kuitenkaan liian dominoivaa.

    kauppapotentiaali = (
        0.55 * pisteet_meri *
        0.45 * pisteet_vuori
    )
    kauppapotentiaali  = normalize(kauppapotentiaali )    
    #plt.imshow(kauppapotentiaali)
    #plt.show()
    # ---------------------------------------------------------
    # 8. LOPULLINEN PISTEYTYS
    # ---------------------------------------------------------
    #
    # Painot:
    #
    # 30 % kasteluviljelyn potentiaali
    # 25 % lämpötila
    # 20 % joki
    # 15 % kauppapotentiaali
    # 10 % ympäristön sopiva kuivuus
    #
    # Kastelu + joki ovat tarkoituksella tärkeimmät.

    kokonaispisteet = (
        (normalize(kastelupotentiaali) * 0.30) *
        (normalize(pisteet_lampo)      * 0.25) *1
        #(normalize(pisteet_joki)       * 0.20) 
        #(normalize(kauppapotentiaali)  * 0.15) *
        #(normalize(sade_kuivuus)       * 0.609
    )
    mannerkoot, mannerjarjestys=laske_mantereet(relief-120, planet_radius=planet_radius) ## ICE AGE! taken account for spreading
    mannerkoot=normalize(mannerkoot)

    kokonaispisteet=normalize(kokonaispisteet*(1/(mannerjarjestys*mannerjarjestys)))
   
    #plt.imshow(kokonaispisteet, cmap="rainbow")
    #plt.show()    
    # ---------------------------------------------------------
    # 9. TURVALLINEN RAJAUS
    # ---------------------------------------------------------

    return np.clip(kokonaispisteet, 0.0, 1.0)


def hunter_gatherer_suitability(
    elevation,
    precipitation,
    temperature,
    npp,
    twi,
    distance_to_water,
    tpi
):
    """
    Metsästäjä-keräilijäkulttuurin ympäristöllinen soveltuvuus 0–100.

    Korkea arvo tarkoittaa ympäristöä, jossa:
    - primäärituotanto on korkea
    - suurriistan potentiaali on hyvä
    - vettä on lähellä
    - ilmasto ei ole liian rajoittava
    - maasto on kulkukelpoista

    Kaikki inputit ovat numpy-arrayta ja niiden tulee olla
    samassa rasteriruudukossa.
    """

    # =========================================================
    # 1. NPP
    # =========================================================
    # NPP on tärkein ravintoverkon tuotannon proxy.
    #
    # Käytetään log-muunnosta, jotta erittäin korkea NPP
    # ei hallitse kaikkea.

    npp_positive = np.maximum(npp, 0)

    npp_log = np.log1p(npp_positive)

    npp_min = np.nanpercentile(npp_log, 2)
    npp_max = np.nanpercentile(npp_log, 98)

    npp_score = (
        (npp_log - npp_min) /
        (npp_max - npp_min)
    ) * 100

    npp_score = np.clip(npp_score, 0, 100)


    # =========================================================
    # 2. VESI
    # =========================================================

    # Etäisyys metreinä.
    # Lähellä vettä erittäin hyvä.
    #
    # Eksponentiaalinen lasku on tässä järkevämpi kuin
    # jyrkät luokat.

    water_score = 100 * np.exp(
        -distance_to_water / 3000
    )

    water_score = np.clip(water_score, 0, 100)


    # =========================================================
    # 3. TWI
    # =========================================================
    #
    # TWI:n suuri arvo tarkoittaa potentiaalisesti kosteaa
    # ja tuottavaa maastonkohtaa.
    #
    # Ei kuitenkaan tehdä siitä liian dominoivaa.

    twi_min = np.nanpercentile(twi, 5)
    twi_max = np.nanpercentile(twi, 95)

    twi_score = (
        (twi - twi_min) /
        (twi_max - twi_min)
    ) * 100

    twi_score = np.clip(twi_score, 0, 100)


    # =========================================================
    # 4. LÄMPÖTILA
    # =========================================================

    temp_score = np.zeros_like(
        temperature,
        dtype=float
    )

    # erittäin kylmä
    temp_score[temperature < -10] = 10

    temp_score[
        (temperature >= -10) &
        (temperature < 0)
    ] = 40

    temp_score[
        (temperature >= 0) &
        (temperature < 5)
    ] = 65

    temp_score[
        (temperature >= 5) &
        (temperature <= 25)
    ] = 100

    temp_score[
        (temperature > 25) &
        (temperature <= 30)
    ] = 90

    temp_score[
        (temperature > 30) &
        (temperature <= 35)
    ] = 60

    temp_score[
        temperature > 35
    ] = 30


    # =========================================================
    # 5. SADE
    # =========================================================

    rain_score = np.zeros_like(
        precipitation,
        dtype=float
    )

    rain_score[precipitation < 100] = 20

    rain_score[
        (precipitation >= 100) &
        (precipitation < 200)
    ] = 40

    rain_score[
        (precipitation >= 200) &
        (precipitation < 400)
    ] = 75

    rain_score[
        (precipitation >= 400) &
        (precipitation <= 1000)
    ] = 100

    rain_score[
        (precipitation > 1000) &
        (precipitation <= 2000)
    ] = 85

    rain_score[precipitation > 2000] = 70


    # =========================================================
    # 6. KORKEUS
    # =========================================================

    elevation_score = np.ones_like(
        elevation,
        dtype=float
    ) * 100

    elevation_score[
        (elevation > 2000) &
        (elevation <= 3000)
    ] = 70

    elevation_score[
        (elevation > 3000) &
        (elevation <= 4000)
    ] = 40

    elevation_score[elevation > 4000] = 10


    # =========================================================
    # 7. TPI
    # =========================================================
    #
    # TPI:tä ei kannata tässä vaiheessa käyttää suorana
    # "enemmän = parempi" -muuttujana.
    #
    # Tasaiset laaksot ja loivat maastonmuodot ovat yleensä
    # helpompia kulkea kuin äärimmäiset harjanteet.
    #
    # Käytetään vain pienenä bonus/malus-tekijänä.

    tpi_abs = np.abs(tpi)

    tpi_min = np.nanpercentile(tpi_abs, 5)
    tpi_max = np.nanpercentile(tpi_abs, 95)

    ruggedness_penalty = (
        (tpi_abs - tpi_min) /
        (tpi_max - tpi_min)
    )

    ruggedness_penalty = np.clip(
        ruggedness_penalty,
        0,
        1
    )

    tpi_score = 100 - ruggedness_penalty * 30


    # =========================================================
    # 8. YHDISTÄ
    # =========================================================

    suitability = (
        0.40 * npp_score +
        0.20 * water_score +
        0.10 * twi_score +
        0.10 * temp_score +
        0.05 * rain_score +
        0.05 * elevation_score +
        0.10 * tpi_score
    )

    return np.clip(suitability, 0, 100)











BIOMIT = {
    "MERI": 0,
    "AAVIKKO": 1,
    "SAVANNI_RUOHOKKO": 2,
    "SADEMETSÄ": 3,
    "LAUHKEA_METSÄ": 4,
    "HAVUMETSÄ": 5,
    "TUNDRA": 6,
    "IKIJÄÄ": 7,
}


def _normalize_percent(x, low=2, high=98):
    """Muuntaa rasterin 0–100 asteikolle."""
    lo = np.nanpercentile(x, low)
    hi = np.nanpercentile(x, high)

    if hi == lo:
        return np.zeros_like(x, dtype=float)

    return np.clip((x - lo) / (hi - lo) * 100, 0, 100)


def subsistence_suitability(
    elevation,
    precipitation,
    temperature,
    npp,
    twi,
    distance_to_water,
    tpi,
    biome,
):
    """
    Laskee neljä ympäristöllistä soveltuvuusindeksiä:

        agriculture
        pastoralism
        nomadism
        hunter_gatherer

    Kaikki palautetaan asteikolla 0–100.

    Input-rasterien tulee olla samassa koordinaatistossa,
    resoluutiossa ja rasterilaajuudessa.

    Parametrit
    ----------
    elevation : np.ndarray
        Korkeus metreinä.

    precipitation : np.ndarray
        Vuotuinen sademäärä mm.

    temperature : np.ndarray
        Keskimääräinen lämpötila °C.

    npp : np.ndarray
        Nettoprimäärituotanto.

    twi : np.ndarray
        Topographic Wetness Index.

    distance_to_water : np.ndarray
        Etäisyys lähimpään merkittävään vesistöön metreinä.

    tpi : np.ndarray
        Topographic Position Index.

    biome : np.ndarray
        Biomin numeerinen ID 0–7.

    Returns
    -------
    dict
        {
            "agriculture": ...,
            "pastoralism": ...,
            "nomadism": ...,
            "hunter_gatherer": ...
        }
    """

    # =========================================================
    # YHTEISET MUUTTUJAT
    # =========================================================

    # ---------------------------------------------------------
    # Lämpötila
    # ---------------------------------------------------------

    temp_ag = np.zeros_like(temperature, dtype=float)

    temp_ag[(temperature >= 5) & (temperature < 7)] = 30
    temp_ag[(temperature >= 7) & (temperature < 9)] = 60
    temp_ag[(temperature >= 9) & (temperature <= 12)] = 100
    temp_ag[(temperature > 12) & (temperature <= 15)] = 80
    temp_ag[temperature > 15] = 50

    # Paimentolaisuus ja metsästys sallivat laajemman
    # lämpötila-alueen.

    temp_past = np.zeros_like(temperature, dtype=float)

    temp_past[temperature < -10] = 10
    temp_past[(temperature >= -10) & (temperature < 0)] = 40
    temp_past[(temperature >= 0) & (temperature < 5)] = 65
    temp_past[(temperature >= 5) & (temperature <= 25)] = 100
    temp_past[(temperature > 25) & (temperature <= 30)] = 85
    temp_past[(temperature > 30) & (temperature <= 35)] = 60
    temp_past[temperature > 35] = 30

    temp_hunter = np.zeros_like(temperature, dtype=float)

    temp_hunter[temperature < -20] = 10
    temp_hunter[(temperature >= -20) & (temperature < -10)] = 40
    temp_hunter[(temperature >= -10) & (temperature < 0)] = 65
    temp_hunter[(temperature >= 0) & (temperature <= 25)] = 100
    temp_hunter[(temperature > 25) & (temperature <= 30)] = 90
    temp_hunter[(temperature > 30) & (temperature <= 35)] = 70
    temp_hunter[temperature > 35] = 40


    # ---------------------------------------------------------
    # Sademäärä
    # ---------------------------------------------------------

    rain_ag = np.zeros_like(precipitation, dtype=float)

    rain_ag[precipitation < 300] = 0
    rain_ag[(precipitation >= 300) & (precipitation < 450)] = 50
    rain_ag[(precipitation >= 450) & (precipitation <= 700)] = 100
    rain_ag[(precipitation > 700) & (precipitation <= 1000)] = 80
    rain_ag[precipitation > 1000] = 50


    rain_past = np.zeros_like(precipitation, dtype=float)

    rain_past[precipitation < 100] = 5
    rain_past[(precipitation >= 100) & (precipitation < 200)] = 25
    rain_past[(precipitation >= 200) & (precipitation < 300)] = 55
    rain_past[(precipitation >= 300) & (precipitation < 500)] = 90
    rain_past[(precipitation >= 500) & (precipitation <= 800)] = 100
    rain_past[(precipitation > 800) & (precipitation <= 1200)] = 85
    rain_past[precipitation > 1200] = 70


    # Nomadismi suosii kuivempaa ympäristöä kuin
    # yleinen pastoralismi.

    rain_nomad = np.zeros_like(precipitation, dtype=float)

    rain_nomad[precipitation < 100] = 10
    rain_nomad[(precipitation >= 100) & (precipitation < 200)] = 45
    rain_nomad[(precipitation >= 200) & (precipitation < 300)] = 80
    rain_nomad[(precipitation >= 300) & (precipitation < 450)] = 100
    rain_nomad[(precipitation >= 450) & (precipitation < 600)] = 90
    rain_nomad[(precipitation >= 600) & (precipitation < 800)] = 60
    rain_nomad[(precipitation >= 800) & (precipitation < 1000)] = 35
    rain_nomad[precipitation >= 1000] = 15


    # Metsästäjä-keräilijälle sateen määrä ei ole itsessään
    # yhtä tärkeä kuin siitä syntyvä biologinen tuotanto.

    rain_hunter = np.zeros_like(precipitation, dtype=float)

    rain_hunter[precipitation < 100] = 15
    rain_hunter[(precipitation >= 100) & (precipitation < 200)] = 40
    rain_hunter[(precipitation >= 200) & (precipitation < 400)] = 75
    rain_hunter[(precipitation >= 400) & (precipitation <= 1000)] = 100
    rain_hunter[(precipitation > 1000) & (precipitation <= 2000)] = 90
    rain_hunter[precipitation > 2000] = 75


    # =========================================================
    # KORKEUS
    # =========================================================

    elevation_score = np.ones_like(elevation, dtype=float) * 100

    elevation_score[
        (elevation > 2000) & (elevation <= 3000)
    ] = 70

    elevation_score[
        (elevation > 3000) & (elevation <= 4000)
    ] = 40

    elevation_score[elevation > 4000] = 10

    elevation_ag = elevation_score.copy()
    elevation_past = elevation_score.copy()
    elevation_nomad = elevation_score.copy()
    elevation_hunter = elevation_score.copy()


    # =========================================================
    # NPP
    # =========================================================

    # Logaritminen muunnos vähentää erittäin tuottavien
    # trooppisten alueiden ylivaltaa.

    npp_log = np.log1p(np.maximum(npp, 0))
    npp_score = _normalize_percent(npp_log)


    # =========================================================
    # VESI
    # =========================================================

    # Etäisyys metreinä.
    # 3 km on tässä karkea skaala.

    water_score = 100 * np.exp(
        -distance_to_water / 3000
    )

    water_score = np.clip(water_score, 0, 100)


    # =========================================================
    # TWI
    # =========================================================

    twi_score = _normalize_percent(twi, 5, 95)


    # =========================================================
    # TPI
    # =========================================================

    # Äärimmäisen suuri positiivinen/negatiivinen TPI
    # tulkitaan vaikeammaksi maastoksi.
    #
    # Tämä on tarkoituksella vain pieni vaikutus.

    tpi_abs = np.abs(tpi)

    tpi_extreme = _normalize_percent(tpi_abs, 5, 95)

    terrain_score = 100 - 0.30 * tpi_extreme


    # =========================================================
    # BIOMI
    # =========================================================

    biome_score = np.zeros_like(
        biome,
        dtype=float
    )

    biome_score[biome == BIOMIT["MERI"]] = 0
    biome_score[biome == BIOMIT["AAVIKKO"]] = 15
    biome_score[biome == BIOMIT["SAVANNI_RUOHOKKO"]] = 100
    biome_score[biome == BIOMIT["SADEMETSÄ"]] = 65
    biome_score[biome == BIOMIT["LAUHKEA_METSÄ"]] = 90
    biome_score[biome == BIOMIT["HAVUMETSÄ"]] = 80
    biome_score[biome == BIOMIT["TUNDRA"]] = 75
    biome_score[biome == BIOMIT["IKIJÄÄ"]] = 0


    # =========================================================
    # 1. SADEVILJELY
    # =========================================================

    agriculture = (
        0.45 * rain_ag +
        0.40 * temp_ag +
        0.10 * elevation_ag +
        0.05 * twi_score
    )

    # Meri ja ikijää pois.
    agriculture[
        (biome == BIOMIT["MERI"]) |
        (biome == BIOMIT["IKIJÄÄ"])
    ] = 0


    # =========================================================
    # 2. PASTORALISMI
    # =========================================================

    pastoralism = (
        0.40 * npp_score +
        0.25 * rain_past +
        0.15 * temp_past +
        0.10 * water_score +
        0.05 * biome_score +
        0.05 * elevation_past
    )

    pastoralism[
        (biome == BIOMIT["MERI"]) |
        (biome == BIOMIT["IKIJÄÄ"])
    ] = 0


    # =========================================================
    # 3. NOMADISMI
    # =========================================================

    nomadism = (
        0.30 * npp_score +
        0.25 * rain_nomad +
        0.15 * temp_past +
        0.15 * water_score +
        0.05 * biome_score +
        0.05 * terrain_score +
        0.05 * elevation_nomad
    )

    nomadism[
        (biome == BIOMIT["MERI"]) |
        (biome == BIOMIT["IKIJÄÄ"])
    ] = 0


    # =========================================================
    # 4. METSÄSTÄJÄ-KERÄILIJÄT
    # =========================================================
    #
    # NPP + biomi ovat tässä tärkeimmät.
    #
    # NPP = paljon energiaa ravintoverkkoon
    # Biomi = minkä tyyppinen eläinyhteisö
    # Vesi = eläinten ja ihmisten saavutettavuus
    #

    hunter_gatherer = (
        0.35 * npp_score +
        0.20 * biome_score +
        0.15 * water_score +
        0.10 * twi_score +
        0.10 * temp_hunter +
        0.05 * terrain_score +
        0.05 * rain_hunter
    )

    hunter_gatherer[
        (biome == BIOMIT["MERI"]) |
        (biome == BIOMIT["IKIJÄÄ"])
    ] = 0


    # =========================================================
    # RAJAA 0–100
    # =========================================================

    agriculture = np.clip(agriculture, 0, 100)
    pastoralism = np.clip(pastoralism, 0, 100)
    nomadism = np.clip(nomadism, 0, 100)
    hunter_gatherer = np.clip(hunter_gatherer, 0, 100)


    return {
        "agriculture": agriculture,
        "pastoralism": pastoralism,
        "nomadism": nomadism,
        "hunter_gatherer": hunter_gatherer,
    }



def lbk_sadeviljely_soveltuvuus(elevation, precipitation, temperature):
    """
    Arvioi LBK-tasoisen sadeviljelyn ilmastollista soveltuvuutta.

    Parametrit
    ----------
    elevation : np.ndarray
        Korkeus metreinä.
    precipitation : np.ndarray
        Vuotuinen sademäärä millimetreinä.
    temperature : np.ndarray
        Vuotuinen keskilämpötila °C.

    Palauttaa
    ----------
    np.ndarray
        Soveltuvuus 0–100.
    """

    # -------------------------
    # Lämpötila
    # -------------------------
    temp_score = np.zeros_like(temperature, dtype=float)

    temp_score[(temperature >= 5) & (temperature < 7)] = 30
    temp_score[(temperature >= 7) & (temperature < 9)] = 60
    temp_score[(temperature >= 9) & (temperature <= 12)] = 100
    temp_score[(temperature > 12) & (temperature <= 15)] = 80
    temp_score[temperature > 15] = 50

    # -------------------------
    # Sademäärä
    # -------------------------
    rain_score = np.zeros_like(precipitation, dtype=float)

    rain_score[(precipitation >= 300) & (precipitation < 450)] = 50
    rain_score[(precipitation >= 450) & (precipitation <= 700)] = 100
    rain_score[(precipitation > 700) & (precipitation <= 1000)] = 80
    rain_score[precipitation > 1000] = 50

    # -------------------------
    # Korkeus
    # -------------------------
    elevation_score = np.ones_like(elevation, dtype=float) * 100

    elevation_score[(elevation > 500) & (elevation <= 1000)] = 70
    elevation_score[(elevation > 1000) & (elevation <= 1500)] = 30
    elevation_score[elevation > 1500] = 0

    # -------------------------
    # Yhdistelmä
    # -------------------------
    suitability = (
        temp_score *
        rain_score *
        elevation_score
    ) / 10000

    # Varmistetaan rajat 0–100
    suitability = np.clip(suitability, 0, 100)

    return suitability


def nomadism_suitability(elevation, precipitation, temperature):
    """
    Arvioi luonnonlaidunnukseen perustuvan nomadisen
    paimentolaisuuden soveltuvuutta.

    Palauttaa arvon 0-100.

    Parameters
    ----------
    elevation : np.ndarray
        Korkeus metreinä.

    precipitation : np.ndarray
        Vuotuinen sademäärä mm.

    temperature : np.ndarray
        Vuotuinen keskilämpötila °C.
    """

    # --------------------------------------------------
    # 1. SADE
    # --------------------------------------------------
    # Optimi noin 300-600 mm.
    # Liian kuiva -> vähän biomassaa.
    # Liian kostea -> maanviljely ja metsäisempi ympäristö
    # tulevat suhteellisesti houkuttelevammiksi.
    
    rain_score = np.zeros_like(precipitation, dtype=float)

    rain_score[precipitation < 100] = 5

    mask = (precipitation >= 100) & (precipitation < 200)
    rain_score[mask] = 20

    mask = (precipitation >= 200) & (precipitation < 300)
    rain_score[mask] = 50

    mask = (precipitation >= 300) & (precipitation < 400)
    rain_score[mask] = 80

    mask = (precipitation >= 400) & (precipitation <= 600)
    rain_score[mask] = 100

    mask = (precipitation > 600) & (precipitation <= 800)
    rain_score[mask] = 80

    mask = (precipitation > 800) & (precipitation <= 1000)
    rain_score[mask] = 50

    mask = (precipitation > 1000) & (precipitation <= 1500)
    rain_score[mask] = 25

    rain_score[precipitation > 1500] = 10


    # --------------------------------------------------
    # 2. LÄMPÖTILA
    # --------------------------------------------------
    # Lammas ja vuohi kestävät melko laajan alueen.
    # Liian kylmä / liian kuuma kuitenkin rajoittaa.

    temp_score = np.zeros_like(temperature, dtype=float)

    temp_score[temperature < -5] = 0

    mask = (temperature >= -5) & (temperature < 0)
    temp_score[mask] = 20

    mask = (temperature >= 0) & (temperature < 5)
    temp_score[mask] = 50

    mask = (temperature >= 5) & (temperature < 10)
    temp_score[mask] = 80

    mask = (temperature >= 10) & (temperature <= 25)
    temp_score[mask] = 100

    mask = (temperature > 25) & (temperature <= 30)
    temp_score[mask] = 80

    mask = (temperature > 30) & (temperature <= 35)
    temp_score[mask] = 50

    temp_score[temperature > 35] = 20


    # --------------------------------------------------
    # 3. KORKEUS
    # --------------------------------------------------
    elevation_score = np.ones_like(elevation, dtype=float) * 100

    mask = (elevation > 1000) & (elevation <= 2000)
    elevation_score[mask] = 80

    mask = (elevation > 2000) & (elevation <= 3000)
    elevation_score[mask] = 50

    elevation_score[elevation > 3000] = 10


    # --------------------------------------------------
    # 4. YHDISTÄ
    # --------------------------------------------------

    suitability = (
        rain_score *
        temp_score *
        elevation_score
    ) / 10000

    return np.clip(suitability, 0, 100)

import numpy as np


def subsistence_suitability_threepars(elevation, precipitation, temperature):
    """
    Laskee kolme soveltuvuutta:
        1. sadeviljely
        2. pastoralismi / laidunnus
        3. nomadinen pastoralismi

    Kaikki palautetaan asteikolla 0–100.

    Parameters
    ----------
    elevation : np.ndarray
        Korkeus metreinä.

    precipitation : np.ndarray
        Vuotuinen sademäärä mm.

    temperature : np.ndarray
        Vuotuinen keskilämpötila °C.

    Returns
    -------
    agriculture, pastoralism, nomadism : np.ndarray
    """

    # ==========================================================
    # 1. SADEVILJELY
    # ==========================================================

    agriculture_rain = np.zeros_like(precipitation, dtype=float)

    agriculture_rain[
        (precipitation >= 300) &
        (precipitation < 450)
    ] = 50

    agriculture_rain[
        (precipitation >= 450) &
        (precipitation <= 700)
    ] = 100

    agriculture_rain[
        (precipitation > 700) &
        (precipitation <= 1000)
    ] = 80

    agriculture_rain[
        (precipitation > 1000)
    ] = 50


    agriculture_temp = np.zeros_like(temperature, dtype=float)

    agriculture_temp[
        (temperature >= 5) &
        (temperature < 7)
    ] = 30

    agriculture_temp[
        (temperature >= 7) &
        (temperature < 9)
    ] = 60

    agriculture_temp[
        (temperature >= 9) &
        (temperature <= 12)
    ] = 100

    agriculture_temp[
        (temperature > 12) &
        (temperature <= 15)
    ] = 80

    agriculture_temp[
        temperature > 15
    ] = 50


    agriculture_elevation = np.ones_like(elevation, dtype=float) * 100

    agriculture_elevation[
        (elevation > 500) &
        (elevation <= 1000)
    ] = 70

    agriculture_elevation[
        (elevation > 1000) &
        (elevation <= 1500)
    ] = 30

    agriculture_elevation[
        elevation > 1500
    ] = 0


    agriculture = (
        agriculture_rain *
        agriculture_temp *
        agriculture_elevation
    ) / 10000


    # ==========================================================
    # 2. PASTORALISMI
    # ==========================================================
    #
    # Laidunnus voi olla mahdollista paljon kuivemmassa
    # ympäristössä kuin sadeviljely.
    #
    # Hyvä laidun:
    #     ~300–800 mm
    #
    # Myös kuivempi alue voi olla käyttökelpoinen,
    # mutta biomassaa syntyy vähemmän.
    #

    pastoral_rain = np.zeros_like(precipitation, dtype=float)

    pastoral_rain[
        precipitation < 100
    ] = 5

    pastoral_rain[
        (precipitation >= 100) &
        (precipitation < 200)
    ] = 25

    pastoral_rain[
        (precipitation >= 200) &
        (precipitation < 300)
    ] = 55

    pastoral_rain[
        (precipitation >= 300) &
        (precipitation < 500)
    ] = 90

    pastoral_rain[
        (precipitation >= 500) &
        (precipitation <= 800)
    ] = 100

    pastoral_rain[
        (precipitation > 800) &
        (precipitation <= 1200)
    ] = 85

    pastoral_rain[
        (precipitation > 1200)
    ] = 70


    # Lampaat/vuohet kestävät melko laajan lämpötila-alueen.
    pastoral_temp = np.zeros_like(temperature, dtype=float)

    pastoral_temp[
        (temperature >= -5) &
        (temperature < 0)
    ] = 30

    pastoral_temp[
        (temperature >= 0) &
        (temperature < 5)
    ] = 60

    pastoral_temp[
        (temperature >= 5) &
        (temperature <= 25)
    ] = 100

    pastoral_temp[
        (temperature > 25) &
        (temperature <= 30)
    ] = 85

    pastoral_temp[
        (temperature > 30) &
        (temperature <= 35)
    ] = 60

    pastoral_temp[
        temperature > 35
    ] = 30


    pastoral_elevation = np.ones_like(elevation, dtype=float) * 100

    pastoral_elevation[
        (elevation > 2000) &
        (elevation <= 3000)
    ] = 70

    pastoral_elevation[
        elevation > 3000
    ] = 40


    pastoralism = (
        pastoral_rain *
        pastoral_temp *
        pastoral_elevation
    ) / 10000


    # ==========================================================
    # 3. NOMADISMI
    # ==========================================================
    #
    # Tärkeä ero:
    #
    # hyvä laidun != hyvä nomadialue
    #
    # Nomadismi hyötyy siitä, että:
    #   - sade ei riitä hyvin viljelyyn
    #   - luonnonlaidunta kuitenkin syntyy
    #   - eläimet voivat liikkua resurssien perässä
    #
    # Siksi optimum on kuivempi kuin pastoralismissa yleisesti.
    #

    nomad_rain = np.zeros_like(precipitation, dtype=float)

    nomad_rain[
        precipitation < 100
    ] = 10

    nomad_rain[
        (precipitation >= 100) &
        (precipitation < 200)
    ] = 45

    nomad_rain[
        (precipitation >= 200) &
        (precipitation < 300)
    ] = 80

    nomad_rain[
        (precipitation >= 300) &
        (precipitation < 450)
    ] = 100

    nomad_rain[
        (precipitation >= 450) &
        (precipitation < 600)
    ] = 90

    nomad_rain[
        (precipitation >= 600) &
        (precipitation < 800)
    ] = 60

    nomad_rain[
        (precipitation >= 800) &
        (precipitation < 1000)
    ] = 35

    nomad_rain[
        precipitation >= 1000
    ] = 15


    # Nomadismi ei vaadi yhtä lämmintä ilmastoa kuin viljely.
    nomad_temp = np.zeros_like(temperature, dtype=float)

    nomad_temp[
        temperature < -10
    ] = 10

    nomad_temp[
        (temperature >= -10) &
        (temperature < 0)
    ] = 40

    nomad_temp[
        (temperature >= 0) &
        (temperature < 5)
    ] = 70

    nomad_temp[
        (temperature >= 5) &
        (temperature <= 25)
    ] = 100

    nomad_temp[
        (temperature > 25) &
        (temperature <= 30)
    ] = 90

    nomad_temp[
        (temperature > 30) &
        (temperature <= 35)
    ] = 60

    nomad_temp[
        temperature > 35
    ] = 30


    nomad_elevation = np.ones_like(elevation, dtype=float) * 100

    nomad_elevation[
        (elevation > 2500) &
        (elevation <= 3500)
    ] = 70

    nomad_elevation[
        elevation > 3500
    ] = 30


    nomadism = (
        nomad_rain *
        nomad_temp *
        nomad_elevation
    ) / 10000


    # Rajataan 0–100
    agriculture = np.clip(agriculture, 0, 100)
    pastoralism = np.clip(pastoralism, 0, 100)
    nomadism = np.clip(nomadism, 0, 100)

    return agriculture, pastoralism, nomadism


import numpy as np


def laske_jaatikko(
    dem,
    temp_degC,
    precip_mm,
    planeetan_sade_km,
    gee_ms,
    num_years=1000,
    sea_level=0.0,
    degree_day_factor=4.0,
):
    """
    Yksinkertainen jäätikkömalli vuosikeskilämpötilan perusteella.

    dem:
        (height, width)
        Koko planeetan korkeusmatriisi metreinä.

    temp_degC:
        (height, width)
        Vuosikeskilämpötila °C.

    precip_mm:
        (height, width)
        Vuosittainen sadanta mm/vuosi.

    planeetan_sade_km:
        Planeetan säde kilometreinä.

    gee_ms:
        Painovoima m/s².

    num_years:
        Kuinka monta vuotta jäätikköä kasvatetaan.

    Palauttaa:
        dict
    """

    dem = np.asarray(dem, dtype=np.float64)
    temp_degC = np.asarray(temp_degC, dtype=np.float64)
    precip_mm = np.asarray(precip_mm, dtype=np.float64)

    # ---------------------------------------------------------
    # Tarkistukset
    # ---------------------------------------------------------

    if temp_degC.shape != dem.shape:
        raise ValueError(
            f"temp_degC pitää olla muodossa {dem.shape}, "
            f"nyt {temp_degC.shape}"
        )

    if precip_mm.shape != dem.shape:
        raise ValueError(
            f"precip_mm pitää olla muodossa {dem.shape}, "
            f"nyt {precip_mm.shape}"
        )

    height, width = dem.shape

    # ---------------------------------------------------------
    # Planeetan säde
    # ---------------------------------------------------------

    R = planeetan_sade_km * 1000.0

    # ---------------------------------------------------------
    # Pikselien pinta-alat pallopinnalla
    # ---------------------------------------------------------

    dlat = np.pi / height
    dlon = 2.0 * np.pi / width

    lat = (
        -np.pi / 2.0
        + (np.arange(height) + 0.5) * dlat
    )

    row_area = (
        R**2
        * dlon
        * (
            np.sin(lat + dlat / 2.0)
            - np.sin(lat - dlat / 2.0)
        )
    )

    pixel_area = row_area[:, None]

    # ---------------------------------------------------------
    # Maa ja meri
    # ---------------------------------------------------------

    land = dem >= sea_level
    ocean = ~land

    ocean_area_m2 = np.sum(
        pixel_area * ocean
    )

    # ---------------------------------------------------------
    # Lumi
    #
    # Yksinkertainen oletus:
    # alle 0 °C -> kaikki sade lunta
    # yli 0 °C  -> kaikki sade vettä
    # ---------------------------------------------------------

    snowfall_mm = np.where(
        temp_degC < 0.0,
        precip_mm,
        0.0
    )

    # ---------------------------------------------------------
    # Sulaminen
    #
    # Vuosikeskilämpötilan perusteella arvioitu.
    #
    # Esimerkiksi:
    # +1 °C -> 1 * 365 * DDF
    # +5 °C -> 5 * 365 * DDF
    # ---------------------------------------------------------

    positive_temp = np.maximum(
        temp_degC,
        0.0
    )

    melt_mm = (
        positive_temp
        * 365.0
        * degree_day_factor
    )

    # ---------------------------------------------------------
    # Vuotuinen massatase
    # ---------------------------------------------------------

    annual_balance_mm = (
        snowfall_mm
        - melt_mm
    )

    # Vain maa-alueelle
    annual_balance_mm = np.where(
        land,
        annual_balance_mm,
        0.0
    )

    # ---------------------------------------------------------
    # Jään paksuus
    # ---------------------------------------------------------

    ice_thickness = np.zeros_like(
        dem,
        dtype=np.float64
    )

    annual_change_m = (
        annual_balance_mm / 1000.0
    )

    # ---------------------------------------------------------
    # Kasvatetaan jäätikköä
    # ---------------------------------------------------------

    for year in range(num_years):

        ice_thickness += annual_change_m

        # Jään paksuus ei voi olla negatiivinen
        ice_thickness = np.maximum(
            ice_thickness,
            0.0
        )

        # Ei jäätä meressä tässä versiossa
        ice_thickness[~land] = 0.0

    # ---------------------------------------------------------
    # Jäätikköalue
    # ---------------------------------------------------------

    glacier_mask = ice_thickness > 0.0

    glacier_area_m2 = np.sum(
        pixel_area * glacier_mask
    )

    # ---------------------------------------------------------
    # Jäätilavuus
    # ---------------------------------------------------------

    ice_volume_m3 = np.sum(
        ice_thickness * pixel_area
    )

    # ---------------------------------------------------------
    # Jään massa
    # ---------------------------------------------------------

    rho_ice = 917.0

    ice_mass_kg = (
        ice_volume_m3
        * rho_ice
    )

    # ---------------------------------------------------------
    # Jään sisältämä vesimäärä
    # ---------------------------------------------------------

    rho_water = 1000.0

    water_volume_m3 = (
        ice_volume_m3
        * rho_ice
        / rho_water
    )

    # ---------------------------------------------------------
    # Merenpinnan lasku
    # ---------------------------------------------------------

    if ocean_area_m2 > 0.0:

        sea_level_change_m = (
            -water_volume_m3
            / ocean_area_m2
        )

    else:

        sea_level_change_m = 0.0

    # ---------------------------------------------------------
    # Tulokset
    # ---------------------------------------------------------

    return {
        "ice_thickness": ice_thickness,
        "glacier_mask": glacier_mask,

        "glacier_area_m2":
            glacier_area_m2,

        "ice_volume_m3":
            ice_volume_m3,

        "ice_mass_kg":
            ice_mass_kg,

        "water_volume_m3":
            water_volume_m3,

        "sea_level_change_m":
            sea_level_change_m,

        "annual_balance_mm":
            annual_balance_mm,

        "ocean_area_m2":
            ocean_area_m2,
    }




import math
import numpy as np


# Stefan–Boltzmannin vakio
SIGMA = 5.670374419e-8

# Maan nykytilan kalibrointi:
# vesihoyry_kerroin = 1.0 tarkoittaa Maa-tyyppistä vesihöyryn vaikutusta.
#
# Tätä arvoa käytetään parametrisoimaan luonnollista kasvihuoneilmiötä.
# Arvo voidaan myöhemmin korvata fysikaalisemmalla H2O/CO2-mallilla.
MAA_GHG_DELTA_T = 31.82


def laske_planeetan_lampotila(
    ecc,
    tilt,
    mvelp,
    S1,
    atmos_co2_ppm,

    # Planeetan ominaisuudet
    air_pressure_atm=1.0,
    planet_radius_re=1.0,
    rotation_period_days=1.0,

    # Pinnan ominaisuudet
    albedo_keski=0.30,

    # Kasvihuoneilmiö
    vesihoyry_kerroin=1.0,
    kasvihuone_kerroin=None,

    # CO2
    co2_viite_ppm=280.0,
    ilmastosensitiivisyys=0.8,

    # Lämmön kuljetus
    lammon_kuljetus_paivantaasaajalta_navoille=3.5,

    # Lämpökapasiteetti
    lampo_kapasiteetti=1.0,

    # Vuodenaikaisvaihtelu
    vuodenaika_kerroin=1.0,
):
    """
    Yhtenäinen planeetan lämpötilamalli.

    Palauttaa:
        - globaalin lämpötilan
        - kasvihuoneettoman lämpötilan
        - luonnollisen kasvihuoneilmiön vaikutuksen
        - CO2:n säteilypakotteen ja Delta-T:n
        - päiväntasaajan ja napojen lämpötilat
        - päiväntasaaja–navat lämpötilaeron
        - lämmönkuljetuksen vaikutuksen
        - vuosittaisen lämpötilavaihtelun päiväntasaajalla
        - vuosittaisen lämpötilavaihtelun navoilla

    Parametrit
    ----------
    vesihoyry_kerroin:
        1.0 = Maata muistuttava vesihöyryvaikutus
        0.0 = ei vesihöyryn kasvihuonevaikutusta
        >1 = kosteampi / voimakkaampi vesihöyryvaikutus

    kasvihuone_kerroin:
        Valinnainen kokonaiskasvihuoneen lisäkerroin.
        None = ei ylimääräistä kerrointa.

    lammon_kuljetus_paivantaasaajalta_navoille:
        Lämmönkuljetuksen voimakkuus.
        Suurempi arvo tasaa päiväntasaajan ja napojen lämpötilaeroa.

    lampo_kapasiteetti:
        Terminen lämpökapasiteetti.
        Suurempi arvo pienentää vuodenaikaisvaihtelua.

    vuodenaika_kerroin:
        Vuodenaikaisvaihtelun säätökerroin.
        1.0 = normaali mallin mukainen vaihtelu.
    """

    # ============================================================
    # 1. TARKISTUKSET
    # ============================================================

    if not 0.0 <= ecc < 1.0:
        raise ValueError("ecc pitää olla välillä 0...1.")

    if not 0.0 <= tilt <= 90.0:
        raise ValueError("tilt pitää olla välillä 0...90 astetta.")

    if S1 <= 0:
        raise ValueError("S1 pitää olla > 0.")

    if atmos_co2_ppm <= 0:
        raise ValueError("CO2-pitoisuuden pitää olla > 0 ppm.")

    if co2_viite_ppm <= 0:
        raise ValueError("CO2-vertailuarvon pitää olla > 0 ppm.")

    if air_pressure_atm <= 0:
        raise ValueError("air_pressure_atm pitää olla > 0.")

    if planet_radius_re <= 0:
        raise ValueError("planet_radius_re pitää olla > 0.")

    if rotation_period_days <= 0:
        raise ValueError("rotation_period_days pitää olla > 0.")

    if lampo_kapasiteetti <= 0:
        raise ValueError("lampo_kapasiteetti pitää olla > 0.")

    if vesihoyry_kerroin < 0:
        raise ValueError("vesihoyry_kerroin ei voi olla negatiivinen.")

    if not 0 <= albedo_keski < 1:
        raise ValueError("albedo pitää olla välillä 0...1.")

    # ============================================================
    # 2. GLOBAALI ABSORBOITUVA AURINKOENERGIA
    # ============================================================

    # Eksentrisyys vaikuttaa vuosittaiseen keskimääräiseen energiaan.
    #
    # 1/sqrt(1-e²) on vuosikeskiarvon approksimaatio.

    vuo_globaali = (
        (S1 / 4.0)
        * (1.0 - albedo_keski)
        / math.sqrt(1.0 - ecc**2)
    )

    # ============================================================
    # 3. KASVIHUONEETON LÄMPÖTILA
    # ============================================================

    T_effective_K = (
        vuo_globaali / SIGMA
    ) ** 0.25

    T_effective_C = (
        T_effective_K - 273.15
    )

    # ============================================================
    # 4. LUONNOLLINEN KASVIHUONEILMIÖ
    # ============================================================

    # 1.0 = Maa
    #
    # Maan luonnollisen kasvihuoneilmiön vaikutus on kalibroitu
    # noin 31.82 °C:een tässä mallissa.
    #
    # Tämä on ERILLINEN asia kuin CO2:n pitoisuuden muutos.

    delta_T_vesihoyry = (
        MAA_GHG_DELTA_T
        * vesihoyry_kerroin
    )

    # Mahdollinen ylimääräinen kokonaiskasvihuoneen säätö.
    if kasvihuone_kerroin is None:
        kasvihuone_kerroin = 1.0

    delta_T_natural_ghg = (
        delta_T_vesihoyry
        * kasvihuone_kerroin
    )

    # ============================================================
    # 5. CO2:N SÄTEILYPAKOTE
    # ============================================================

    # Standardi logaritminen CO2-pakotekaava:
    #
    # dF = 5.35 ln(C/C0)
    #
    # Yksikkö: W/m²

    dF_co2 = (
        5.35
        * math.log(
            (air_pressure_atm*atmos_co2_ppm) / co2_viite_ppm
        )
    )

    # CO2:n aiheuttama lämpötilamuutos.
    #
    # ilmastosensitiivisyys:
    # K / (W/m²)

    delta_T_co2 = (
        ilmastosensitiivisyys
        * dF_co2
    )

    # ============================================================
    # 6. GLOBAALI LÄMPÖTILA
    # ============================================================

    T_global_C = (
        T_effective_C
        + delta_T_natural_ghg
        + delta_T_co2
    )

    # ============================================================
    # 7. ALUEELLINEN AURINKOENERGIA
    # ============================================================

    tilt_rad = math.radians(tilt)
    mvelp_rad = math.radians(mvelp)

    # ------------------------------------------------------------
    # Päiväntasaaja
    # ------------------------------------------------------------

    vuo_korjaus_eq = (
        (2.0 / math.pi)
        * math.cos(tilt_rad)
        +
        (tilt_rad / math.pi)
        * math.sin(tilt_rad)
    )

    vuo_eq = (
        (S1 / math.pi)
        * vuo_korjaus_eq
        / math.sqrt(1.0 - ecc**2)
    )

    # ------------------------------------------------------------
    # Navat
    # ------------------------------------------------------------

    vuo_pole_base = (
        (S1 / math.pi)
        * math.sin(tilt_rad)
        / math.sqrt(1.0 - ecc**2)
    )

    # Perihelin vaikutus pohjoisen ja eteläisen pallonpuoliskon
    # vuotuiseen säteilyyn.

    pohjoinen_epasymmetria = (
        1.0
        + ecc * math.sin(mvelp_rad)
    )

    etelainen_epasymmetria = (
        1.0
        - ecc * math.sin(mvelp_rad)
    )

    vuo_pole_N = (
        vuo_pole_base
        * pohjoinen_epasymmetria
    )

    vuo_pole_S = (
        vuo_pole_base
        * etelainen_epasymmetria
    )

    # ============================================================
    # 8. ALUEELLINEN EFEKTIIVINEN LÄMPÖTILA
    # ============================================================

    def vuo_to_effective_temp(vuo, geometria):
        """
        Alueellisen säteilyn aiheuttama efektiivinen lämpötila.
        Kasvihuoneilmiötä ei vielä lisätä tässä.
        """

        absorboituva_vuo = (
            vuo
            * geometria
            * (1.0 - albedo_keski)
        )

        T_K = (
            absorboituva_vuo / SIGMA
        ) ** 0.25

        return T_K - 273.15

    T_eq_effective = vuo_to_effective_temp(
        vuo_eq,
        0.75
    )

    T_pole_N_effective = vuo_to_effective_temp(
        vuo_pole_N,
        0.85
    )

    T_pole_S_effective = vuo_to_effective_temp(
        vuo_pole_S,
        0.85
    )

    # ============================================================
    # 9. ALUEELLISET POIKKEAMAT GLOBAALISTA
    # ============================================================

    # TÄRKEÄ MUUTOS:
    #
    # Emme lisää kasvihuoneilmiötä uudelleen päiväntasaajan
    # ja napojen lämpötiloihin.
    #
    # Sen sijaan alueellinen säteilymalli kertoo poikkeaman
    # globaalista keskilämpötilasta.

    T_eq_anomalia = (
        T_eq_effective - T_effective_C
    )

    T_pole_N_anomalia = (
        T_pole_N_effective - T_effective_C
    )

    T_pole_S_anomalia = (
        T_pole_S_effective - T_effective_C
    )

    # ============================================================
    # 10. GLOBAALIIN LÄMPÖTILAAN LISÄTTÄVÄT ALUEELLISET ARVOT
    # ============================================================

    T_eq_raw = (
        T_global_C
        + T_eq_anomalia
    )

    T_pole_N_raw = (
        T_global_C
        + T_pole_N_anomalia
    )

    T_pole_S_raw = (
        T_global_C
        + T_pole_S_anomalia
    )

    T_poles_raw = (
        T_pole_N_raw
        + T_pole_S_raw
    ) / 2.0

    delta_T_eq_poles_raw = (
        T_eq_raw
        - T_poles_raw
    )

    # ============================================================
    # 11. LÄMMÖNKULJETUS
    # ============================================================

    # Ilmakehä ja meri tasoittavat lämpötilaeroa.
    #
    # Suurempi ilmanpaine:
    #     tehokkaampi kuljetus
    #
    # Pidempi vuorokausi / suurempi planeetta:
    #     muuttaa parametrisoitua kuljetusta.

    kuljetuskerroin = (
        lammon_kuljetus_paivantaasaajalta_navoille
        * math.sqrt(rotation_period_days)
        * math.sqrt(planet_radius_re)
        / air_pressure_atm
    )

    # Lämmönkuljetus pienentää päiväntasaaja–napaeroa.

    delta_T_eq_poles = (
        delta_T_eq_poles_raw
        / (1.0 + kuljetuskerroin)
    )

    # Korjataan alueelliset lämpötilat siten,
    # että niiden keskimääräinen rakenne säilyy.

    T_poles = (
        T_global_C
        - delta_T_eq_poles / 2.0
    )

    T_eq = (
        T_global_C
        + delta_T_eq_poles / 2.0
    )

    # Pohjoisen ja eteläisen navan ero säilytetään
    # perihelion vaikutuksen mukaisesti.

    pole_N_difference = (
        T_pole_N_raw - T_poles_raw
    )

    pole_S_difference = (
        T_pole_S_raw - T_poles_raw
    )

    T_pole_N = (
        T_poles
        + pole_N_difference / (1.0 + kuljetuskerroin)
    )

    T_pole_S = (
        T_poles
        + pole_S_difference / (1.0 + kuljetuskerroin)
    )

    # ============================================================
    # 12. VUODENAIKAINEN VAIHTELU
    # ============================================================

    # Tässä käytetään planeetan alueellista lämpötilaeroa
    # vuodenaikaisvaihtelun mittakaavana.
    #
    # Lämpökapasiteetti vaimentaa vaihtelua.
    #
    # Vuodenaika_kerroin mahdollistaa myöhemmin tarkemman
    # fysikaalisen kalibroinnin.

    delta_T_vuosi = (
        abs(delta_T_eq_poles)
        * vuodenaika_kerroin
        / lampo_kapasiteetti
    )

    # Päiväntasaajalla ja navoilla vaihtelu on vastakkaissuuntainen.

    delta_t_paivantaasaajalla_vuosi = (
        delta_T_vuosi / 2.0
    )

    delta_t_navoilla_vuosi = (
        -delta_T_vuosi / 2.0
    )

    # ============================================================
    # 13. VUODEN MAKSIMI / MINIMI
    # ============================================================

    T_eq_max = (
        T_eq
        + delta_t_paivantaasaajalla_vuosi
    )

    T_eq_min = (
        T_eq
        - delta_t_paivantaasaajalla_vuosi
    )

    T_poles_max = (
        T_poles
        + abs(delta_t_navoilla_vuosi)
    )

    T_poles_min = (
        T_poles
        - abs(delta_t_navoilla_vuosi)
    )

    # ============================================================
    # 14. PALAUTUS
    # ============================================================

    return {

        # -------------------------
        # Globaali
        # -------------------------

        "globaali_lampotila_C":
            T_global_C,

        "kasvihuoneeton_lampotila_C":
            T_effective_C,

        # -------------------------
        # Kasvihuone
        # -------------------------

        "vesihoyry_kerroin":
            vesihoyry_kerroin,

        "kasvihuone_delta_T_C":
            delta_T_natural_ghg,

        # -------------------------
        # CO2
        # -------------------------

        "co2_ppm":
            atmos_co2_ppm,

        "co2_sateilypakote_W_m2":
            dF_co2,

        "co2_delta_T_C":
            delta_T_co2,

        # -------------------------
        # Alueelliset
        # -------------------------

        "paivantaasaaja_C":
            T_eq,

        "pohjoisnapa_C":
            T_pole_N,

        "etelanapa_C":
            T_pole_S,

        "navat_keski_C":
            T_poles,

        "delta_T_paivantaasaaja_navat_C":
            delta_T_eq_poles,

        # -------------------------
        # Lämmönkuljetus
        # -------------------------

        "lammon_kuljetuskerroin":
            kuljetuskerroin,

        # -------------------------
        # Vuodenaika
        # -------------------------

        "delta_t_paivantaasaajalla_vuosi_C":
            delta_t_paivantaasaajalla_vuosi,

        "delta_t_navoilla_vuosi_C":
            delta_t_navoilla_vuosi,

        "paivantaasaaja_max_C":
            T_eq_max,

        "paivantaasaaja_min_C":
            T_eq_min,

        "navat_max_C":
            T_poles_max,

        "navat_min_C":
            T_poles_min,
    }





import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def luokittele_koppen(temp, sade):
    """
    Luokittelee Köppen–Geiger-ilmastoluokan jokaiselle ruudulle.

    temp:
        Kuukausittaiset lämpötilat [12, height, width] °C

    sade:
        Kuukausittainen sademäärä [12, height, width]

    Palauttaa:
        koppen:
            2D-taulukko, jossa merkkijonoina Köppen-luokat.
    """

    kuukausia, height, width = temp.shape

    if kuukausia != 12:
        raise ValueError("Lämpötila- ja sademäärässä pitää olla 12 kuukautta.")

    koppen = np.full(
        (height, width),
        "??",
        dtype="<U3"
    )

    # ------------------------------------------------------------
    # Kuukausittaiset arvot
    # ------------------------------------------------------------

    vuosikeskilampo = np.mean(temp, axis=0)

    kylmin = np.min(temp, axis=0)
    lampimin = np.max(temp, axis=0)

    vuosisade = np.sum(sade, axis=0)

    # Kesäkuukaudet pohjoisella pallonpuoliskolla.
    # Tässä oletetaan, että kk 0-5 = vuoden lämmin puolisko.
    #
    # Eteläisen pallonpuoliskon tarkkaa sijaintia ei tästä
    # funktiosta tiedetä, joten vuodenaikojen jako tehdään
    # lämpötilan perusteella.
    lampimat_kuukaudet = temp >= 10.0
    lampimien_kuukausien_lkm = np.sum(lampimat_kuukaudet, axis=0)

    # ------------------------------------------------------------
    # KÖPPEN A: TROPIIKKI
    #
    # Kaikkien kuukausien keskilämpötila >= 18 °C
    # ------------------------------------------------------------

    A = kylmin >= 18.0

    # ------------------------------------------------------------
    # KÖPPEN E: POLAARISET
    #
    # EF = lämpimin kuukausi < 0 °C
    # ET = lämpimin kuukausi 0...10 °C
    # ------------------------------------------------------------

    EF = lampimin < 0.0

    ET = (
        (lampimin >= 0.0)
        &
        (lampimin < 10.0)
    )

    # ------------------------------------------------------------
    # KÖPPEN B: KUIVAT
    #
    # Kuivuusraja perustuu vuosittaiseen sademäärään
    # ja sateen vuodenaikaiseen jakaumaan.
    # ------------------------------------------------------------

    # Sateen jakauma
    sade_kesalla = np.sum(sade[0:6], axis=0)
    sade_talvella = np.sum(sade[6:12], axis=0)

    kesasade_prosentti = (
        sade_kesalla / np.maximum(vuosisade, 1e-9)
    ) * 100.0

    # Köppenin yksinkertaistettu kuivuusraja.
    #
    # Jos vähintään 70 % sateesta tulee lämpimällä puoliskolla,
    # käytetään korkeampaa kuivuusrajaa.
    #
    # Muuten käytetään alempaa rajaa.

    sadanta_raja = np.where(
        kesasade_prosentti >= 70.0,
        2.0 * (vuosikeskilampo + 14.0),
        2.0 * (vuosikeskilampo + 7.0)
    )

    B = vuosisade < sadanta_raja

    # ------------------------------------------------------------
    # B-LUOKKIEN JAKO
    # ------------------------------------------------------------

    # Aavikko = alle puolet kuivuusrajasta
    aavikko = vuosisade < (sadanta_raja * 0.5)

    aro = (
        B
        &
        ~aavikko
    )

    # h = kuuma
    # k = kylmä
    #
    # Köppenin raja: 18 °C vuosikeskilämpötila.

    BWh = (
        aavikko
        &
        (vuosikeskilampo >= 18.0)
    )

    BWk = (
        aavikko
        &
        (vuosikeskilampo < 18.0)
    )

    BSh = (
        aro
        &
        (vuosikeskilampo >= 18.0)
    )

    BSk = (
        aro
        &
        (vuosikeskilampo < 18.0)
    )

    # ------------------------------------------------------------
    # A-LUOKAT
    # ------------------------------------------------------------

    # Sadetta kaikissa kuukausissa:
    # ei varsinaista kuivaa kuukautta.

    kuivin_A = np.min(sade, axis=0)

    # Trooppinen sademetsä:
    # kuivin kuukausi >= 60 mm

    Af = (
        A
        &
        (kuivin_A >= 60.0)
    )

    # Monsuuni:
    # ei Af, mutta kuiva kuukausi vähintään 100 - Pann
    # yksinkertaistettuna tässä.

    vuosisade_A = vuosisade

    Am = (
        A
        &
        ~Af
        &
        (kuivin_A >= (100.0 - vuosisade_A / 25.0))
    )

    # Savanni:
    # loput A-ilmastosta.

    Aw = (
        A
        &
        ~Af
        &
        ~Am
    )

    # ------------------------------------------------------------
    # C-LUOKAT
    #
    # Kylmin kuukausi > 0 °C
    # Lämpimin kuukausi >= 10 °C
    # ------------------------------------------------------------

    C = (
        ~A
        &
        ~B
        &
        ~EF
        &
        ~ET
        &
        (kylmin > 0.0)
        &
        (kylmin < 18.0)
        &
        (lampimin >= 10.0)
    )

    # ------------------------------------------------------------
    # D-LUOKAT
    #
    # Kylmin kuukausi <= 0 °C
    # Lämpimin kuukausi >= 10 °C
    # ------------------------------------------------------------

    D = (
        ~A
        &
        ~B
        &
        ~EF
        &
        ~ET
        &
        (kylmin <= 0.0)
        &
        (lampimin >= 10.0)
    )

    # ------------------------------------------------------------
    # C / D - SADANTALUOKAT
    # ------------------------------------------------------------

    # Kuivin kesäkuukausi
    # ja kuivin talvikuukausi.

    kesan_sade = np.min(
        sade[0:6],
        axis=0
    )

    talven_sade = np.min(
        sade[6:12],
        axis=0
    )

    kesan_sade_summa = np.sum(
        sade[0:6],
        axis=0
    )

    talven_sade_summa = np.sum(
        sade[6:12],
        axis=0
    )

    # ------------------------------------------------------------
    # C/D - a, b, c, d
    # ------------------------------------------------------------

    # a:
    # kuumin kuukausi >= 22 °C
    #
    # b:
    # kuumin kuukausi < 22 °C ja vähintään 4 kk >= 10 °C
    #
    # c:
    # 1-3 kk >= 10 °C
    #
    # d:
    # D-ilmastojen erittäin kylmä talvi

    kirjain2_a = (
        lampimin >= 22.0
    )

    kirjain2_b = (
        (lampimin < 22.0)
        &
        (lampimien_kuukausien_lkm >= 4)
    )

    kirjain2_c = (
        lampimien_kuukausien_lkm >= 1
    )

    kirjain2_d = (
        (kylmin <= -38.0)
    )

    # ------------------------------------------------------------
    # C/D - SADANTYYPIT
    #
    # s = kuiva kesä
    # w = kuiva talvi
    # f = ei kuivaa kautta
    # ------------------------------------------------------------

    kuiva_kesa = (
        kesan_sade < 40.0
    )

    kuiva_talvi = (
        talven_sade < 40.0
    )

    f = (
        ~kuiva_kesa
        &
        ~kuiva_talvi
    )

    s = (
        kuiva_kesa
        &
        (talven_sade_summa > kesan_sade_summa * 3.0)
    )

    w = (
        kuiva_talvi
        &
        (kesan_sade_summa > talven_sade_summa * 10.0)
    )

    # ------------------------------------------------------------
    # C-LUOKAT
    # ------------------------------------------------------------

    # Cfa
    Cfa = C & f & kirjain2_a

    # Cfb
    Cfb = C & f & kirjain2_b

    # Cfc
    Cfc = C & f & kirjain2_c & ~kirjain2_b

    # Csa
    Csa = C & s & kirjain2_a

    # Csb
    Csb = C & s & kirjain2_b

    # Csc
    Csc = C & s & kirjain2_c & ~kirjain2_b

    # Cwa
    Cwa = C & w & kirjain2_a

    # Cwb
    Cwb = C & w & kirjain2_b

    Cwc = C & w & kirjain2_c & ~kirjain2_b

    # ------------------------------------------------------------
    # D-LUOKAT
    # ------------------------------------------------------------

    Dfa = D & f & kirjain2_a
    Dfb = D & f & kirjain2_b
    Dfc = D & f & kirjain2_c & ~kirjain2_b
    Dfd = D & f & kirjain2_d

    Dsa = D & s & kirjain2_a
    Dsb = D & s & kirjain2_b
    Dsc = D & s & kirjain2_c & ~kirjain2_b
    Dsd = D & s & kirjain2_d

    Dwa = D & w & kirjain2_a
    Dwb = D & w & kirjain2_b
    Dwc = D & w & kirjain2_c & ~kirjain2_b
    Dwd = D & w & kirjain2_d

    # ============================================================
    # KIRJOITETAAN LUOKAT KARTALLE
    # ============================================================

    # A
    koppen[Af] = "Af"
    koppen[Am] = "Am"
    koppen[Aw] = "Aw"

    # B
    koppen[BWh] = "BWh"
    koppen[BWk] = "BWk"
    koppen[BSh] = "BSh"
    koppen[BSk] = "BSk"

    # C
    koppen[Cfa] = "Cfa"
    koppen[Cfb] = "Cfb"
    koppen[Cfc] = "Cfc"
    koppen[Csa] = "Csa"
    koppen[Csb] = "Csb"
    koppen[Csc] = "Csc"
    koppen[Cwa] = "Cwa"
    koppen[Cwb] = "Cwb"
    koppen[Cwc] = "Cwc"

    # D
    koppen[Dfa] = "Dfa"
    koppen[Dfb] = "Dfb"
    koppen[Dfc] = "Dfc"
    koppen[Dfd] = "Dfd"
    koppen[Dsa] = "Dsa"
    koppen[Dsb] = "Dsb"
    koppen[Dsc] = "Dsc"
    koppen[Dsd] = "Dsd"
    koppen[Dwa] = "Dwa"
    koppen[Dwb] = "Dwb"
    koppen[Dwc] = "Dwc"
    koppen[Dwd] = "Dwd"

    # E
    koppen[ET] = "ET"
    koppen[EF] = "EF"

    return koppen


def piirra_koppen(koppen):
    """
    Piirtää Köppen–Geiger-ilmastoluokituksen karttana.

    Parametrit
    ----------
    koppen : 2D numpy-taulukko
        luokittele_koppen()-funktion palauttama taulukko.
    """

    # ------------------------------------------------------------
    # Köppen-luokkien värit
    # ------------------------------------------------------------

    varit = {
        "Af": "#006400",
        "Am": "#228B22",
        "Aw": "#9ACD32",

        "BWh": "#FF4500",
        "BWk": "#FFA07A",
        "BSh": "#F4A460",
        "BSk": "#DEB887",

        "Csa": "#FFD700",
        "Csb": "#F0E68C",
        "Csc": "#EEE8AA",

        "Cfa": "#00CED1",
        "Cfb": "#40E0D0",
        "Cfc": "#AFEEEE",

        "Cwa": "#66CDAA",
        "Cwb": "#7FFFD4",
        "Cwc": "#B0E0E6",

        "Dfa": "#1E90FF",
        "Dfb": "#4169E1",
        "Dfc": "#6495ED",
        "Dfd": "#483D8B",

        "Dsa": "#9370DB",
        "Dsb": "#8A2BE2",
        "Dsc": "#BA55D3",
        "Dsd": "#800080",

        "Dwa": "#4682B4",
        "Dwb": "#5F9EA0",
        "Dwc": "#708090",
        "Dwd": "#2F4F4F",

        "ET": "#B0C4DE",
        "EF": "#FFFFFF",

        "??": "#000000",
    }

    # ------------------------------------------------------------
    # Muutetaan luokat numeroiksi
    # ------------------------------------------------------------

    luokat = list(varit.keys())

    numero = np.zeros(
        koppen.shape,
        dtype=int
    )

    for i, luokka in enumerate(luokat):
        numero[koppen == luokka] = i

    cmap = ListedColormap(
        [varit[x] for x in luokat]
    )

    # ------------------------------------------------------------
    # Piirretään
    # ------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(15, 7)
    )

    ax.imshow(
        numero,
        cmap=cmap,
        interpolation="nearest",
        aspect="auto"
    )

    ax.set_title(
        "Köppen–Geiger-ilmastoluokitus"
    )

    ax.set_xlabel("Pituusaste")
    ax.set_ylabel("Leveysaste")

    # ------------------------------------------------------------
    # Leveysasteiden asteikko
    # ------------------------------------------------------------

    height, width = koppen.shape

    y_ticks = np.linspace(
        0,
        height - 1,
        7
    )

    y_labels = [
        "90° N",
        "60° N",
        "30° N",
        "0°",
        "30° S",
        "60° S",
        "90° S"
    ]

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)

    # ------------------------------------------------------------
    # Pituusasteet
    # ------------------------------------------------------------

    x_ticks = np.linspace(
        0,
        width - 1,
        9
    )

    x_labels = [
        "-180°",
        "-135°",
        "-90°",
        "-45°",
        "0°",
        "45°",
        "90°",
        "135°",
        "180°"
    ]

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)

    # ------------------------------------------------------------
    # Selite
    # ------------------------------------------------------------

    from matplotlib.patches import Patch

    legend_items = []

    for luokka in luokat:
        if np.any(koppen == luokka):
            legend_items.append(
                Patch(
                    facecolor=varit[luokka],
                    edgecolor="black",
                    label=luokka
                )
            )

    ax.legend(
        handles=legend_items,
        title="Köppen",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        ncol=1
    )

    plt.tight_layout()
    plt.show()




def plot_ocean_currents(
    dem,
    current_x,
    current_y,
    ocean,
    step=25,
    figsize=(16, 8),
    cmap="terrain",
    color="cyan",
    scale=0.08,
    title="Merivirrat",
):
    """
    Piirtää merivirrat DEM-kartan päälle.

    Parameters
    ----------
    dem : 2D numpy array
        Maaston korkeus / meren syvyys metreinä.

    current_x : 2D numpy array
        Merivirran X-komponentti.

    current_y : 2D numpy array
        Merivirran Y-komponentti.

    ocean : 2D bool array
        True merellä, False maalla.

    step : int
        Kuinka monta pikseliä jätetään nuolten väliin.

    scale : float
        Quiver-nuolten skaala.
    """

    h, w = dem.shape

    # ---------------------------------------------------------
    # Koordinaatit
    # ---------------------------------------------------------

    lon = np.linspace(
        -180,
        180,
        w,
        endpoint=False
    )

    lat = np.linspace(
        90,
        -90,
        h
    )

    LON, LAT = np.meshgrid(
        lon,
        lat
    )

    # ---------------------------------------------------------
    # Harvenna dataa
    # ---------------------------------------------------------

    X = LON[::step, ::step]
    Y = LAT[::step, ::step]

    U = current_x[::step, ::step]
    V = current_y[::step, ::step]

    MASK = ocean[::step, ::step]

    # ---------------------------------------------------------
    # Poista maa-alueiden nuolet
    # ---------------------------------------------------------

    U = np.where(
        MASK,
        U,
        np.nan
    )

    V = np.where(
        MASK,
        V,
        np.nan
    )

    # ---------------------------------------------------------
    # Virtausnopeus väritystä varten
    # ---------------------------------------------------------

    speed = np.sqrt(
        U**2 + V**2
    )

    # ---------------------------------------------------------
    # Kuva
    # ---------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=figsize
    )

    ax.imshow(
        dem,
        extent=[
            -180,
            180,
            -90,
            90
        ],
        origin="upper",
        cmap=cmap
    )

    # ---------------------------------------------------------
    # Merivirrat
    # ---------------------------------------------------------

    q = ax.quiver(
        X,
        Y,
        U,
        V,
        speed,
        color=color,
        angles="xy",
        scale_units="xy",
        scale=scale,
        width=0.002,
        pivot="mid"
    )

    # ---------------------------------------------------------
    # Väriasteikko
    # ---------------------------------------------------------

    cbar = fig.colorbar(
        q,
        ax=ax
    )

    cbar.set_label(
        "Virran suhteellinen nopeus"
    )

    # ---------------------------------------------------------
    # Akselit
    # ---------------------------------------------------------

    ax.set_xlim(
        -180,
        180
    )

    ax.set_ylim(
        -90,
        90
    )

    ax.set_xlabel(
        "Pituusaste"
    )

    ax.set_ylabel(
        "Leveysaste"
    )

    ax.set_title(
        title
    )

    plt.tight_layout()

    return fig, ax





def avaa_rasterit():
    # 1. Tiedostopolut (Korvaa omilla tiedostoillasi)
    lampo_tiedosto = 'lampotilat.tif'
    sade_tiedosto = 'sateet.tif'
    
    # 2. Datan lataaminen rasterio-kirjastolla
    with rasterio.open(lampo_tiedosto) as src_temp:
        lampotilat = src_temp.read(1)
        meta = src_temp.meta
        nodata_val = meta.get('nodata', None)

    with rasterio.open(sade_tiedosto) as src_rain:
        sateet = src_rain.read(1)
    return(lampotilat, sateet)
        
 


#############################
## paaohjelma




# --- VISUALISOINTIASAKSET ---

# Määritetään kullekin biomille luonnollinen väri
biomi_varit = [
    '#1a365d',  # 0: Meri (tummansininen)
    '#e2cb99',  # 1: Aavikko (hiekanruskea)
    '#adc178',  # 2: Savanni/Ruohikko (vaaleanvihreä)
    '#132a13',  # 3: Sademetsä (tummanvihreä)
    '#4f772d',  # 4: Lauhkea metsä (keskivihreä)
    '#283618',  # 5: Havumetsä (havunvihreä)
    '#9ba5a0',  # 6: Tundra (harmaanvihreä)
    '#ffffff'   # 7: Ikijää (valkoinen)
]

biomi_nimet = [
    'Sea', 'Desert', 'Savanna / Grassland', 'Rain forest', 
    'Temperate forest', 'Needle forest', 'Tundra', 'Ice'
]

# Luodaan kustomoitu värikartta matplotlibille
cmap_biomit = mcolors.ListedColormap(biomi_varit)





#############################################
## Python main program, map of planet
# --- Esimerkki käytöstä ---
if __name__ == "__main__":
    if(will_load_image==True):
        imagee, width, height=load_image_and_normalize(imagename)
    else:
        imagee = generate_spherical_noise(width, height, scale=base_noisescale, octaves=16, seed=seed1)
        #imagee=np.pow(imagee,2)
        #imagee=np.exp(np.exp(np.exp(imagee)))
        imagee=np.pow(imagee,3)
        imagee=normalize(imagee)
        #imagee=sigmoid_dem(imagee, saatokerroin=0.5, keskiarvo=0.5) 
        #imagee=sigmoid_meri_manner_jakauma(imagee, säätökerroin=-2)
        imagee=muotoile_maan_jakauma(imagee, sealevel=0.5)
        imagee=normalize(imagee)
        imagee=spherical_noise_offset(imagee, move_amount_max=distort_coeff*height, scale=distort_noisescale, seed=seed1)
        imagee=normalize(imagee)

    
    print("S1, ecc, tilt, mvelp")
    print(S1, ecc, tilt, mvelp)
    print("Rotation        P / h       :" , rotation_period_days*24)
    print("Mass, radius in earth units :", planet_mass_me, planet_radius_re)
    
    rotation_period_hours=rotation_period_days*24
    planet_year_length_days=orbital_period_years*365.25
    
    gee_earths=planet_mass_me/math.pow(planet_radius_re,2)
    gee_ms2=gee_earths*9.81
    
    np.random.seed(seed=seed1)
 
 
    tulokset = laske_planeetan_lampotila(
    ecc=ecc,
    tilt=tilt,
    mvelp=mvelp,
    S1=S1,

    atmos_co2_ppm=atmos_co2_ppm,

    air_pressure_atm=air_pressure_atm,

    albedo_keski=0.30,

    co2_viite_ppm=280,
    ilmastosensitiivisyys=0.8,

    # 1 = Maa
    vesihoyry_kerroin=1.0,

    lammon_kuljetus_paivantaasaajalta_navoille=1.0,
    lampo_kapasiteetti=1.0,

    rotation_period_days=rotation_period_days,
    planet_radius_re=planet_radius_re,
    )

    #for nimi, arvo in tulokset.items():
    #    print(f"{nimi:45s}: {arvo:.2f}")

    mean_temp=tulokset["globaali_lampotila_C"]
    temp_diff=70*math.sqrt(rotation_period_days)*math.sqrt(planet_radius_re)/air_pressure_atm
    temp_dev=temp_diff/2
    polar_temp=mean_temp-temp_dev
    temp_diff=mean_temp+temp_dev
    print(mean_temp, temp_diff)
    #quit(-1)
 
    lats = np.linspace(90, -90, height)
    lons= np.linspace(-180, -180, width)
    painot_1d = np.cos(np.radians(lats))
    painom_matriisi = np.repeat(painot_1d[:, np.newaxis], width, axis=1)
    	
    dem, landmask=create_dem_from_array(imagee, sealevel, dem_min, dem_max)

    pix_areas = laske_globaalit_pikselialat(height, width, planet_radius)
    sphere_area=4 * np.pi * planet_radius**2
    relief=(np.copy(dem))*landmask
    seamask=np.copy(relief)
    seamask=np.where(seamask>0,0,1)
 
    #eroosio
    relief = simulate_erosion(relief, num_droplets=30000) 
    #relief = simulate_thermal_erosion(relief, iterations=15, c_repose=2.0, talus_rate=0.2)
    
    #plt.imshow(relief)
    #plt.show()
    #quit(-1)


    distsea=distance_to_sea(relief, planet_radius)
    distmountains=distance_to_someheight(relief, planet_radius, 1500)
    twi = calculate_twi(dem, cell_size=100.0)    
    tpi=calculate_tpi(dem, window_size=5)
  
    #plt.imshow(distmountains)
    #plt.show()
    #quit(-1)
    
    relief2=np.copy(relief)
    relief2=np.where(relief2<=0,np.nan,relief2 )
    light_direction = [-0.0, 1, 0.0] 
    # 3. Lasketaan varjot
    #hillshade2 = laske_ray_trace_shadows(relief, light_direction)
    # Lasketaan emboss/hillshade (valo luoteesta, 45 asteen kulmassa)
    hillshade1 = laske_hillshade(relief, azimuth=315, angle_altitude=45)
    print(np.shape(dem))
    tulokset = calculate_land_sea_percentage(landmask)
    print(f"Land: {tulokset['maa_prosentti']}%")
    print(f"Sea: {tulokset['meri_prosentti']}%")
    stats = calculate_dem_statistics(dem)    
    print(f"Meren suurin syvyys (minimi):  {stats['min_sea_depth_m']} m")
    print(f"Maan suurin korkeus (maksimi): {stats['max_land_height_m']} m")
    print(f"Meren keskisyvyys:             {stats['mean_sea_depth_m']} m")
    print(f"Maan keskikorkeus:             {stats['mean_land_height_m']} m")

    #temps12, winds12, precips12 = laske_perus_ilmasto(relief, tilt=tilt, ecc=ecc, P=1.0)
    kiertosolut=int(3/rotation_period_days)
    temps12, winds12_x, winds12_y, winds12_z, precips12 = (
    laske_perus_ilmasto(
        relief, t_mean=mean_temp, delta_t=temp_dev,
        tilt=tilt,
        ecc=ecc,
        mvelp=mvelp,
        P=1,
        kiertosolut=kiertosolut
    )
    )

    lampotilat=np.mean(temps12, axis=0)
    sateet=np.sum(precips12, axis=0)



    # Lasketaan lämpötilat
    #lampotilat = laske_vuosittainen_lampotila(relief, polar_temp, temp_diff, sealevel=0.5, max_korkeus_m=5000)
    pet = calculate_pet(lampotilat,relief,landmask)
    ice=np.copy(lampotilat)
    ice=np.where(lampotilat<-10,1,np.nan)
    noice=np.where(lampotilat<-10,np.nan,1)
    maski = np.isnan(lampotilat)
    alueen_keski_lampotila = np.average(lampotilat, weights=painom_matriisi)
    moisture_coeff=tulokset['meri_prosentti']/100.0
    #moisture_temperature=
    delta_T = (alueen_keski_lampotila - 15.0) ## earth now 15 c
    sademäärä_kerroin = 0.02
    uusi_sadek = (1 + sademäärä_kerroin * delta_T)
    uusi_sadek = max(0.0, uusi_sadek)
    kosteus_kerroin_kaatosade = 0.07
   
    uusi_kosteus_kapasiteetti_kaatosade = 1* math.exp(kosteus_kerroin_kaatosade * delta_T)
    print(moisture_coeff, uusi_sadek)
    moisture_coeff=moisture_coeff*uusi_sadek*5
    print(moisture_coeff)
    ## earth now 15 c, 990 mm
    #sateet, tuulet = laske_sademaara_soluilla(relief, distsea, sealevel=0.5, moisture_coeff=moisture_coeff)
    precips12=precips12*moisture_coeff
    sateet=sateet*moisture_coeff

    alueen_keski_sademaara  = np.average(sateet, weights=painom_matriisi)
    alueen_keski_lampotila  = np.average(lampotilat, weights=painom_matriisi)
    print(f"Alueen painotettu keskilämpötila: {alueen_keski_lampotila:.2f} °C")
    print(f"Alueen painotettu keskisademäärä: {alueen_keski_sademaara:.2f} mm")
    koppen = luokittele_koppen(temps12,precips12)

    #plt.show()
    #plt.imshow(lampotilat)
    #piirra_koppen(koppen)        
    #plt.imshow(temps12[0])
    #plt.show()
    #plt.imshow(temps12[6])
    #plt.show()
    #plt.imshow(precips12[0])
    #plt.show()
    #plt.imshow(precips12[0])
    #plt.show()
    #plt.imshow(precips12[6]-precips12[0])
    #plt.show()
    #plt.imshow(sateet)
    #plt.contour(sateet, levels=[250,500,1000,2000])
    #plt.show()
    #quit(-1)  


    tuuli_x, tuuli_y, tuuli_z=laske_tuuli(dem, kiertosolut=kiertosolut)#3 mean
	    
    ocean_currents_result= ocean_currents(
    dem=dem,
    wind_x=tuuli_x,
    wind_y=tuuli_y,
    wind_z=tuuli_z,
    temperature=lampotilat,
    precipitation=sateet,

    planet_radius_km=planet_radius,
    rotation_period_hours=rotation_period_days*24,
    rotation_direction=1,
    )

    current_x = ocean_currents_result["current_x"]
    current_y = ocean_currents_result["current_y"]
    current_z = ocean_currents_result["current_z"]

    ocean = ocean_currents_result["ocean"]
    upwelling = ocean_currents_result["upwelling"]

    #fig, ax = plot_ocean_currents(dem,ocean_currents_result["surface_x"]*10,ocean_currents_result["surface_y"]*10,ocean_currents_result["ocean"],step=18,color="white",title="Pintamerivirrat")
    #fig, ax = plot_ocean_currents(dem,ocean_currents_result["deep_x"]*100,ocean_currents_result["deep_y"]*100,ocean_currents_result["ocean"],step=18,color="white",title="Syvämerivirrat")
    #fig, ax = plot_ocean_currents(dem,ocean_currents_result["current_x"]*10,ocean_currents_result["current_y"]*10,ocean_currents_result["ocean"],step=18,color="white",title="Syvämerivirrat")

    
    #print("Vuositaulukon muoto ", np.shape(lampotilat_12) )
     
    #plt.show()
    jaatikko_tulos = laske_jaatikko(
    dem,
    lampotilat,
    sateet,
    planet_radius,
    gee_ms2,
    num_years=5000
    )
    print("Jäätikköala:",
      jaatikko_tulos["glacier_area_m2"] / 1e12,
      "milj. km²")

    print("Jäätilavuus:",
      jaatikko_tulos["ice_volume_m3"] / 1e12,
      "km³")

    print("Jään massa:",
      jaatikko_tulos["ice_mass_kg"] / 1e18,
      "Gt")

    print("Merenpinnan muutos:",
      jaatikko_tulos["sea_level_change_m"],
      "m")
    glacier_ice_thickness=jaatikko_tulos["ice_thickness"]
    glacier_mask_ice_age=jaatikko_tulos["glacier_mask"]

    ## possible ice age
    #sea_level_ice_age=jaatikko_tulos["sea_level_change_m"]
    sea_level_ice_age=-120
    
    dem_ice_age=np.copy(dem)-sea_level_ice_age
    relief_ice_age=np.copy(dem_ice_age)
        
    seamask_ice_age=np.copy(relief_ice_age)
    seamask_ice_age=np.where(seamask_ice_age>0,0,1)    
    landmask_ice_age=np.copy(relief_ice_age)
    landask_ice_age=np.where(landmask_ice_age<=0,1,0) 
    temperature_ice_age=np.copy(lampotilat)-10
    precipitation_ice_age=np.copy(sateet)*0.5 
    loess_areas_ice_age=np.copy(sea_level_ice_age)
      
    ## mioseeni sade  1.08x,  lampotila + 3.5 jopa .1x, jopa 5-8 C +      
    #plt.imshow(glacier_mask_ice_age)
    
    #plt.show()
    #quit(-1)
    #rivers0, accumulation0, flow_to0=calculate_rivers(relief,sateet, pet, 10,100000)
    rivers0, accumulation0, flow_to0=calculate_rivers(relief,sateet, pet, 100,10000)
    lakes1, lake_depths1, lake_ids1 = calculate_lakes(relief,sateet,pet,flow_to0,accumulation0,cell_size=1000,min_inflow=1000000)
    #plt.imshow(lakes1*landmask)
    #plt.imshow(accumulation0*landmask)
    #plt.imshow(twi*landmask)
    #plt.imshow(twi*landmask*accumulation0)
    #plt.show()    
    #rivers, accumulation, flow_to

    rivers1=np.copy(rivers0)
    rivers1=np.where(rivers1==0,np.nan, rivers1)
    distrivers=np.copy(rivers0)
    #distrivers=np.where(distrivers==np.nan,-1,1)
    distrivers=distance_to_someheight(distrivers, planet_radius, 1)
    
    distwater=np.copy(distsea)
    distwater=np.where(distwater>distrivers, distrivers, distwater)

    manner_osuudet, manner_koko_jarjestys = laske_mantereet(relief, planet_radius=planet_radius)
    manner_osuudet_ice_age, manner_koko_jarjestys_ice_age = laske_mantereet(relief_ice_age, planet_radius=planet_radius)   
    print("\nValmis! Taulukoiden muodot:", manner_osuudet.shape, manner_koko_jarjestys.shape)
    #plt.imshow(rivers1)
    #plt.imshow(distrivers)
    #plt.imshow(distwater)
    #plt.show()
    #quit(-1)
    
    maan_albedo=laske_albedoluokat(landmask, lampotilat, sateet)
    merijaa=arvioi_merijaa(landmask, lampotilat, sateet)
    meren_albedo=np.copy(merijaa)
    meren_albedo=np.where(meren_albedo>0,0.6,0.06)*seamask
    albedo=np.copy(meren_albedo)
    albedo=np.where(albedo==0,maan_albedo, meren_albedo)
    #plt.imshow(albedo)
    #plt.show()
    #quit(-1)

    #plt.imshow(pet)
    #plt.imshow(merijaa)
    #plt.show()
    #quit(-1)
    # 2. Generoidaan biomikartta
    biomi_kartta = luo_biomikartta(relief, lampotilat, sateet, sealevel=0.5)
   # 3. Kutsutaan aluohjelmaa laskentaa varten
    holdridge_tulos = laske_holdridge_luokat(lampotilat, sateet, nodata_arvo=np.nan)
    npp_tulos = laske_npp_miami(lampotilat, sateet, nodata_arvo=np.nan)
    
    # 4. NÄYTTÖ (Visualisointi erikseen main-funktiossa)
    # Maskataan NoData (-1) pois visualisointia varten, jotta se ei vääristä väriskaalaa
    npp_naytettava = np.where(npp_tulos == -1, np.nan, npp_tulos)*landmask
    # 3. Kutsutaan aluohjelmaa keskiarvon laskentaan
    manner_keski_npp = laske_mannerten_keski_npp(npp_tulos, landmask, height, width)
    
    maski=np.isnan(sateet)
    painom_matriisi[maski] = 0

    alueen_keski_sademaara  = np.average(sateet, weights=painom_matriisi)
    print(f"Alueen painotettu keskilämpötila: {alueen_keski_lampotila:.2f} °C")
    print(f"Alueen painotettu keskisademäärä: {alueen_keski_sademaara:.2f} mm")
    # 4. NÄYTTÖ (Tulostetaan laskettu arvo erikseen mainissa)
    print("\n--- ANALYYSIN TULOKSET ---")
    print(f"Rasterin resoluutio: {width} x {height} pikseliä")
    print(f"Mantereiden pinta-alapainotettu keski-NPP: {manner_keski_npp:.2f} g/m²/vuosi")
    print("--------------------------")   

    suitability_rain_agriculture = lbk_sadeviljely_soveltuvuus(relief,sateet, lampotilat)*landmask
    agriculture, pastoralism, nomadism = subsistence_suitability_threepars(relief,sateet, lampotilat )
 
    results1=subsistence_suitability(
    elevation=relief,
    precipitation=sateet,
    temperature=lampotilat,
    npp=npp_tulos,
    twi=twi,
    distance_to_water=distwater,
    tpi=tpi, biome=biomi_kartta
    )

    agriculture = results1["agriculture"]
    pastoralism = results1["pastoralism"]
    nomadism = results1["nomadism"]
    hunter_gatherer = results1["hunter_gatherer"]

	# Lasketaan todennäköisyyskartta
    aly_todennakoisyyskartta = laske_synnyinsija_todennakoisyys(relief, lampotilat, sateet)
    # Etsitään suurin arvo ja sen sijainti (y, x -pikselit)
    aly_idx = np.unravel_index(np.argmax(aly_todennakoisyyskartta), aly_todennakoisyyskartta.shape)
    aly_max_todennakoisyys = aly_todennakoisyyskartta[aly_idx]
    alyy=aly_idx[0]
    alyx=aly_idx[1]    
    aly_lon, aly_lat = pixel_to_lonlat(alyx, alyy, width, height)
    print(f"Älyllisen lajin pikseli ({alyx}, {alyy}) -> Longitude: {aly_lon:.4f}, Latitude: {aly_lat:.4f}")    
    #print(f"Älyllisen lajin todennäköisin syntypaikka on koordinaateissa: {max_idx}", alyx, alyy)
    print(f"Paikan optimaalisuusindeksi: {aly_max_todennakoisyys * 100:.1f} %")
    
    alyevo=laske_evoluutiopaine(relief, lampotilat, sateet, vuosisadat=100)
    habitability=laske_human_habitability_index(lampotilat, sateet)
    ## mureybat
    target_alt = 250.0
    target_temp = 17
    target_rain = 250.0
    ## cayonu
    #target_alt = 830.0
    #target_temp = 16.5
    #target_rain = 650.0
    ## roma
    #target_alt = 100
    #target_temp = 16
    #target_rain = 880
    ##  cuzco
    ##target_alt = 3400
    ##target_temp = 8
    ##target_rain = 750
    ##  anyang
    #target_alt = 72
    #target_temp = 15
    #target_rain = 580
    ##  tenochtitlan
    ##target_alt = 2240
    ##target_temp = 16.5
    ##target_rain = 850
    ## la venta, oaxaca
    ##target_alt = 2240
    ##target_temp = 27
    ##target_rain = 1000
    ##  tikal
    #target_alt = 200
    #target_temp = 25
    #target_rain = 1275
    #cayonu_index,agriculture_index,pastoral_index,suitability,kulttuuri,leviamisaika=etsi_paikka_ja_levia(target_alt, target_temp, target_rain,relief, lampotilat, sateet, ajo_vuodet=10000)
    #kulttuuri_index,agriculture_index,pastoral_index,suitability,kulttuuri,leviamisaika=etsi_paikka_ja_levia(target_alt, target_temp, target_rain,relief, lampotilat, sateet, ajo_vuodet=10000)
    kulttuuri_tod,agriculture_index,pastoral_index,suitability,kulttuuri,leviamisaika,vaesto,carrying_capacity=etsi_paikka_ja_levia(target_alt, target_temp, target_rain,relief, lampotilat, sateet, ajo_vuodet=10000)
    kulttuuri_idx = np.unravel_index(np.argmax(kulttuuri_tod), kulttuuri_tod.shape)
    kulttuuri_max_todennakoisyys = kulttuuri_tod[aly_idx]
    kulttuuri_y=kulttuuri_idx[0]
    kulttuuri_x=kulttuuri_idx[1]
   
    kulttuuri_lon, kulttuuri_lat = pixel_to_lonlat(kulttuuri_x, kulttuuri_y, width, height)
    print(f"Kulttuurin 1 alkupaikka pikseli ({kulttuuri_x}, {kulttuuri_y}) -> K Longitude: {kulttuuri_lon:.4f}, K Latitude: {kulttuuri_lat:.4f}")    

    konduktanssi=np.copy(npp_tulos)
    konduktanssi=1/npp_tulos
    kantokyky=np.copy(npp_tulos)

    ##  uruk
    target2_alt = 20.0
    target2_temp = 25.0
    target2_rain = 80

    kulttuuri2_tod,agriculture_index2,pastoral_index2,suitability2,kulttuuri2,leviamisaika2,vaesto2,carrying_capacity2=etsi_paikka_ja_levia(target2_alt, target2_temp, target2_rain,relief, lampotilat, sateet, ajo_vuodet=1000)
    kulttuuri2_idx = np.unravel_index(np.argmax(kulttuuri2_tod), kulttuuri2_tod.shape)
    kulttuuri2_max_todennakoisyys = kulttuuri2_tod[aly_idx]
    kulttuuri2_y=kulttuuri2_idx[0]
    kulttuuri2_x=kulttuuri2_idx[1]
   
    kulttuuri2_lon, kulttuuri2_lat = pixel_to_lonlat(kulttuuri2_x, kulttuuri2_y, width, height)
    print(f"Kulttuurin 2 alkupaikka pikseli ({kulttuuri2_x}, {kulttuuri2_y}) -> K Longitude: {kulttuuri2_lon:.4f}, K Latitude: {kulttuuri2_lat:.4f}")    

    konduktanssi2=np.copy(npp_tulos)
    konduktanssi2=1/npp_tulos
    kantokyky2=np.copy(npp_tulos)


    #kutulos = leviamis_rasteri(
    #alku_lon=kulttuuri_lon,
    #alku_lat=kulttuuri_lat,alku_vaesto=1,
    #kaksink_aika=250,
    #konduktanssi=konduktanssi,
    #kantokyky=kantokyky,
    #aika_kysytty=5000,
	#)
	
    #saapuminen = kutulos["saapumisaika"]
    #vaesto = kutulos["vaesto"]
    #K = kutulos["kantokyky"]
    #aika_K = kutulos["aika_kantokykyyn"]
    #plt.imshow(vaesto)
    #plt.imshow(saapuminen*landmask)
    #plt.show()
    #quit(-1)
    #plt.imshow(suitability_rain_agriculture*landmask)
  
  
    #plt.imshow(hunter_gatherer*landmask)
    #plt.imshow(nomadism*landmask)
    #plt.imshow(pastoralism*landmask)
    #plt.imshow(agriculture*landmask)
    #plt.imshow(alyevo*landmask) #aly_todennakoisyyskartta
    #plt.imshow(aly_todennakoisyyskartta*landmask) 
    #plt.imshow(habitability*landmask) 
    #plt.imshow(cayonu_index*landmask) 
    #plt.imshow(kulttuuri*landmask) 
    #plt.show()
    
    civilization_temp=np.copy(lampotilat)
    civilization_temp=np.where(civilization_temp>12,1,0)
    #plt.imshow(civilization_temp)
    #plt.show()
    kuiva=np.copy(sateet)
    kuiva=np.where(kuiva<300,1,0)
    vuoriin=np.copy(distmountains)
    mereen=np.copy(distsea)
    jokiin=np.copy(distrivers)
    jokiin=np.where(jokiin<250,1,0)
    vuoriin=np.where(vuoriin<250,1,0)
    mereen=np.where(mereen<250,1,0)
    sivilisaatio=landmask*civilization_temp*kuiva*mereen*vuoriin

    sivilisaatio=np.where(sivilisaatio==1,1,np.nan )
    #plt.imshow(jokiin)
    #plt.imshow(sivilisaatio)
    #plt.show()    
    civ_raster=laske_sivilisaatiopisteet(sateet, lampotilat, distrivers, distsea, distmountains, relief)# --- ESIMERKKIKÄYTTÖ ---
    civ_sites = find_max_points(civ_raster,planet_radius_km=6371.0,min_distance_km=500.0,n_points=5)
    
    civ_threshold = 0.8
    civ_sites = [
    p for p in civ_sites
        if p["value"] > civ_threshold
    ]

    for p in civ_sites:
        print(
        f"pixel=({p['row']}, {p['col']}), "
        f"lon={p['lon']:.3f}, "
        f"lat={p['lat']:.3f}, "
        f"value={p['value']:.4f}"
        )
    # Pisteet
    civ_lons = [p["lon"] for p in civ_sites]
    civ_lats = [p["lat"] for p in civ_sites]


    
    #plt.imshow(distsea)
    #plt.imshow(sivilisaatio)
    #plt.imshow(sivilisaatio)
    #plt.show()
    #quit(-1)
    
    
    #plt.imshow(imagee)
    #plt.imshow(relief2, cmap="terrain",origin='upper')
    #plt.imshow(rivers1, cmap="viridis",origin='upper')    
    #plt.imshow(varjostus, cmap='gray', alpha=0.2, origin='lower')
    #plt.imshow(dem)
    #plt.imshow(landmask)
    #plt.show()
    # --- PIIRETÄÄN KARTTA ---
    #plt.figure(figsize=(12, 5))
    # Käytetään 'coolwarm'-värikarttaa (sininen = kylmä, punainen = kuuma)
    #plt.imshow(lampotilat, cmap='coolwarm', origin='upper')
    #plt.colorbar(label="Vuoden keskilämpötila (°C)")
    #plt.title("Maailman lämpötilajakautuma (Leveyspiiri + Korkeusasema)")
    #plt.axis('off')
    #plt.show()

    # --- VISUALISOINTI ---
    #fig, ax = plt.subplots(figsize=(12, 6))
    # Piirretään sademääräkartta
    #im = ax.imshow(sateet, cmap='YlGnBu', origin='upper')
    #fig.colorbar(im, label="Sademäärä (mm / vuosi)")
    # Lisätään nuolia kuvaamaan tuulen suuntaa valituilla leveyspiireillä (näytetään joka 20. rivi)
    #for row in range(10, height, 20):
    #    suunta = "--> LÄNSITUULI" if tuulet[row, 0] > 0 else "<-- PASAATI / ITÄTUULI"
    #    ax.text(10, row, suunta, color='red', fontsize=9, fontweight='bold', va='center')
    #ax.set_title("Sademäärä globaaleilla ilmastosoluilla (Hadley & Ferrel)")
    #ax.axis('off')
    #plt.show()

    # 3. Piirretään lopputulos
    plt.figure(figsize=(14, 7))
    im = plt.imshow(biomi_kartta, cmap=cmap_biomit, origin='upper', vmin=0, vmax=7, extent=[-180,180,-90,90])
    shad = plt.imshow(hillshade1*landmask, cmap="gray", origin='upper', extent=[-180,180,-90,90], alpha=0.3)   
    plt.imshow(rivers1*noice, cmap="Blues_r",origin='upper', extent=[-180,180,-90,90]) 
    
    civim=plt.imshow(sivilisaatio, cmap="coolwarm_r",origin='upper', alpha=0.4, extent=[-180,180,-90,90])


    # Tehdään selkeä selite (legend) biomeille
    #plt.contour(relief, lw=2, levels=[1500], alpha=0.5, color="#100000", origin="upper",extent=[-180,180,-90,90])
    #plt.contour(suitability_rain_agriculture, lw=1, levels=[50], alpha=1.0, color="#FFFF00", origin="upper",extent=[-180,180,-90,90])
    plt.contour(aly_todennakoisyyskartta, lw=2, levels=[0.75], alpha=1.0, color="#FF0000", origin="upper",extent=[-180,180,-90,90])
    #plt.contour(cayonu_index, lw=2, levels=[0.5], alpha=1.0, color="#0000ff", origin="upper",extent=[-180,180,-90,90])
    #plt.contour(leviamisaika*landmask, lw=1.5, levels=[1000,2000,3000,4000,5000,7000,10000], alpha=1.0, color="#00ff00", origin="upper",extent=[-180,180,-90,90])
    #plt.imshow(leviamisaika*landmask,cmap="rainbow_r", alpha=0.3, origin="upper",extent=[-180,180,-90,90])

    plt.scatter([aly_lon],[aly_lat],s=120,c="black",edgecolors="gray",linewidths=1.5,zorder=12)    
    plt.text(aly_lon, aly_lat, "Human", fontsize=18) 
    plt.scatter([kulttuuri_lon],[kulttuuri_lat],s=120,c="yellow",edgecolors="green",linewidths=1.5,zorder=11)
    plt.text(kulttuuri_lon, kulttuuri_lat, "Agricult", fontsize=18) 
    plt.scatter([kulttuuri2_lon],[kulttuuri2_lat],s=120, marker="s", c="red",edgecolors="black",linewidths=1.5,zorder=11)
    plt.text(kulttuuri2_lon, kulttuuri2_lat, "Cities", fontsize=18)  
  
    #plt.scatter(civ_lons,civ_lats,s=80,c="red",edgecolors="white",linewidths=1.5,zorder=10)
    cbar = plt.colorbar(im, ticks=np.arange(8), fraction=0.03, pad=0.04)
    cbar.ax.set_yticklabels(biomi_nimet)
    plt.xticks(lons)
    plt.yticks(lats)
    plt.title("Biomes and primary civilization areas", fontsize=18)
    plt.axis('off')
    plt.show()
    quit(-1)
    
    # 4. NÄYTTÖ (Visualisointi erikseen main-funktiossa)
    # Luodaan diskreetti värimaailma (5 luokkaa)
    cmap = plt.get_cmap('terrain', 5)
    cmap.set_under('white') # NoData-arvot (-1) näkyvät valkoisina
    
    plt.figure(figsize=(10, 8))
    # vmin=1 ja vmax=5 rajaa värikartan vain voimassa oleville luokille
    im = plt.imshow(holdridge_tulos, cmap=cmap, vmin=1, vmax=5)
    #plt.xticks(lons)
    #plt.yticks(lats)    
    # Tehdään selkeä väripalkki luokkien nimillä
    cbar = plt.colorbar(im, ticks=[1, 2, 3, 4, 5])
    cbar.ax.set_yticklabels(['Aavikko', 'Steppe', 'Kuiva metsä', 'Kostea metsä', 'Tundra/Alpiininen'])
    cbar.set_label('Holdridge-luokat')
    
    plt.title('Holdridgen elämänmuotoluokitus')
    plt.xlabel('X-pikselit')
    plt.ylabel('Y-pikselit')
    plt.show()
    # 3. Kutsutaan aluohjelmaa NPP-laskentaan
 
    plt.figure(figsize=(10, 8))
    # Käytetään kasvillisuuteen sopivaa 'YlGn' (Yellow-Green) -värikarttaa
    im = plt.imshow(npp_naytettava, cmap='YlGn')
    
    # Lisätään väripalkki kuvaamaan NPP:n arvoja
    cbar = plt.colorbar(im)
    cbar.set_label('NPP ($g / m^2 / vuosi$)')
    
    plt.title('Nettoprimaarituotanto (NPP) – Miamin malli')
    plt.xlabel('X-pikselit')
    plt.ylabel('Y-pikselit')
    plt.show()

    # Tapaus 4: Avomeri, suojakeli ja vesisade
    #print(arvioi_merijaa(landmask=False, lampotila_c=3.5, sademaara_mm=5.0))
    # Valinnainen: Tallenna tulos uutena GeoTIFF-tiedostona
    # meta.update(dtype=rasterio.float32, count=1, nodata=-1)
    # with rasterio.open('npp_tulos.tif', 'w', **meta) as dst:
    #     dst.write(npp_tulos.astype(rasterio.float32), 1)
    # Valinnainen: Tallenna tulos uutena GeoTIFF-tiedostona
    # meta.update(dtype=rasterio.int16, count=1, nodata=-1)
    # with rasterio.open('holdridge_tulos.tif', 'w', **meta) as dst:
    #     dst.write(holdridge_tulos.astype(rasterio.int16), 1)    
    
    
quit(-1)

#Maailman mantereiden keskimääräinen nettoprimaarituotanto (NPP) on tutkimuksesta 
#ja mallinnuksesta riippuen noin 400 – 500 g/m²/vuosi (kuiva-aineena ilmaistuna).
# Jos tuottavuus ilmoitetaan pelkän sitoutuneen hiilen massana, 
#keskiarvo on noin 220 – 250 g C/m²/vuosi
#Koko maapallon mittakaavassa mantereet tuottavat vuosittain yhteensä 
#noin 55 – 60 miljardia tonnia (Pg) hiiltä
## mantereet tuottavat keskim 450 g/m2/v ka, joka on 4.5 tn/ha hiilenä tämä on 220 g g c/m2/v 
## alle 200 karu, 200.-600 kohtalaine ja yli 1000 hyvin rehevä

# 3. Litistetään 2D-kuvamatriisi 1D-jonoksi ja piirretään histogrammi
plt.figure(figsize=(8, 4))
# kuva_taulukko.ravel() muuttaa esim. 500x500 kuvan 250000 pikselin pitkäksi jonoksi
plt.hist(dem.ravel(), bins=256, range=(0, 256), color='gray', rwidth=1.0)

plt.title('Kuvan sävyjakauma (Histogrammi)')
plt.xlabel('Sävy (0 = musta, 255 = valkoinen)')
plt.ylabel('Pikselien määrä')
plt.xlim()

plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.show()
