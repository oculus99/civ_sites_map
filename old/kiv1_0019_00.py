
#######################################
#
## Create fractal world climate and biomes
##
## estimate from dem and climate mean values
##
## human-like specie birth location
#
## primary civilization areas
#
## simple mean t, deltat approach
# 
## ## WARNING NOTE: not all parameterrs to procude map, program may crash !!!
#
## 10.09.2026 0000.0019.00
##
######################################

import argparse

import numpy as np
import math
import heapq

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap


from scipy.ndimage import (
    gaussian_filter,generic_filter,uniform_filter,
    distance_transform_edt, map_coordinates, label
)

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from PIL import Image

import rasterio
from rasterio.enums import Resampling
import heapq

from noise import pnoise3



#generate_dem=2 ##generate dem and flood with earth ocean units
##generate_dem=1 ## generate ifractal dem type 1, if will	
generate_dem=3 ## attempt to generate Earth-like 2-element ocean/continent difference hypsometric dem
	
add_water_in_oceans=0.5 ## only in dem type 2	
will_load_image=False ## input gray scale image for dem
will_load_rasters=False ## input netcdf or geotiff dem, metere height and depth 

#input_imagename='./indata1/orogen1.png'
#input_imagename='./indata1/diku1.bmp'

input_dem_path="../data/etopo/etopo7200.tif"
input_imagename='./indata1/helliconia2.png'

##seed1=42


#seed1=333

#seed1=1246
#seed1=32
#seed1=321
#seed1=42

#seed1=88
#seed1=123
#seed1=38


#seed1=6

#seed1=3

seed1=73
#seed1=66


base_noisescale=0.6
distort_noisescale=0.75
distort_coeff=0.1


sealevel=0.5

#land_fraction=0.29
land_fraction=0.3

ocean_fraction=1-land_fraction
dem_min=-4000
dem_max=4000
delta_height=8000
sealevel_from_min=abs(dem_min)

width=360*2
height=180*2

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

## stelelr properties

star_mass=1.0
star_luminosity=1.00
star_teff=5778

## planet properites

distance_au=1.0

ecc=0.0167*1 ## eccentricity
tilt=23.44 ## axis tilt or obliquity
mvelp=102.7 ## position of axis against orbit

planet_mass_me=1
rotation_period_days=1
rotation_direction=1

planet_atmosphere_pressure=math.pow(planet_mass_me, 1.2) ## fom hat, check this
planet_atmos_co2_ppm= 280

# .....

Sk=star_luminosity/math.pow(distance_au,2)

S1=1361*Sk*math.pow((star_teff/5778), -0.8) ## S1 solar constant W m-2



orbital_period_years=math.sqrt(math.pow(distance_au,3))
rotation_period_hours=rotation_period_days*24
orbital_period_days=orbital_period_years*365.25
    

planet_radius_re=math.pow(planet_mass_me, 0.27) ## terra-like internal composition 
planet_radius=6371*planet_radius_re #3 km
gee_earths=planet_mass_me/math.pow(planet_radius_re,2)
gee_ms2=gee_earths*9.81
    

#mean_temp=15
mean_temp=15
temp_diff=70**math.sqrt(rotation_period_days)*math.sqrt(planet_radius_re)/planet_atmosphere_pressure

temp_dev=temp_diff/2
polar_temp=mean_temp-temp_dev
temp_diff=mean_temp+temp_dev

SIGMA = 5.670374419e-8
#quit(-1)


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


def load_gray_image_and_normalize(imagename):
    try:
        with Image.open(imagename) as img:
            # Otetaan leveys ja korkeus talteen
            leveys, korkeus = img.size
            
            print(f"Kuva '{imagename}' ladattu onnistuneesti.")
            print(f"Leveys: {leveys} px")
            print(f"Korkeus: {korkeus} px")

            # Tehdään kuvasta NumPy-taulukko (säilyttää 1, 3 tai 4 kanavaa)
            kuva_taulukko = np.array(img, dtype=np.float32)
            shape1=np.shape(kuva_taulukko)
            print(shape1)
            layers1=1
            if (len(shape1)==3):
                layers1=shape1[2]         
            # Katsotaan kuvan muoto (shape). Jos kanavia on useampi, shape on (korkeus, leveys, kanavat)
            if layers1 > 1:
                kanavat = kuva_taulukko.shape[2]
                kuva_taulukko2=kuva_taulukko[:,:,0]
                print(f"Kuvan kanavat (layers): {kanavat}")
            else:
                print("Kuvan kanavat (layers): 1 (Harmaasävy)")
                kuva_taulukko2=kuva_taulukko
            # Etsitään koko kuvan minimi- ja maksimiarvot
            kuva_min = np.min(kuva_taulukko2)
            kuva_max = np.max(kuva_taulukko2)
            kuva_delta = kuva_max - kuva_min
            print(kuva_min, kuva_max, kuva_delta)
            # Estetään nollalla jakaminen, jos kuva on täysin tasavärinen
            #if kuva_delta == 0:
            #    normalized_image = kuva_taulukko2 - kuva_min
            #else:
            #    # Normalisoidaan kaikki layerit kerralla välille 0.0 - 1.0
            normalized_image = (kuva_taulukko2 - kuva_min) / kuva_delta
            #plt.imshow(normalized_image)
            #plt.show()
            #quit(-1)              
            # Palautetaan normalisoitu taulukko sekä oikeat kokomuuttujat
            return normalized_image, leveys, korkeus
            
    except FileNotFoundError:
        print(f"Virhe: tiedostoa '{imagename}' ei löytynyt.")
        return None, None, None


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






def hypsometric_terrain_00(
    dem,
    landmask,
    peak_height=3000,
    sea_level=0,
    shelf_height=-200,
    deep_height=-4500,

    # Kuinka monta korkeusaluetta maalle tehdään
    land_levels=5,

    # Kuinka paljon alkuperäisestä reliefistä säilytetään
    relief=1.0,
):
    """
    Muokkaa koko planeetan DEM:n hypsometriseksi.

    dem:
        2D DEM metreinä.

    landmask:
        True = maa
        False = meri.

    peak_height:
        Maan korkein tavoitekorkeus.

    sea_level:
        Merenpinnan korkeus.

    shelf_height:
        Mannerjalustan tavoitekorkeus.

    deep_height:
        Syvänmeren tavoitekorkeus.

    land_levels:
        Kuinka moneen hypsometriseen korkeustasoon maa jaetaan.

    relief:
        0 = tasaiset alueet
        1 = alkuperäinen vaihtelu täysimääräisenä.
    """

    dem = np.asarray(dem, dtype=float)
    landmask = np.asarray(landmask, dtype=bool)

    result = np.empty_like(dem)

    # ==========================================================
    # MAA
    # ==========================================================

    land = dem[landmask]

    # DEM:n alkuperäiset kvantiilit.
    # Näillä jaetaan maa esimerkiksi:
    #
    # 0-20 % = alanko
    # 20-40 % = matala ylänkö
    # 40-60 % = ylänkö
    # 60-80 % = korkea maa
    # 80-100 % = vuoristo
    #
    q = np.linspace(0, 1, land_levels + 1)
    boundaries = np.quantile(land, q)

    land_result = np.zeros_like(land)

    for i in range(land_levels):

        lo = boundaries[i]
        hi = boundaries[i + 1]

        # Viimeiseen luokkaan kuuluu myös maksimi
        if i == land_levels - 1:
            selected = (land >= lo) & (land <= hi)
        else:
            selected = (land >= lo) & (land < hi)

        if not np.any(selected):
            continue

        # Paikallinen normalisointi 0...1
        if hi > lo:
            local = (land[selected] - lo) / (hi - lo)
        else:
            local = np.zeros(np.sum(selected))

        # ----------------------------------------------
        # Tämän korkeustason tavoitealue
        # ----------------------------------------------

        target_lo = sea_level + (
            peak_height - sea_level
        ) * (i / land_levels)

        target_hi = sea_level + (
            peak_height - sea_level
        ) * ((i + 1) / land_levels)

        # ----------------------------------------------
        # Säilytetään alkuperäistä reliefiä
        # ----------------------------------------------

        local = 0.5 + (local - 0.5) * relief

        land_result[selected] = (
            target_lo
            + local * (target_hi - target_lo)
        )

    result[landmask] = land_result

    # ==========================================================
    # MERI
    # ==========================================================

    sea = dem[~landmask]

    # Merelle tehdään kaksi pääaluetta:
    #
    # 0 ... shelf_height     mannerjalusta
    # shelf_height ... deep  syvänmeri

    # Alkuperäisen DEM:n normalisointi
    if len(sea) > 0:
        sea_min = np.percentile(sea, 1)
        sea_max = np.percentile(sea, 99)

        sea_norm = np.clip(
            (sea - sea_min) / (sea_max - sea_min),
            0,
            1
        )

        # Syvyysprofiili
        sea_result = (
            shelf_height
            + sea_norm * (deep_height - shelf_height)
        )

        result[~landmask] = sea_result

    return result


import numpy as np
from scipy.ndimage import gaussian_filter


def make_hypsometric_dem_01(
    dem,
    landmask,
    peak_height=3000.0,
    sea_level=0.0,
    shelf_height=-200.0,
    deep_height=-4500.0,

    # Maan suuren mittakaavan vaihtelu
    terrain_scale=80,

    # Paikallisen reliefin voimakkuus
    relief_strength=0.35,

    # Kuinka paljon alkuperäisen DEM:n korkeuksien jakaumaa seurataan
    height_strength=0.75,

    # Kuinka nopeasti mannerjalusta muuttuu syvänmereksi
    shelf_width=100,
    deep_width=250,
):
    """
    Luo DEM:stä pehmeästi hypsometrisesti muokatun version.

    dem:
        2D numpy array, korkeudet metreinä.

    landmask:
        True = maa
        False = meri

    terrain_scale:
        Gaussian-suodatuksen sigma pikseleinä.
        Suurempi arvo -> suurempia alanko-/ylänköalueita.

    relief_strength:
        Alkuperäisen paikallisen reliefin voimakkuus.

    height_strength:
        Kuinka voimakkaasti alkuperäisen DEM:n korkeustaso vaikuttaa
        lopputulokseen.
    """

    dem = np.asarray(dem, dtype=float)
    landmask = np.asarray(landmask, dtype=bool)

    if dem.shape != landmask.shape:
        raise ValueError(
            "dem ja landmask pitää olla saman kokoisia"
        )

    result = np.zeros_like(dem)

    # ==========================================================
    # MAA
    # ==========================================================

    land = dem[landmask]

    # Alkuperäisen maan normalisointi
    lo = np.percentile(land, 1)
    hi = np.percentile(land, 99)

    land_norm = np.clip(
        (dem - lo) / (hi - lo),
        0.0,
        1.0
    )

    # ----------------------------------------------------------
    # LOW FREQUENCY
    #
    # Tämä muodostaa suuret alanko-, ylänkö- ja vuoristoalueet.
    # ----------------------------------------------------------

    low = gaussian_filter(
        dem,
        sigma=terrain_scale
    )

    low_land = low[landmask]

    low_lo = np.percentile(low_land, 1)
    low_hi = np.percentile(low_land, 99)

    low_norm = np.clip(
        (low - low_lo) / (low_hi - low_lo),
        0.0,
        1.0
    )

    # ----------------------------------------------------------
    # SOFT HYPSONOMETRIC CURVE
    #
    # Käyrä tekee alangoista suhteellisen laajoja ja vuoristoista
    # harvinaisempia.
    # ----------------------------------------------------------

    x = low_norm

    # Smoothstep
    x = x * x * (3.0 - 2.0 * x)

    # Pieni painotus alankoihin
    x = x ** 1.25

    target_land = (
        sea_level
        + x * (peak_height - sea_level)
    )

    # ----------------------------------------------------------
    # ALKUPERÄINEN KORKEUSMALLI MUKAAN
    # ----------------------------------------------------------

    target_land = (
        target_land * height_strength
        + (
            sea_level
            + land_norm * (peak_height - sea_level)
        ) * (1.0 - height_strength)
    )

    # ----------------------------------------------------------
    # HIGH FREQUENCY
    #
    # Paikallinen reliefi säilytetään, mutta sitä vaimennetaan.
    # ----------------------------------------------------------

    high = dem - low

    high_land = high[landmask]

    # Robustisti skaalattu reliefi
    scale = np.percentile(
        np.abs(high_land),
        95
    )

    if scale > 0:
        high_normalized = high / scale
    else:
        high_normalized = np.zeros_like(high)

    target_land += (
        high_normalized
        * relief_strength
        * (peak_height - sea_level)
        * 0.15
    )

    # Ei alle merenpinnan maalla
    target_land = np.maximum(
        target_land,
        sea_level
    )

    result[landmask] = target_land[landmask]

    # ==========================================================
    # MERI
    # ==========================================================

    # Merellä käytetään alkuperäisen DEM:n suurimittakaavaista
    # syvyysrakennetta.
    sea_low = low[~landmask]

    if len(sea_low) > 0:

        sea_lo = np.percentile(sea_low, 1)
        sea_hi = np.percentile(sea_low, 99)

        sea_norm = np.clip(
            (sea_low - sea_lo) /
            (sea_hi - sea_lo),
            0.0,
            1.0
        )

        # Smoothstep
        sea_norm = (
            sea_norm
            * sea_norm
            * (3.0 - 2.0 * sea_norm)
        )

        target_sea = (
            shelf_height
            + sea_norm
            * (deep_height - shelf_height)
        )

        result[~landmask] = target_sea

    return result


import numpy as np
from scipy.ndimage import gaussian_filter


def make_hypsometric_dem_02(
    dem,
    landmask,
    peak_height=3000.0,
    sea_level=0.0,
    shelf_height=-200.0,
    deep_height=-4500.0,

    # Gaussian-skaalat pikseleinä
    continental_scale=150,
    regional_scale=50,
    local_scale=10,

    # Eri mittakaavojen vaikutus
    continental_weight=0.60,
    regional_weight=0.30,
    local_weight=0.10,

    # Alkuperäisen DEM:n paikallisen reliefin määrä
    relief_strength=0.30,

    # Hypsometrisen käyrän muoto
    elevation_exponent=1.35,

    # Meren mannerjalustan leveys
    shelf_width=100,

    # Mannerjalustan -> syvänmeren siirtymän leveys
    deep_width=250,
):
    """
    Luo DEM:stä pehmeän, monimittakaavaisen hypsometrisen planeetan.

    dem:
        2D DEM metreinä.

    landmask:
        True = maa
        False = meri.

    peak_height:
        Maan suurin tavoitekorkeus.

    sea_level:
        Merenpinta.

    shelf_height:
        Mannerjalustan tavoitekorkeus.

    deep_height:
        Syvänmeren tavoitesyvyys.

    *_scale:
        Gaussian-suodatusten koot pikseleinä.

    *_weight:
        Eri mittakaavojen suhteelliset painot.
    """

    dem = np.asarray(dem, dtype=float)
    landmask = np.asarray(landmask, dtype=bool)

    if dem.shape != landmask.shape:
        raise ValueError(
            "dem ja landmask pitää olla saman kokoisia"
        )

    result = np.empty_like(dem)

    # ==========================================================
    # 1. DEM:N ERI MITTAKAAVAT
    # ==========================================================

    continental = gaussian_filter(
        dem,
        sigma=continental_scale
    )

    regional = gaussian_filter(
        dem,
        sigma=regional_scale
    )

    local = dem - gaussian_filter(
        dem,
        sigma=local_scale
    )

    # ==========================================================
    # 2. MANTEREEN SUURIKAAVAINEN TOPOGRAFIA
    # ==========================================================

    large_terrain = (
        continental * continental_weight
        + regional * regional_weight
        + local * local_weight
    )

    land = large_terrain[landmask]

    if len(land) == 0:
        raise ValueError("landmask ei sisällä maa-alueita")

    # ==========================================================
    # 3. NORMALISOIDAAN SUURIKAAVAINEN KORKEUS
    # ==========================================================

    lo = np.percentile(land, 1)
    hi = np.percentile(land, 99)

    land_norm = np.clip(
        (large_terrain - lo) / (hi - lo),
        0.0,
        1.0
    )

    # ==========================================================
    # 4. HYPSONOMETRINEN MUUNNOS
    # ==========================================================

    # Eksponentti > 1:
    #
    #   paljon matalaa maata
    #   vähemmän korkeaa maata
    #
    # Tämä estää planeettaa olemasta liian "vuoristoinen".

    x = land_norm ** elevation_exponent

    # Vielä pehmeämpi siirtymä
    x = x * x * (3.0 - 2.0 * x)

    target_land = (
        sea_level
        + x * (peak_height - sea_level)
    )

    # ==========================================================
    # 5. PAIKALLINEN RELIEF
    # ==========================================================

    local_land = local[landmask]

    relief_scale = np.percentile(
        np.abs(local_land),
        95
    )

    if relief_scale > 0:

        local_normalized = (
            local / relief_scale
        )

        # Reliefin pitäisi pienentyä korkeimmilla alueilla
        # hieman, jotta vuoristosta ei tule liian kohinaista.

        relief_factor = (
            1.0 - 0.5 * land_norm
        )

        target_land += (
            local_normalized
            * relief_strength
            * 500.0
            * relief_factor
        )

    # Maa ei mene meren alle
    target_land = np.maximum(
        target_land,
        sea_level
    )

    result[landmask] = target_land[landmask]

    # ==========================================================
    # 6. MERI
    # ==========================================================

    sea_terrain = continental[~landmask]

    if len(sea_terrain) > 0:

        sea_lo = np.percentile(
            sea_terrain,
            1
        )

        sea_hi = np.percentile(
            sea_terrain,
            99
        )

        sea_norm = np.clip(
            (sea_terrain - sea_lo)
            / (sea_hi - sea_lo),
            0.0,
            1.0
        )

        # Mannerjalusta
        #
        # 0 -> shelf_height

        shelf_x = np.clip(
            sea_norm * shelf_width,
            0.0,
            shelf_width
        ) / shelf_width

        shelf_x = (
            shelf_x
            * shelf_x
            * (3.0 - 2.0 * shelf_x)
        )

        # Syvänmeren suunta
        deep_x = np.clip(
            (sea_norm * shelf_width - shelf_width)
            / deep_width,
            0.0,
            1.0
        )

        deep_x = (
            deep_x
            * deep_x
            * (3.0 - 2.0 * deep_x)
        )

        target_sea = np.where(
            sea_norm <= 1.0,
            shelf_height
            + deep_x
            * (deep_height - shelf_height),
            deep_height
        )

        result[~landmask] = target_sea

    return result


import numpy as np
from scipy.ndimage import gaussian_filter


def make_hypsometric_dem_03(
    dem,
    landmask,
    peak_height=3000.0,
    sea_level=0.0,
    shelf_height=-200.0,
    deep_height=-4500.0,

    # Gaussian-skaalat pikseleinä
    continental_scale=150,
    regional_scale=50,
    local_scale=10,

    # Eri mittakaavojen vaikutus
    continental_weight=0.60,
    regional_weight=0.30,
    local_weight=0.10,

    # Alkuperäisen DEM:n paikallisen reliefin määrä
    relief_strength=0.30,

    # Hypsometrisen käyrän muoto
    elevation_exponent=1.35,

    # Meren mannerjalustan leveys
    shelf_width=100,

    # Mannerjalustan -> syvänmeren siirtymän leveys
    deep_width=250,
):
    """
    Luo DEM:stä pehmeän, monimittakaavaisen hypsometrisen planeetan.

    dem:
        2D DEM metreinä.

    landmask:
        True = maa
        False = meri.

    peak_height:
        Maan suurin tavoitekorkeus.

    sea_level:
        Merenpinta.

    shelf_height:
        Mannerjalustan tavoitekorkeus.

    deep_height:
        Syvänmeren tavoitesyvyys.

    *_scale:
        Gaussian-suodatusten koot pikseleinä.

    *_weight:
        Eri mittakaavojen suhteelliset painot.
    """

    dem = np.asarray(dem, dtype=float)
    landmask = np.asarray(landmask, dtype=bool)

    if dem.shape != landmask.shape:
        raise ValueError(
            "dem ja landmask pitää olla saman kokoisia"
        )

    result = np.empty_like(dem)

    # ==========================================================
    # 1. DEM:N ERI MITTAKAAVAT
    # ==========================================================

    continental = gaussian_filter(
        dem,
        sigma=continental_scale
    )

    regional = gaussian_filter(
        dem,
        sigma=regional_scale
    )

    local = dem - gaussian_filter(
        dem,
        sigma=local_scale
    )

    # ==========================================================
    # 2. MANTEREEN SUURIKAAVAINEN TOPOGRAFIA
    # ==========================================================

    large_terrain = (
        continental * continental_weight
        + regional * regional_weight
        + local * local_weight
    )

    land = large_terrain[landmask]

    if len(land) == 0:
        raise ValueError("landmask ei sisällä maa-alueita")

    # ==========================================================
    # 3. NORMALISOIDAAN SUURIKAAVAINEN KORKEUS
    # ==========================================================

    lo = np.percentile(land, 1)
    hi = np.percentile(land, 99)

    land_norm = np.clip(
        (large_terrain - lo) / (hi - lo),
        0.0,
        1.0
    )

    # ==========================================================
    # 4. HYPSONOMETRINEN MUUNNOS
    # ==========================================================

    # Eksponentti > 1:
    #
    #   paljon matalaa maata
    #   vähemmän korkeaa maata
    #
    # Tämä estää planeettaa olemasta liian "vuoristoinen".

    x = land_norm ** elevation_exponent

    # Vielä pehmeämpi siirtymä
    x = x * x * (3.0 - 2.0 * x)

    target_land = (
        sea_level
        + x * (peak_height - sea_level)
    )

    # ==========================================================
    # 5. PAIKALLINEN RELIEF
    # ==========================================================

    local_land = local[landmask]

    relief_scale = np.percentile(
        np.abs(local_land),
        95
    )

    if relief_scale > 0:

        local_normalized = (
            local / relief_scale
        )

        # Reliefin pitäisi pienentyä korkeimmilla alueilla
        # hieman, jotta vuoristosta ei tule liian kohinaista.

        relief_factor = (
            1.0 - 0.5 * land_norm
        )

        target_land += (
            local_normalized
            * relief_strength
            * 500.0
            * relief_factor
        )

    # Maa ei mene meren alle
    target_land = np.maximum(
        target_land,
        sea_level
    )

    result[landmask] = target_land[landmask]

    # ==========================================================
    # 6. MERI
    # ==========================================================

    sea_terrain = continental[~landmask]

    if len(sea_terrain) > 0:

        sea_lo = np.percentile(
            sea_terrain,
            1
        )

        sea_hi = np.percentile(
            sea_terrain,
            99
        )

        sea_norm = np.clip(
            (sea_terrain - sea_lo)
            / (sea_hi - sea_lo),
            0.0,
            1.0
        )

        # Mannerjalusta
        #
        # 0 -> shelf_height

        shelf_x = np.clip(
            sea_norm * shelf_width,
            0.0,
            shelf_width
        ) / shelf_width

        shelf_x = (
            shelf_x
            * shelf_x
            * (3.0 - 2.0 * shelf_x)
        )

        # Syvänmeren suunta
        deep_x = np.clip(
            (sea_norm * shelf_width - shelf_width)
            / deep_width,
            0.0,
            1.0
        )

        deep_x = (
            deep_x
            * deep_x
            * (3.0 - 2.0 * deep_x)
        )

        target_sea = np.where(
            sea_norm <= 1.0,
            shelf_height
            + deep_x
            * (deep_height - shelf_height),
            deep_height
        )

        result[~landmask] = target_sea

    return result


import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt


def make_hypsometric_dem(
    dem,
    landmask,
    resolution,

    peak_height=5000.0,
    sea_level=0.0,
    shelf_depth=-200.0,
    deep_depth=-4500.0,

    shelf_width=100_000.0,
    slope_width=200_000.0,

    continental_scale=150_000.0,
    regional_scale=50_000.0,
    local_scale=10_000.0,

    continental_weight=0.60,
    regional_weight=0.30,
    local_weight=0.10,

    relief_strength=0.25,
    elevation_exponent=1.4,
):
    """
    Luo DEM:stä liukuvan hypsometrisen planeetan.

    dem:
        2D numpy array, korkeudet metreinä.

    landmask:
        True  = manner
        False = mannerjalustan ulkopuolinen meri

    resolution:
        DEM:n pikselikoko metreinä. Esim. 1000 = 1 km/pikseli.

    peak_height:
        Maan suurin tavoitekorkeus.

    shelf_depth:
        Mannerjalustan syvyys.

    deep_depth:
        Syvänmeren tavoitesyvyys.

    shelf_width:
        Mannerjalustan leveys metreinä.

    slope_width:
        Mannerreunan jyrkän rinteen leveys metreinä.

    *_scale:
        Maaston eri mittakaavat metreinä.

    relief_strength:
        Kuinka paljon alkuperäistä pienimittakaavaista reliefiä säilytetään.

    elevation_exponent:
        Hypsometrisen jakauman muoto.
        Suurempi -> enemmän matalaa maata.
    """

    dem = np.asarray(dem, dtype=float)
    landmask = np.asarray(landmask, dtype=bool)

    if dem.ndim != 2:
        raise ValueError("dem pitää olla 2D-taulukko")

    if dem.shape != landmask.shape:
        raise ValueError(
            "dem ja landmask pitää olla saman kokoisia"
        )

    if resolution <= 0:
        raise ValueError("resolution pitää olla > 0")

    # ==========================================================
    # ETÄISYYS LANDMASKIN RAJAAN
    # ==========================================================

    # Etäisyys maan sisällä lähimpään meripikseliin
    land_distance = distance_transform_edt(
        landmask,
        sampling=resolution
    )

    # Etäisyys merellä lähimpään maapikseliin
    sea_distance = distance_transform_edt(
        ~landmask,
        sampling=resolution
    )

    # ==========================================================
    # MONIMITTAKAAINEN TOPOGRAFIA
    # ==========================================================

    # Sigma muunnetaan metreistä pikseleiksi
    sigma_large = continental_scale / resolution
    sigma_medium = regional_scale / resolution
    sigma_small = local_scale / resolution

    continental = gaussian_filter(
        dem,
        sigma=sigma_large
    )

    regional = gaussian_filter(
        dem,
        sigma=sigma_medium
    )

    # Paikallinen reliefi
    local_base = gaussian_filter(
        dem,
        sigma=sigma_small
    )

    local = dem - local_base

    # Suuri + keskikokoinen + pieni mittakaava
    terrain = (
        continental * continental_weight
        + regional * regional_weight
        + local_base * local_weight
    )

    # ==========================================================
    # MAAN HYPSONOMETRIA
    # ==========================================================

    land_values = terrain[landmask]

    lo = np.percentile(land_values, 1)
    hi = np.percentile(land_values, 99)

    if hi <= lo:
        raise ValueError("DEM:n maa-alueella ei ole riittävästi vaihtelua")

    x = np.clip(
        (terrain - lo) / (hi - lo),
        0.0,
        1.0
    )

    # Painotetaan jakaumaa kohti alankoja
    x = x ** elevation_exponent

    # Pehmeä interpolointi
    x = x * x * (3.0 - 2.0 * x)

    target_land = (
        sea_level
        + x * (peak_height - sea_level)
    )

    # ==========================================================
    # PAIKALLINEN RELIEF
    # ==========================================================

    local_land = local[landmask]

    relief_scale = np.percentile(
        np.abs(local_land),
        95
    )

    if relief_scale > 0:

        local_normalized = local / relief_scale

        # Pienennetään reliefiä hieman korkealla maalla
        relief_factor = (
            1.0 - 0.4 * x
        )

        target_land += (
            local_normalized
            * relief_strength
            * 1000.0
            * relief_factor
        )

    # ==========================================================
    # MERI
    # ==========================================================

    target_sea = np.empty_like(dem)

    d = sea_distance

    # ------------------------------------------
    # Mannerjalusta
    # ------------------------------------------

    shelf_x = np.clip(
        d / shelf_width,
        0.0,
        1.0
    )

    shelf_x = (
        shelf_x
        * shelf_x
        * (3.0 - 2.0 * shelf_x)
    )

    shelf = (
        sea_level
        + (shelf_depth - sea_level)
        * shelf_x
    )

    # ------------------------------------------
    # Mannerreunan rinne
    # ------------------------------------------

    slope_x = np.clip(
        (d - shelf_width) / slope_width,
        0.0,
        1.0
    )

    slope_x = (
        slope_x
        * slope_x
        * (3.0 - 2.0 * slope_x)
    )

    slope = (
        shelf_depth
        + (deep_depth - shelf_depth)
        * slope_x
    )

    # ------------------------------------------
    # Yhdistetään
    # ------------------------------------------

    target_sea[:] = slope

    shelf_region = d <= shelf_width

    target_sea[shelf_region] = shelf[shelf_region]

    # ==========================================================
    # PAIKALLINEN MERENPOHJAN RELIEF
    # ==========================================================

    sea_local = local[~landmask]

    if len(sea_local) > 0:

        sea_relief_scale = np.percentile(
            np.abs(sea_local),
            95
        )

        if sea_relief_scale > 0:

            sea_relief = (
                local[~landmask]
                / sea_relief_scale
            )

            # Merellä reliefi pienenee syvemmälle mentäessä
            relief_factor = np.exp(
                -sea_distance[~landmask]
                / 500_000.0
            )

            target_sea[~landmask] += (
                sea_relief
                * relief_strength
                * 300.0
                * relief_factor
            )

    # ==========================================================
    # LOPULLINEN DEM
    # ==========================================================

    result = np.empty_like(dem)

    result[landmask] = target_land[landmask]
    result[~landmask] = target_sea[~landmask]

    return result



def calculate_ocean_currents_12_origo0(
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
    Laskee yksinkertaistetun merivirtakentän 12 kuukaudelle.

    dem:
        [height, width]
        Maaston korkeus metreinä.
        Merellä arvot < sea_level.

    wind_x, wind_y, wind_z:
        [12, height, width]
        Tuulikenttä m/s.

    temperature:
        [12, height, width]
        Lämpötila, esim. °C.

    precipitation:
        [12, height, width]
        Sademäärä, esim. mm/vuosi tai mm/kk.

    Palauttaa:
        current_x:
            [12, height, width]

        current_y:
            [12, height, width]

        current_z:
            [12, height, width]

        upwelling:
            [12, height, width]
    """

    # ---------------------------------------------------------
    # 1. NUMPY-TAULUKOIKSI
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # 2. DIMENSION TARKISTUKSET
    # ---------------------------------------------------------

    if dem.ndim != 2:
        raise ValueError(
            "dem pitää olla muodossa [height, width]"
        )

    h, w = dem.shape

    expected_shape = (12, h, w)

    for name, arr in [
        ("wind_x", wind_x),
        ("wind_y", wind_y),
        ("wind_z", wind_z),
        ("temperature", temperature),
        ("precipitation", precipitation),
    ]:
        if arr.shape != expected_shape:
            raise ValueError(
                f"{name} pitää olla muodossa "
                f"[12, height, width], mutta shape on {arr.shape}"
            )

    # ---------------------------------------------------------
    # 3. MERI / MAA
    # ---------------------------------------------------------

    ocean = dem < sea_level

    # Syvyys metreinä
    depth = np.maximum(
        sea_level - dem,
        0.0
    )

    # Normalisoitu syvyys
    depth_norm = np.clip(
        depth / 5000.0,
        0.0,
        1.0
    )

    # ---------------------------------------------------------
    # 4. LATITUUDI
    # ---------------------------------------------------------

    # Gridin oletetaan kattavan -90 ... +90
    latitude = np.linspace(
        -90.0,
        90.0,
        h
    )

    latitude_rad = np.radians(latitude)

    # [height, 1]
    lat2d = latitude_rad[:, None]

    # ---------------------------------------------------------
    # 5. PLANEETAN PYÖRIMINEN
    # ---------------------------------------------------------

    rotation_period_sec = (
        rotation_period_hours * 3600.0
    )

    omega = (
        2.0 * np.pi
        / rotation_period_sec
    )

    # Coriolis-parametri
    #
    # Shape:
    # [height, 1]
    #
    # Se voidaan automaattisesti broadcastata
    # muotoon [height, width].

    f = (
        2.0
        * rotation_direction
        * omega
        * np.sin(lat2d)
    )

    # ---------------------------------------------------------
    # 6. TUULEN AIHEUTTAMA PINTAVIRTA
    # ---------------------------------------------------------

    # [12, height, width]

    current_x = (
        wind_x
        * wind_strength
    )

    current_y = (
        wind_y
        * wind_strength
    )

    current_z = (
        wind_z
        * wind_strength
    )

    # ---------------------------------------------------------
    # 7. CORIOLIS
    # ---------------------------------------------------------

    cx = (
        -f
        * current_y
    )

    cy = (
        f
        * current_x
    )

    current_x += (
        cx
        * coriolis_strength
    )

    current_y += (
        cy
        * coriolis_strength
    )

    # ---------------------------------------------------------
    # 8. LÄMPÖTILAN VAIKUTUS
    # ---------------------------------------------------------

    # Lasketaan jokaiselle kuukaudelle oma
    # valtameren keskilämpötila.
    #
    # temperature:
    # [12, H, W]
    #
    # ocean:
    # [H, W]
    #
    # Lopputulos:
    # [12]

    ocean_count = np.sum(ocean)

    if ocean_count == 0:
        raise ValueError(
            "DEM ei sisällä yhtään merialuetta."
        )

    temp_ocean = np.where(
        ocean[None, :, :],
        temperature,
        np.nan
    )

    monthly_temp_mean = (
        np.nanmean(
            temp_ocean,
            axis=(1, 2)
        )
    )

    # [12, 1, 1]
    monthly_temp_mean = (
        monthly_temp_mean[:, None, None]
    )

    # [12, H, W]
    temp_anomaly = (
        temperature
        - monthly_temp_mean
    )

    # Lämmin vesi pyrkii hieman kohti
    # päiväntasaajaa.
    #
    # lat2d:
    # [H, 1]
    #
    # broadcasting -> [12, H, W]

    temp_force_y = (
        -temp_anomaly
        * temperature_strength
        * np.sin(lat2d)
    )

    current_y += temp_force_y

    # ---------------------------------------------------------
    # 9. SADEMÄÄRÄ / SUOLAISUUDEN APPROKSIMAATIO
    # ---------------------------------------------------------

    rain_ocean = np.where(
        ocean[None, :, :],
        precipitation,
        np.nan
    )

    # Jokaiselle kuukaudelle oma keskiarvo
    monthly_rain_mean = (
        np.nanmean(
            rain_ocean,
            axis=(1, 2)
        )
    )

    # [12, 1, 1]
    monthly_rain_mean = (
        monthly_rain_mean[:, None, None]
    )

    # [12, H, W]
    rain_anomaly = (
        precipitation
        - monthly_rain_mean
    )

    # Paljon sadetta -> makeampi / kevyempi vesi.
    #
    # Tämä on edelleen hyvin yksinkertainen
    # approksimaatio.

    current_z += (
        -rain_anomaly
        * rain_strength
    )

    # ---------------------------------------------------------
    # 10. SYVYYDEN VAIKUTUS
    # ---------------------------------------------------------

    depth_factor = (
        1.0
        + depth_norm
        * bathymetry_strength
    )

    # depth_factor:
    # [H, W]
    #
    # broadcasting:
    # [12, H, W]

    current_x *= depth_factor
    current_y *= depth_factor

    # ---------------------------------------------------------
    # 11. RANNIKON SUUNTAINEN VIRTAUS
    # ---------------------------------------------------------

    gy, gx = np.gradient(dem)

    slope = np.sqrt(
        gx * gx
        + gy * gy
    ) + 1e-12

    # Rannikon tangenttivektori

    coast_tx = (
        -gy / slope
    )

    coast_ty = (
        gx / slope
    )

    # Rannikon vaikutus matalassa vedessä

    coastal_weight = np.exp(
        -depth / 500.0
    )

    # Projektio rannikon suuntaan
    #
    # [12,H,W] * [H,W]
    # -> [12,H,W]

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
    # 12. KUMPUAMINEN
    # ---------------------------------------------------------

    # Rannikon normaali

    coast_nx = (
        gx / slope
    )

    coast_ny = (
        gy / slope
    )

    # Tuulen komponentti rannikon normaalin
    # suunnassa

    offshore_wind = (
        wind_x * coast_nx
        + wind_y * coast_ny
    )

    # Kumpuaminen

    upwelling = (
        offshore_wind
        * coastal_weight
        * upwelling_strength
    )

    # Vain merellä

    upwelling *= ocean[None, :, :]

    # Vertical velocity

    current_z += upwelling

    # ---------------------------------------------------------
    # 13. MAA-ALUEET NOLLAKSI
    # ---------------------------------------------------------

    ocean_3d = ocean[None, :, :]

    current_x = np.where(
        ocean_3d,
        current_x,
        0.0
    )

    current_y = np.where(
        ocean_3d,
        current_y,
        0.0
    )

    current_z = np.where(
        ocean_3d,
        current_z,
        0.0
    )

    upwelling = np.where(
        ocean_3d,
        upwelling,
        0.0
    )

    # ---------------------------------------------------------
    # 14. NANIT / INFINIT
    # ---------------------------------------------------------

    current_x = np.nan_to_num(
        current_x
    )

    current_y = np.nan_to_num(
        current_y
    )

    current_z = np.nan_to_num(
        current_z
    )

    upwelling = np.nan_to_num(
        upwelling
    )

    return (
        current_x,
        current_y,
        current_z,
        upwelling,
    )


def calculate_ocean_currents_12(
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

    # Pintavirta
    wind_strength=0.03,
    temperature_strength=0.002,
    rain_strength=0.0005,
    coriolis_strength=1.0,
    bathymetry_strength=0.15,

    # Kumpuaminen
    upwelling_strength=0.02,

    # Syvävirta
    deep_current_strength=0.015,
    density_strength=0.08,
    deep_coriolis_strength=0.5,

    # Meren lämpötila
    initial_sea_temperature=None,
    air_temperature_strength=0.015,
    current_heat_strength=0.08,
    deep_heat_strength=0.04,
    upwelling_cooling_strength=0.15,
    ocean_mixing_strength=0.03,

    # Kylmä / lämmin merivirta
    cold_current_threshold=2.0,
    warm_current_threshold=2.0,

    # Syvyydet
    deep_start_depth=500.0,
    deep_full_depth=2000.0,
):
    """
    Laskee yksinkertaistetun valtameren pintavirta-, syvävirta-
    ja meriveden lämpötilamallin 12 kuukaudelle.

    Parametrit
    ----------
    dem:
        [H, W]
        Maaston korkeus metreinä.
        Merellä arvot < sea_level.

    wind_x, wind_y, wind_z:
        [12, H, W]
        Tuulikenttä m/s.

    temperature:
        [12, H, W]
        Ilman lämpötila, esim. Celsius.

    precipitation:
        [12, H, W]
        Sademäärä.

    planet_radius_km:
        Planeetan säde kilometreinä.

    rotation_period_hours:
        Planeetan pyörähdysaika tunneissa.

    rotation_direction:
        1 tai -1.

    initial_sea_temperature:
        [H, W] tai None.

        Jos None, alkumeriveden lämpötila muodostetaan
        ensimmäisen kuukauden ilman lämpötilasta.

    Palauttaa
    --------
    current_x:
        [12, H, W]
        Pintavirran X-komponentti.

    current_y:
        [12, H, W]
        Pintavirran Y-komponentti.

    current_z:
        [12, H, W]
        Pintavirran Z-komponentti.

    upwelling:
        [12, H, W]
        Kumpuamisen voimakkuus.

    deep_x:
        [12, H, W]
        Syvävirran X-komponentti.

    deep_y:
        [12, H, W]
        Syvävirran Y-komponentti.

    deep_z:
        [12, H, W]
        Syvävirran Z-komponentti.

    sea_temperature:
        [12, H, W]
        Simuloitu meriveden lämpötila.

    cold_current_mask:
        [12, H, W]
        0...1, kylmän merivirran voimakkuus.

    warm_current_mask:
        [12, H, W]
        0...1, lämpimän merivirran voimakkuus.
    """

    # =========================================================
    # 1. NUMPY-TAULUKOT
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

    # =========================================================
    # 2. DIMENSION TARKISTUKSET
    # =========================================================

    if dem.ndim != 2:
        raise ValueError(
            "dem pitää olla muodossa [height, width]"
        )

    h, w = dem.shape

    expected_shape = (12, h, w)

    for name, arr in [
        ("wind_x", wind_x),
        ("wind_y", wind_y),
        ("wind_z", wind_z),
        ("temperature", temperature),
        ("precipitation", precipitation),
    ]:
        if arr.shape != expected_shape:
            raise ValueError(
                f"{name} pitää olla muodossa "
                f"[12, height, width], mutta shape on {arr.shape}"
            )

    # =========================================================
    # 3. MERI / MAA
    # =========================================================

    ocean = dem < sea_level

    ocean_3d = ocean[None, :, :]

    ocean_count = np.sum(ocean)

    if ocean_count == 0:
        raise ValueError(
            "DEM ei sisällä yhtään merialuetta."
        )

    # Syvyys metreinä
    depth = np.maximum(
        sea_level - dem,
        0.0
    )

    # =========================================================
    # 4. SYVYYDEN NORMALISOINTI
    # =========================================================

    depth_norm = np.clip(
        depth / 5000.0,
        0.0,
        1.0
    )

    # Syvävirran painotus.
    #
    # 0 m -> lähes ei syvävirtaa
    # 500 m -> alkaa muodostua
    # 2000+ m -> täysi vaikutus

    deep_weight = np.clip(
        (depth - deep_start_depth)
        / (
            deep_full_depth
            - deep_start_depth
        ),
        0.0,
        1.0
    )

    # =========================================================
    # 5. LATITUUDI
    # =========================================================

    latitude = np.linspace(
        -90.0,
        90.0,
        h
    )

    latitude_rad = np.radians(
        latitude
    )

    lat2d = latitude_rad[:, None]

    # =========================================================
    # 6. PLANEETAN PYÖRIMINEN
    # =========================================================

    rotation_period_sec = (
        rotation_period_hours
        * 3600.0
    )

    if rotation_period_sec <= 0:
        raise ValueError(
            "rotation_period_hours pitää olla > 0"
        )

    omega = (
        2.0
        * np.pi
        / rotation_period_sec
    )

    f = (
        2.0
        * rotation_direction
        * omega
        * np.sin(lat2d)
    )

    # =========================================================
    # 7. RANNIKON GEOMETRIA
    # =========================================================

    gy, gx = np.gradient(dem)

    slope = (
        np.sqrt(
            gx * gx
            + gy * gy
        )
        + 1e-12
    )

    coast_tx = (
        -gy / slope
    )

    coast_ty = (
        gx / slope
    )

    coast_nx = (
        gx / slope
    )

    coast_ny = (
        gy / slope
    )

    # Rannikon vaikutus vähenee syvyyden mukana.

    coastal_weight = np.exp(
        -depth / 500.0
    )

    # =========================================================
    # 8. TULOSARRAYT
    # =========================================================

    current_x = np.zeros(
        expected_shape,
        dtype=np.float64
    )

    current_y = np.zeros(
        expected_shape,
        dtype=np.float64
    )

    current_z = np.zeros(
        expected_shape,
        dtype=np.float64
    )

    deep_x = np.zeros(
        expected_shape,
        dtype=np.float64
    )

    deep_y = np.zeros(
        expected_shape,
        dtype=np.float64
    )

    deep_z = np.zeros(
        expected_shape,
        dtype=np.float64
    )

    upwelling = np.zeros(
        expected_shape,
        dtype=np.float64
    )

    sea_temperature = np.zeros(
        expected_shape,
        dtype=np.float64
    )

    cold_current_mask = np.zeros(
        expected_shape,
        dtype=np.float64
    )

    warm_current_mask = np.zeros(
        expected_shape,
        dtype=np.float64
    )

    # =========================================================
    # 9. ALKUMEREN LÄMPÖTILA
    # =========================================================

    if initial_sea_temperature is None:

        # Meri reagoi ilmaan hitaasti.
        #
        # Alkuperäinen meriveden lämpötila asetetaan
        # ensimmäisen kuukauden ilman lämpötilan ympärille.

        sea_temp = (
            temperature[0]
            * 0.8
        )

    else:

        sea_temp = np.asarray(
            initial_sea_temperature,
            dtype=np.float64
        )

        if sea_temp.shape != (h, w):
            raise ValueError(
                "initial_sea_temperature pitää olla "
                "muodossa [height, width]"
            )

    # Maa ei tarvitse meriveden lämpötilaa.

    sea_temp = np.where(
        ocean,
        sea_temp,
        0.0
    )

    # =========================================================
    # 10. KUUKAUSISIMULAATIO
    # =========================================================

    for month in range(12):

        air_temp = temperature[month]

        rain = precipitation[month]

        wx = wind_x[month]
        wy = wind_y[month]
        wz = wind_z[month]

        # =====================================================
        # 10.1 PINTAVIRTA
        # =====================================================

        cx = (
            wx
            * wind_strength
        )

        cy = (
            wy
            * wind_strength
        )

        cz = (
            wz
            * wind_strength
        )

        # =====================================================
        # 10.2 CORIOLIS
        # =====================================================

        coriolis_x = (
            -f
            * cy
        )

        coriolis_y = (
            f
            * cx
        )

        cx += (
            coriolis_x
            * coriolis_strength
        )

        cy += (
            coriolis_y
            * coriolis_strength
        )

        # =====================================================
        # 10.3 LÄMPÖTILAAN LIITTYVÄ PINTAVIRTA
        # =====================================================

        ocean_temp_mean = np.nanmean(
            np.where(
                ocean,
                sea_temp,
                np.nan
            )
        )

        temp_anomaly = (
            sea_temp
            - ocean_temp_mean
        )

        temp_force_y = (
            -temp_anomaly
            * temperature_strength
            * np.sin(lat2d)
        )

        cy += temp_force_y

        # =====================================================
        # 10.4 SYVYYSVAIKUTUS
        # =====================================================

        depth_factor = (
            1.0
            + depth_norm
            * bathymetry_strength
        )

        cx *= depth_factor
        cy *= depth_factor

        # =====================================================
        # 10.5 RANNIKON SUUNTAINEN VIRTA
        # =====================================================

        coastal_flow = (
            cx * coast_tx
            + cy * coast_ty
        )

        cx += (
            coast_tx
            * coastal_flow
            * coastal_weight
            * 0.15
        )

        cy += (
            coast_ty
            * coastal_flow
            * coastal_weight
            * 0.15
        )

        # =====================================================
        # 10.6 KUMPUAMINEN
        # =====================================================

        offshore_wind = (
            wx * coast_nx
            + wy * coast_ny
        )

        month_upwelling = (
            offshore_wind
            * coastal_weight
            * upwelling_strength
        )

        month_upwelling *= ocean

        # =====================================================
        # 10.7 SADE -> SUOLAISUUSAPPROKSIMAATIO
        # =====================================================

        rain_mean = np.nanmean(
            np.where(
                ocean,
                rain,
                np.nan
            )
        )

        rain_anomaly = (
            rain
            - rain_mean
        )

        # Positiivinen rain_anomaly =
        # enemmän makeaa vettä =
        # pienempi tiheys.

        freshwater_density_effect = (
            -rain_anomaly
            * rain_strength
        )

        # =====================================================
        # 10.8 LÄMPÖTILAN TIHEYSVAIKUTUS
        # =====================================================

        # Kylmä vesi on tiheämpää.
        #
        # temp_density:
        # positiivinen -> tiheämpi
        # negatiivinen -> kevyempi

        temp_density = (
            -(sea_temp - 4.0)
            * density_strength
        )

        density_field = (
            temp_density
            + freshwater_density_effect
        )

        # =====================================================
        # 10.9 SYVÄVIRTA
        # =====================================================

        # Syvävesi pyrkii liikkumaan tiheysgradientin
        # suuntaisesti.

        density_gy, density_gx = np.gradient(
            density_field
        )

        dx = (
            density_gx
            * deep_current_strength
            * deep_weight
        )

        dy = (
            density_gy
            * deep_current_strength
            * deep_weight
        )

        # Syvävirran Coriolis

        dcx = (
            -f
            * dy
            * deep_coriolis_strength
        )

        dcy = (
            f
            * dx
            * deep_coriolis_strength
        )

        dx += dcx
        dy += dcy

        # =====================================================
        # 10.10 SYVÄVIRRAN VERTIKAALINEN KOMPONENTTI
        # =====================================================

        # Tiheä kylmä vesi pyrkii painumaan.

        density_positive = np.maximum(
            density_field,
            0.0
        )

        dz = (
            density_positive
            * deep_current_strength
            * deep_weight
        )

        # Kumpuaminen toimii vastakkaiseen suuntaan.

        dz -= (
            month_upwelling
            * deep_current_strength
        )

        # =====================================================
        # 10.11 KYLMIEN / LÄMPIMIEN VIRTAUSTEN MASKIT
        # =====================================================

        # Virtaavan veden lämpötila suhteessa
        # ympäröivään ilmaan.

        temperature_difference = (
            sea_temp
            - air_temp
        )

        cold_mask = np.clip(
            -temperature_difference
            / cold_current_threshold,
            0.0,
            1.0
        )

        warm_mask = np.clip(
            temperature_difference
            / warm_current_threshold,
            0.0,
            1.0
        )

        # Vain merellä.

        cold_mask *= ocean
        warm_mask *= ocean

        # =====================================================
        # 10.12 PINTAVIRTA MUUTTAA MEREN LÄMPÖTILAA
        # =====================================================

        horizontal_speed = np.sqrt(
            cx * cx
            + cy * cy
        )

        # Virran lämpövaikutus.
        #
        # Lämmin virta -> lämmittää
        # Kylmä virta -> viilentää

        current_heat = (
            warm_mask
            - cold_mask
        )

        current_heat *= (
            horizontal_speed
            * current_heat_strength
        )

        # =====================================================
        # 10.13 ILMAN JA MEREN LÄMPÖVAIHTO
        # =====================================================

        air_exchange = (
            air_temp
            - sea_temp
        )

        air_exchange *= (
            air_temperature_strength
        )

        # =====================================================
        # 10.14 SYVÄVEDEN LÄMPÖTILA
        # =====================================================

        # Syvä vesi oletetaan keskimäärin kylmemmäksi.
        #
        # Tätä käytetään yksinkertaistettuna reservoirina.

        deep_reference_temperature = (
            np.clip(
                sea_temp
                - 8.0,
                -2.0,
                20.0
            )
        )

        deep_temperature_effect = (
            deep_reference_temperature
            - sea_temp
        )

        deep_temperature_effect *= (
            deep_heat_strength
            * deep_weight
        )

        # =====================================================
        # 10.15 KUMPUAMISEN LÄMPÖVAIKUTUS
        # =====================================================

        # Kumpuaminen tuo syvää, yleensä kylmempää vettä
        # pintaan.

        upwelling_temperature_effect = (
            deep_temperature_effect
            * np.abs(month_upwelling)
            * upwelling_cooling_strength
        )

        # =====================================================
        # 10.16 SEKOITTUMINEN
        # =====================================================

        neighboring_temperature = (
            np.zeros_like(sea_temp)
        )

        neighboring_temperature += (
            np.roll(sea_temp, 1, axis=0)
        )

        neighboring_temperature += (
            np.roll(sea_temp, -1, axis=0)
        )

        neighboring_temperature += (
            np.roll(sea_temp, 1, axis=1)
        )

        neighboring_temperature += (
            np.roll(sea_temp, -1, axis=1)
        )

        neighboring_temperature *= 0.25

        mixing = (
            neighboring_temperature
            - sea_temp
        )

        mixing *= ocean_mixing_strength

        # =====================================================
        # 10.17 PÄIVITÄ MEREN LÄMPÖTILA
        # =====================================================

        sea_temp = (
            sea_temp
            + air_exchange
            + current_heat
            + deep_temperature_effect
            + upwelling_temperature_effect
            + mixing
        )

        # Fysikaalisesti järkevä karkea raja.

        sea_temp = np.clip(
            sea_temp,
            -2.0,
            40.0
        )

        # Maa nollaksi.

        sea_temp = np.where(
            ocean,
            sea_temp,
            0.0
        )

        # =====================================================
        # 10.18 TALLENNA KUUKAUSI
        # =====================================================

        current_x[month] = np.where(
            ocean,
            cx,
            0.0
        )

        current_y[month] = np.where(
            ocean,
            cy,
            0.0
        )

        current_z[month] = np.where(
            ocean,
            cz + month_upwelling,
            0.0
        )

        deep_x[month] = np.where(
            ocean,
            dx,
            0.0
        )

        deep_y[month] = np.where(
            ocean,
            dy,
            0.0
        )

        deep_z[month] = np.where(
            ocean,
            dz,
            0.0
        )

        upwelling[month] = np.where(
            ocean,
            month_upwelling,
            0.0
        )

        sea_temperature[month] = sea_temp

        cold_current_mask[month] = np.where(
            ocean,
            cold_mask,
            0.0
        )

        warm_current_mask[month] = np.where(
            ocean,
            warm_mask,
            0.0
        )

    # =========================================================
    # 11. NAN / INFINIT
    # =========================================================

    current_x = np.nan_to_num(
        current_x
    )

    current_y = np.nan_to_num(
        current_y
    )

    current_z = np.nan_to_num(
        current_z
    )

    deep_x = np.nan_to_num(
        deep_x
    )

    deep_y = np.nan_to_num(
        deep_y
    )

    deep_z = np.nan_to_num(
        deep_z
    )

    upwelling = np.nan_to_num(
        upwelling
    )

    sea_temperature = np.nan_to_num(
        sea_temperature
    )

    cold_current_mask = np.nan_to_num(
        cold_current_mask
    )

    warm_current_mask = np.nan_to_num(
        warm_current_mask
    )

    return (
        current_x,
        current_y,
        current_z,
        upwelling,

        deep_x,
        deep_y,
        deep_z,

        sea_temperature,

        cold_current_mask,
        warm_current_mask,
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


def transform_dem(
    dem,
    delta_height,
    land_fraction=0.29,
    debug=False
):
    """
    Muuntaa fraktaali-DEM:n korkeuksiksi ja määrittää
    merenpinnan tulvittamalla DEM:n alimmasta pisteestä ylöspäin.

    Parametrit
    ----------
    dem : np.ndarray
        2D DEM, jonka arvot ovat välillä [0, 1].
        Muoto: (height, width)

        DEM:n oletetaan kattavan koko pallon:
            longitude = [-180, 180]
            latitude  = [-90, 90]

    delta_height : float
        Alkuperäisen DEM:n kokonaiskorkeusero metreinä.
        DEM skaalataan välille:
            0 ... delta_height

    land_fraction : float
        Haluttu maan pinta-alaosuus.
        Esimerkiksi 0.29 = 29 %.

    debug : bool
        Tulostetaanko laskennan debug-tiedot.

    Palauttaa
    ----------
    transformed_dem : np.ndarray
        Lopullinen DEM merenpinnan suhteen:
            meri < 0
            merenpinta = 0
            maa > 0

    dem_min : float
        Lopullisen DEM:n alin kohta.

    dem_max : float
        Lopullisen DEM:n korkein kohta.

    delta_dem : float
        Koko DEM:n korkeusero:
            dem_max - dem_min

    sealevel_from_min : float
        Merenpinnan korkeus alkuperäisestä abyssista mitattuna.
    """

    # ============================================================
    # 1. Tarkistukset
    # ============================================================

    if dem.ndim != 2:
        raise ValueError("DEM:n pitää olla 2-ulotteinen.")

    if not (0.0 < land_fraction < 1.0):
        raise ValueError(
            "land_fraction pitää olla välillä 0 ... 1."
        )

    if delta_height <= 0:
        raise ValueError(
            "delta_height pitää olla positiivinen."
        )

    if not np.all(np.isfinite(dem)):
        raise ValueError(
            "DEM sisältää NaN- tai inf-arvoja."
        )

    # ============================================================
    # 2. Alkuperäisen fraktaali-DEMin tiedot
    # ============================================================

    dem0_min = np.min(dem)
    dem0_max = np.max(dem)

    if debug:
        print()
        print("============================================================")
        print("INPUT DEM")
        print("============================================================")
        print(f"shape              : {dem.shape}")
        print(f"dtype              : {dem.dtype}")
        print(f"input min          : {dem0_min:.6f}")
        print(f"input max          : {dem0_max:.6f}")
        print(f"input mean         : {np.mean(dem):.6f}")
        print(f"input std          : {np.std(dem):.6f}")
        print(f"delta_height       : {delta_height:.3f}")
        print(f"requested land     : {land_fraction:.6f}")

    # ============================================================
    # 3. Skaalataan fraktaali [0, 1]
    #
    #    -> [0, delta_height]
    #
    # Tässä vaiheessa:
    #
    #     0 m = DEM:n abyss
    #     delta_height = korkein kohta
    # ============================================================

    if dem0_max == dem0_min:
        raise ValueError(
            "DEM on vakioarvoinen; skaalausta ei voida tehdä."
        )

    scaled_dem = (
        (dem - dem0_min)
        / (dem0_max - dem0_min)
        * delta_height
    )

    # ============================================================
    # 4. Skaalatun DEM:n min/max
    # ============================================================

    scaled_min = np.min(scaled_dem)
    scaled_max = np.max(scaled_dem)

    if debug:
        print()
        print("------------------------------------------------------------")
        print("SCALED DEM")
        print("------------------------------------------------------------")
        print(f"scaled min         : {scaled_min:.6f} m")
        print(f"scaled max         : {scaled_max:.6f} m")
        print(
            f"scaled delta       : "
            f"{scaled_max - scaled_min:.6f} m"
        )

    # ============================================================
    # 5. Maapallon pikselien pinta-alapainot
    #
    # Lat/lon-ruudukossa pikselit eivät ole saman pinta-alan
    # kokoisia.
    #
    # Pinta-ala on verrannollinen:
    #
    #     cos(latitude)
    #
    # ============================================================

    height, width = scaled_dem.shape

    # Pikselien keskileveydet
    lat = (
        np.linspace(
            -90.0,
            90.0,
            height,
            endpoint=False
        )
        + 90.0 / height
    )

    row_weights = np.cos(
        np.deg2rad(lat)
    )

    pixel_weights = np.broadcast_to(
        row_weights[:, None],
        scaled_dem.shape
    )

    if debug:
        print()
        print("------------------------------------------------------------")
        print("GRID")
        print("------------------------------------------------------------")
        print(f"height             : {height}")
        print(f"width              : {width}")
        print(f"latitude min       : {lat.min():.6f}")
        print(f"latitude max       : {lat.max():.6f}")
        print(f"weight min         : {row_weights.min():.8f}")
        print(f"weight max         : {row_weights.max():.8f}")

    # ============================================================
    # 6. Muutetaan DEM ja pinta-alat yksiulotteisiksi
    # ============================================================

    elevations = scaled_dem.ravel()
    weights = pixel_weights.ravel()

    # ============================================================
    # 7. Järjestetään pisteet korkeuden mukaan
    #
    # Tämä on "tulvaus":
    #
    # aloitetaan alimmasta pisteestä ja nostetaan vedenpintaa
    # ylöspäin.
    # ============================================================

    order = np.argsort(elevations)

    sorted_elevations = elevations[order]
    sorted_weights = weights[order]

    # ============================================================
    # 8. Kumulatiivinen pinta-ala
    #
    # cumulative_area[i] =
    #     kaikkien pisteiden pinta-ala,
    #     jotka ovat korkeudella <= sorted_elevations[i]
    # ============================================================

    cumulative_area = np.cumsum(
        sorted_weights
    )

    total_area = cumulative_area[-1]

    # ============================================================
    # 9. Haluttu vesiosuus
    #
    # Jos maata halutaan esim. 29 %:
    #
    #     vettä = 71 %
    #
    # ============================================================

    water_fraction = 1.0 - land_fraction

    target_water_area = (
        water_fraction * total_area
    )

    if debug:
        print()
        print("------------------------------------------------------------")
        print("FLOODING")
        print("------------------------------------------------------------")
        print(f"total area         : {total_area:.6f}")
        print(f"land fraction      : {land_fraction:.6f}")
        print(f"water fraction     : {water_fraction:.6f}")
        print(
            f"target water area  : "
            f"{target_water_area:.6f}"
        )

    # ============================================================
    # 10. Etsitään merenpinnan korkeus
    #
    # Kun veden alle jää haluttu pinta-ala,
    # tämän pisteen korkeus on merenpinta.
    # ============================================================

    index = np.searchsorted(
        cumulative_area,
        target_water_area,
        side="left"
    )

    index = min(
        index,
        len(sorted_elevations) - 1
    )

    sealevel_from_min = sorted_elevations[index]

    # ============================================================
    # 11. Tarkistetaan saavutettu pinta-alaosuus
    # ============================================================

    actual_water_area = cumulative_area[index]

    actual_water_fraction = (
        actual_water_area / total_area
    )

    actual_land_fraction = (
        1.0 - actual_water_fraction
    )

    if debug:
        print()
        print("------------------------------------------------------------")
        print("SEA LEVEL")
        print("------------------------------------------------------------")
        print(
            f"flood index        : {index:,}"
        )
        print(
            f"sealevel_from_min  : "
            f"{sealevel_from_min:.6f} m"
        )
        print(
            f"actual water       : "
            f"{actual_water_fraction:.8f}"
        )
        print(
            f"actual land        : "
            f"{actual_land_fraction:.8f}"
        )
        print(
            f"requested land     : "
            f"{land_fraction:.8f}"
        )
        print(
            f"land error         : "
            f"{actual_land_fraction - land_fraction:+.8f}"
        )

    # ============================================================
    # 12. SIIRRETÄÄN DEM MERENPINNAN SUHTEEN
    #
    # Tämä on olennainen kohta:
    #
    #     uusi DEM = vanha DEM - merenpinta
    #
    # Jolloin:
    #
    #     meri       < 0
    #     meripinta  = 0
    #     maa        > 0
    # ============================================================

    transformed_dem = (
        scaled_dem - sealevel_from_min
    )

    # ============================================================
    # 13. Lopullisen DEM:n min/max
    # ============================================================

    dem_min = np.min(
        transformed_dem
    )

    dem_max = np.max(
        transformed_dem
    )

    delta_dem = (
        dem_max - dem_min
    )

    # ============================================================
    # 14. Lopputarkistukset
    # ============================================================

    if debug:
        print()
        print("------------------------------------------------------------")
        print("FINAL DEM")
        print("------------------------------------------------------------")
        print(f"dem_min            : {dem_min:.6f} m")
        print(f"dem_max            : {dem_max:.6f} m")
        print(f"delta_dem          : {delta_dem:.6f} m")
        print(
            f"sealevel_from_min  : "
            f"{sealevel_from_min:.6f} m"
        )

        print()
        print("------------------------------------------------------------")
        print("CHECKS")
        print("------------------------------------------------------------")

        # Merenpinnan pitäisi olla käytännössä 0
        sea_mask = (
            np.abs(transformed_dem) < 1e-10
        )

        print(
            f"sea level pixels   : "
            f"{np.sum(sea_mask):,}"
        )

        print(
            f"final min check    : "
            f"{np.min(transformed_dem):.6f}"
        )

        print(
            f"final max check    : "
            f"{np.max(transformed_dem):.6f}"
        )

        print(
            f"delta check        : "
            f"{dem_max - dem_min:.6f}"
        )

        print()
        print("============================================================")

    # ============================================================
    # 15. Palautus
    # ============================================================

    return (
        transformed_dem,
        dem_min,
        dem_max,
        delta_dem,
        sealevel_from_min,
    )



import numpy as np


def flood_planet_with_ocean(
    dem,
    delta_height,
    add_water_in_earth_ocean_units,
    planet_radius_km,
    debug=False,
):
    """
    Tulvittaa planeetan lisäämällä sille valtamerta
    Maan valtameren tilavuuden yksikköinä.

    Parametrit
    ----------
    dem : np.ndarray
        2D DEM metreinä.

        DEM:n oletetaan kattavan koko pallon:

            longitude = [-180, 180]
            latitude  = [-90, 90]

        DEM:n arvot ovat siis suoraan metrejä.

    delta_height : float
        DEM:n ilmoitettu kokonaiskorkeusero metreinä.

        Tätä EI käytetä DEM:n skaalaamiseen.
        Funktio tarkistaa debug-tilassa todellisen
        korkeuseron:

            dem.max() - dem.min()

    add_water_in_earth_ocean_units : float
        Lisättävän veden määrä Maan valtameren tilavuuden
        yksikköinä.

            0.0 = ei vettä
            0.5 = puoli Maan valtamerta
            1.0 = yksi Maan valtameri
            2.0 = kaksi Maan valtamerta
            10.0 = kymmenen Maan valtamerta

    planet_radius_km : float
        Planeetan säde kilometreinä.

    debug : bool
        Tulostetaanko debug-tietoja.

    Palauttaa
    ----------
    transformed_dem : np.ndarray
        DEM merenpinnan suhteen metreinä:

            meri < 0
            merenpinta = 0
            maa > 0

    dem_min : float
        Lopullisen DEM:n alin kohta metreinä.

    dem_max : float
        Lopullisen DEM:n korkein kohta metreinä.

    delta_dem : float
        Lopullinen korkeusero metreinä.

    sealevel_from_dem_zero : float
        Merenpinnan korkeus alkuperäisen DEM:n
        koordinaatistossa metreinä.

    added_water_volume_m3 : float
        Toteutunut vesitilavuus kuutiometreinä.

    target_water_volume_m3 : float
        Tavoiteltu vesitilavuus kuutiometreinä.
    """

    # ============================================================
    # 1. TARKISTUKSET
    # ============================================================

    if not isinstance(dem, np.ndarray):
        dem = np.asarray(dem)

    if dem.ndim != 2:
        raise ValueError(
            "DEM:n pitää olla 2-ulotteinen."
        )

    if not np.issubdtype(dem.dtype, np.number):
        raise ValueError(
            "DEM:n pitää sisältää numeerisia arvoja."
        )

    if not np.all(np.isfinite(dem)):
        raise ValueError(
            "DEM sisältää NaN- tai inf-arvoja."
        )

    if delta_height <= 0:
        raise ValueError(
            "delta_height pitää olla positiivinen."
        )

    if add_water_in_earth_ocean_units < 0:
        raise ValueError(
            "add_water_in_earth_ocean_units ei voi olla negatiivinen."
        )

    if planet_radius_km <= 0:
        raise ValueError(
            "planet_radius_km pitää olla positiivinen."
        )

    # Käytetään float64-laskentaa
    dem = dem.astype(
        np.float64,
        copy=False
    )

    height, width = dem.shape

    # ============================================================
    # 2. ALKUPERÄISEN DEM:N TIEDOT
    # ============================================================

    dem0_min = np.min(dem)
    dem0_max = np.max(dem)

    actual_delta_height = (
        dem0_max - dem0_min
    )

    if actual_delta_height <= 0:
        raise ValueError(
            "DEM on vakioarvoinen."
        )

    # ============================================================
    # 3. MAAN VALTAMEREN TILAVUUS
    #
    # Maan valtameren tilavuuden likiarvo:
    #
    #     1.332 × 10^18 m³
    #
    # ============================================================

    EARTH_OCEAN_VOLUME_M3 = 1.332e18

    target_water_volume_m3 = (
        add_water_in_earth_ocean_units
        * EARTH_OCEAN_VOLUME_M3
    )

    # ============================================================
    # 4. PLANEETAN SÄDE
    #
    # kilometrit -> metrit
    # ============================================================

    planet_radius_m = (
        planet_radius_km * 1000.0
    )

    # ============================================================
    # 5. LATITUDE-PIKSELIEN KESKIKOHdat
    #
    # DEM kattaa:
    #
    #     -90 ... +90
    #
    # Käytetään pikselien keskikohtia.
    # ============================================================

    lat = (
        -90.0
        + (
            np.arange(height) + 0.5
        )
        * 180.0
        / height
    )

    lat_rad = np.deg2rad(lat)

    # ============================================================
    # 6. PALLON PIKSELIN PINTA-ALA
    #
    # dA =
    #
    # R² cos(latitude) dlat dlon
    #
    # ============================================================

    dlat = (
        np.pi / height
    )

    dlon = (
        2.0 * np.pi / width
    )

    row_area = (
        planet_radius_m ** 2
        * np.cos(lat_rad)
        * dlat
        * dlon
    )

    pixel_area = np.broadcast_to(
        row_area[:, None],
        dem.shape
    )

    # ============================================================
    # 7. TARKISTETAAN PLANEETAN PINTA-ALA
    # ============================================================

    total_planet_area = np.sum(
        pixel_area
    )

    theoretical_planet_area = (
        4.0
        * np.pi
        * planet_radius_m ** 2
    )

    # ============================================================
    # 8. DEBUG: INPUT
    # ============================================================

    if debug:

        print()
        print(
            "============================================================"
        )
        print(
            "PLANET FLOODING"
        )
        print(
            "============================================================"
        )

        print(
            f"DEM shape                  : {dem.shape}"
        )

        print(
            f"DEM dtype                  : {dem.dtype}"
        )

        print(
            f"DEM min                    : "
            f"{dem0_min:.6f} m"
        )

        print(
            f"DEM max                    : "
            f"{dem0_max:.6f} m"
        )

        print(
            f"DEM actual delta           : "
            f"{actual_delta_height:.6f} m"
        )

        print(
            f"delta_height parameter     : "
            f"{delta_height:.6f} m"
        )

        print(
            f"planet radius              : "
            f"{planet_radius_km:.6f} km"
        )

        print(
            f"planet area calculated     : "
            f"{total_planet_area:.6e} m²"
        )

        print(
            f"planet area theoretical    : "
            f"{theoretical_planet_area:.6e} m²"
        )

        print(
            f"area relative error        : "
            f"{(
                total_planet_area
                / theoretical_planet_area
                - 1.0
            ):.6e}"
        )

        print(
            f"Earth ocean units          : "
            f"{add_water_in_earth_ocean_units:.8f}"
        )

        print(
            f"Earth ocean volume         : "
            f"{EARTH_OCEAN_VOLUME_M3:.6e} m³"
        )

        print(
            f"target water volume       : "
            f"{target_water_volume_m3:.6e} m³"
        )

    # ============================================================
    # 9. EI VETTÄ
    # ============================================================

    if target_water_volume_m3 == 0.0:

        # Ei lisätä vettä.
        #
        # Asetetaan alin DEM-piste merenpinnaksi.
        #
        # Tällöin kaikki muu on positiivista.

        sealevel_from_dem_zero = dem0_min

        transformed_dem = (
            dem - sealevel_from_dem_zero
        )

        dem_min = np.min(
            transformed_dem
        )

        dem_max = np.max(
            transformed_dem
        )

        delta_dem = (
            dem_max - dem_min
        )

        if debug:

            print()
            print(
                "------------------------------------------------------------"
            )
            print(
                "NO WATER"
            )
            print(
                "------------------------------------------------------------"
            )

            print(
                f"sea level                 : "
                f"{sealevel_from_dem_zero:.6f} m"
            )

        return (
            transformed_dem,
            dem_min,
            dem_max,
            delta_dem,
            sealevel_from_dem_zero,
            0.0,
            0.0,
        )

    # ============================================================
    # 10. MUUTETAAN DEM YKSIULOTTEISEKSI
    # ============================================================

    elevations = dem.ravel()

    areas = pixel_area.ravel()

    # ============================================================
    # 11. JÄRJESTETÄÄN KORKEUDEN MUKAAN
    # ============================================================

    order = np.argsort(
        elevations
    )

    sorted_elevations = (
        elevations[order]
    )

    sorted_areas = (
        areas[order]
    )

    # ============================================================
    # 12. KUMULATIIVINEN PINTA-ALA
    #
    # cumulative_area[i]
    #
    # = kaikkien korkeudella <= h olevien
    #   pikseleiden pinta-ala
    # ============================================================

    cumulative_area = np.cumsum(
        sorted_areas
    )

    # ============================================================
    # 13. KUMULATIIVINEN ∫ h dA
    #
    # Tätä tarvitaan veden tilavuuden laskemiseen.
    #
    # ============================================================

    cumulative_elevation_area = np.cumsum(
        sorted_elevations
        * sorted_areas
    )

    # ============================================================
    # 14. KOKO PLANEETAN ∫ h dA
    # ============================================================

    total_elevation_area = (
        cumulative_elevation_area[-1]
    )

    # ============================================================
    # 15. FUNKTIO:
    #
    # VEDEN TILAVUUS ANNETULLA MERENPINNALLA
    #
    # V(S) =
    #
    #     ∫ (S - h) dA
    #
    # kaikille h <= S.
    #
    # ============================================================

    def water_volume_at_sea_level(
        sea_level
    ):
        """
        Laskee veden tilavuuden merenpinnalla sea_level.
        """

        index = np.searchsorted(
            sorted_elevations,
            sea_level,
            side="right"
        )

        # Merenpinta on kaikkien DEM-pisteiden alapuolella.
        #
        # Tällöin vettä ei ole.

        if index == 0:
            return 0.0

        # --------------------------------------------------------
        # Tapaus 1:
        #
        # Merenpinta on DEM:n sisällä.
        # --------------------------------------------------------

        if index < len(sorted_elevations):

            area_below = (
                cumulative_area[index - 1]
            )

            elevation_area_below = (
                cumulative_elevation_area[index - 1]
            )

            volume = (
                sea_level * area_below
                - elevation_area_below
            )

            return volume

        # --------------------------------------------------------
        # Tapaus 2:
        #
        # Merenpinta on koko DEM:n yläpuolella.
        #
        # Koko planeetta on veden alla.
        #
        # V =
        #
        #     S * A - ∫h dA
        #
        # --------------------------------------------------------

        volume = (
            sea_level
            * total_planet_area
            - total_elevation_area
        )

        return volume

    # ============================================================
    # 16. ALARAJA
    # ============================================================

    low = dem0_min

    # ============================================================
    # 17. YLÄRAJA
    #
    # Aloitetaan DEM:n korkeimmasta pisteestä.
    #
    # Jos vettä tarvitaan enemmän kuin DEM:n sisälle mahtuu,
    # nostetaan merenpintaa DEM:n yläpuolelle.
    # ============================================================

    high = dem0_max

    # ============================================================
    # 18. KASVATETAAN YLÄRAJAA TARVITTAESSA
    #
    # Tämä on se kohta, joka korjaa aiemman virheen.
    #
    # Merenpinta saa olla korkeampi kuin korkein vuori.
    # ============================================================

    iterations_expand = 0

    while (
        water_volume_at_sea_level(high)
        < target_water_volume_m3
    ):

        # Käytetään vähintään kilometrin askelta.
        #
        # Jos DEM:n korkeusero on esimerkiksi 20 km,
        # käytetään 20 km askelta.

        height_step = max(
            actual_delta_height,
            1000.0
        )

        high += height_step

        iterations_expand += 1

        # Turvaraja mahdollisen virheen varalta.

        if iterations_expand > 100000:

            raise RuntimeError(
                "Merenpinnan ylärajan etsintä ei "
                "konvergoitunut."
            )

    # ============================================================
    # 19. BINÄÄRIHAKU MERENPINNALLE
    # ============================================================

    tolerance_m = 1e-8

    max_iterations = 200

    for _ in range(
        max_iterations
    ):

        mid = (
            0.5
            * (low + high)
        )

        volume = (
            water_volume_at_sea_level(
                mid
            )
        )

        if volume < target_water_volume_m3:

            # Vettä tarvitaan lisää.
            # Nostetaan merenpintaa.

            low = mid

        else:

            # Vettä on jo tarpeeksi.
            # Lasketaan merenpintaa.

            high = mid

        if (
            high - low
            < tolerance_m
        ):
            break

    # ============================================================
    # 20. LOPULLINEN MERENPINTA
    # ============================================================

    sealevel_from_dem_zero = (
        0.5
        * (low + high)
    )

    # ============================================================
    # 21. TODELLINEN VESITILAVUUS
    # ============================================================

    added_water_volume_m3 = (
        water_volume_at_sea_level(
            sealevel_from_dem_zero
        )
    )

    # ============================================================
    # 22. SIIRRETÄÄN DEM MERENPINNAN SUHTEEN
    #
    #     meri < 0
    #     merenpinta = 0
    #     maa > 0
    #
    # ============================================================

    transformed_dem = (
        dem
        - sealevel_from_dem_zero
    )

    # ============================================================
    # 23. LOPULLISET MIN/MAX
    # ============================================================

    dem_min = np.min(
        transformed_dem
    )

    dem_max = np.max(
        transformed_dem
    )

    delta_dem = (
        dem_max
        - dem_min
    )

    # ============================================================
    # 24. VEDEN PINTA-ALA
    # ============================================================

    water_mask = (
        transformed_dem < 0.0
    )

    water_area = np.sum(
        pixel_area[water_mask]
    )

    water_area_fraction = (
        water_area
        / total_planet_area
    )

    land_area_fraction = (
        1.0
        - water_area_fraction
    )

    # ============================================================
    # 25. TOTEUTUNUT VALTAMERIKERROIN
    # ============================================================

    actual_ocean_units = (
        added_water_volume_m3
        / EARTH_OCEAN_VOLUME_M3
    )

    # ============================================================
    # 26. DEBUG
    # ============================================================

    if debug:

        volume_error = (
            added_water_volume_m3
            - target_water_volume_m3
        )

        relative_volume_error = (
            volume_error
            / target_water_volume_m3
        )

        print()
        print(
            "------------------------------------------------------------"
        )
        print(
            "SEA LEVEL"
        )
        print(
            "------------------------------------------------------------"
        )

        print(
            f"sea level                 : "
            f"{sealevel_from_dem_zero:.9f} m"
        )

        print(
            f"highest original point    : "
            f"{dem0_max:.6f} m"
        )

        print(
            f"sea level above max DEM   : "
            f"{(
                sealevel_from_dem_zero
                - dem0_max
            ):.6f} m"
        )

        print(
            f"added water volume        : "
            f"{added_water_volume_m3:.9e} m³"
        )

        print(
            f"target water volume       : "
            f"{target_water_volume_m3:.9e} m³"
        )

        print(
            f"volume error              : "
            f"{volume_error:.9e} m³"
        )

        print(
            f"relative volume error     : "
            f"{relative_volume_error:.9e}"
        )

        print(
            f"actual ocean units        : "
            f"{actual_ocean_units:.12f}"
        )

        print()
        print(
            "------------------------------------------------------------"
        )
        print(
            "SURFACE AREA"
        )
        print(
            "------------------------------------------------------------"
        )

        print(
            f"water area                : "
            f"{water_area:.9e} m²"
        )

        print(
            f"water fraction            : "
            f"{water_area_fraction:.9f}"
        )

        print(
            f"land fraction             : "
            f"{land_area_fraction:.9f}"
        )

        print()
        print(
            "------------------------------------------------------------"
        )
        print(
            "FINAL DEM"
        )
        print(
            "------------------------------------------------------------"
        )

        print(
            f"dem_min                  : "
            f"{dem_min:.9f} m"
        )

        print(
            f"dem_max                  : "
            f"{dem_max:.9f} m"
        )

        print(
            f"delta_dem                : "
            f"{delta_dem:.9f} m"
        )

        print()
        print(
            "============================================================"
        )

    # ============================================================
    # 27. PALAUTUS
    # ============================================================

    return (
        transformed_dem,
        dem_min,
        dem_max,
        delta_dem,
        sealevel_from_dem_zero,
        added_water_volume_m3,
        target_water_volume_m3,
    )





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





def levia(alku_x, alku_y,
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
    #alku_y, alku_x = np.unravel_index(
    #    np.argmax(cayonu_index),
    #    shape
    #)
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
    # 19. PALAUTUS
    # ============================================================

    return (
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

import numpy as np
from scipy import ndimage


def dem_curvature(dem, cellsize_x, cellsize_y=None):
    """
    Laskee DEM:n paikallisen kuperuuden/koveruuden.

    Parameters
    ----------
    dem : 2D numpy.ndarray
        DEM-korkeudet.
    cellsize_x : float
        Solukoko X-suunnassa metreinä.
    cellsize_y : float, optional
        Solukoko Y-suunnassa metreinä.
        Jos None, käytetään cellsize_x.

    Returns
    -------
    curvature : 2D numpy.ndarray
        Pinnan kokonaiskuperuus/koveruus.
    profile_curvature : 2D numpy.ndarray
        Kuperuus/koveruus rinteen suunnassa.
    plan_curvature : 2D numpy.ndarray
        Kuperuus/koveruus rinteen poikkisuunnassa.
    """

    if cellsize_y is None:
        cellsize_y = cellsize_x

    z = np.asarray(dem, dtype=float)

    # 1. Ensimmäiset derivaatat
    dz_dy, dz_dx = np.gradient(z, cellsize_y, cellsize_x)

    # 2. Toiset derivaatat
    d2z_dx2 = np.gradient(dz_dx, cellsize_x, axis=1)
    d2z_dy2 = np.gradient(dz_dy, cellsize_y, axis=0)
    d2z_dxdy = np.gradient(dz_dx, cellsize_y, axis=0)

    # Gradientin neliö
    p = dz_dx
    q = dz_dy

    # Vältetään nollalla jako
    eps = np.finfo(float).eps

    # Profile curvature
    denominator_profile = (
        (p**2 + q**2) *
        (1.0 + p**2 + q**2)**1.5
    )

    numerator_profile = (
        d2z_dx2 * p**2 +
        2.0 * d2z_dxdy * p * q +
        d2z_dy2 * q**2
    )

    profile_curvature = (
        -numerator_profile /
        np.maximum(denominator_profile, eps)
    )

    # Plan curvature
    denominator_plan = (
        (p**2 + q**2)**1.5
    )

    numerator_plan = (
        d2z_dx2 * q**2 -
        2.0 * d2z_dxdy * p * q +
        d2z_dy2 * p**2
    )

    plan_curvature = (
        -numerator_plan /
        np.maximum(denominator_plan, eps)
    )

    # Kokonaiskuperuus
    curvature = profile_curvature + plan_curvature

    return curvature, profile_curvature, plan_curvature

def laske_lammonkuljetus(maannerosuus):
    """
    Laskee suhteellisen lämmönkuljetuksen päiväntasaajalta navoille (Maa = 1.0).
    
    Parametri:
    maannerosuus (float): Planeetan maapinta-alan osuus välillä 0.0 - 1.0 
                          (esim. 0.30 = 30% maata, eli Maan nykytila).
    """
    if not (0.0 <= maannerosuus <= 1.0):
        raise ValueError("Maannerosuuden on oltava välillä 0.0 (0%) ja 1.0 (100%).")
        
    # Jos planeetalla on vähemmän maata kuin Maassa (0% - 30%)
    if maannerosuus <= 0.3:
        # Interpoloidaan välillä [0.0, 0.3] -> lämmönkuljetus laskee [1.15 -> 1.00]
        kuljetus = 1.15 - (maannerosuus / 0.3) * 0.15
        
    # Jos planeetalla on enemmän maata kuin Maassa (30% - 100%)
    else:
        # Interpoloidaan välillä [0.3, 1.0] -> lämmönkuljetus laskee [1.00 -> 0.75]
        kuljetus = 1.00 - ((maannerosuus - 0.3) / 0.7) * 0.25
        
    return round(kuljetus, 2)



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

def calculate_ice_age(dem, temp_annual, precip_annual):
    jaatikko_tulos = laske_jaatikko(
    dem,
    temp_annual,
    precip_annual,
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
    temperature_ice_age=np.copy(temp_annual)-10
    precipitation_ice_age=np.copy(precip_annual)*0.5 
    loess_areas_ice_age=np.copy(sea_level_ice_age)
    manner_osuudet_ice_age, manner_koko_jarjestys_ice_age = laske_mantereet(relief_ice_age, planet_radius=planet_radius)   
    print("\nValmis! Taulukoiden muodot:", manner_osuudet.shape, manner_koko_jarjestys.shape) 
  

def calculate_bioclim(temps, precs):
    """
    Laskee BIO1-BIO19-bioklimaattiset muuttujat 12 kuukausittaiselle
    rasteridatalle.

    Parametrit
    ----------
    temps : np.ndarray
        Kuukausittaiset keskilämpötilat muodossa:
        (12, height, width)

    precs : np.ndarray
        Kuukausittaiset sademäärät muodossa:
        (12, height, width)

    Palauttaa
    ----------
    dict
        Sanakirja, jossa avaimina BIO1 ... BIO19 ja arvoina
        2D-rasterit muodossa (height, width).

    BIO-muuttujat
    -------------
    BIO1  Annual Mean Temperature
    BIO2  Mean Diurnal Range
    BIO3  Isothermality
    BIO4  Temperature Seasonality
    BIO5  Max Temperature of Warmest Month
    BIO6  Min Temperature of Coldest Month
    BIO7  Temperature Annual Range
    BIO8  Mean Temperature of Wettest Quarter
    BIO9  Mean Temperature of Driest Quarter
    BIO10 Mean Temperature of Warmest Quarter
    BIO11 Mean Temperature of Coldest Quarter
    BIO12 Annual Precipitation
    BIO13 Precipitation of Wettest Month
    BIO14 Precipitation of Driest Month
    BIO15 Precipitation Seasonality
    BIO16 Precipitation of Wettest Quarter
    BIO17 Precipitation of Driest Quarter
    BIO18 Precipitation of Warmest Quarter
    BIO19 Precipitation of Coldest Quarter

    Huomio
    ------
    Neljänneksen (quarter) määrittely tehdään liikkuvana 3 kuukauden
    jaksona. Näin löydetään kullekin pikselille vuoden märin, kuivin,
    lämpimin ja kylmin kolmen kuukauden jakso.
    """

    # ==========================================================
    # 0. INPUTIEN TARKISTUS
    # ==========================================================

    temps = np.asarray(temps, dtype=float)
    precs = np.asarray(precs, dtype=float)

    if temps.ndim != 3 or precs.ndim != 3:
        raise ValueError(
            "temps- ja precs-taulukoiden pitää olla 3D-muodossa "
            "(12, korkeus, leveys)."
        )

    if temps.shape[0] != 12 or precs.shape[0] != 12:
        raise ValueError(
            "temps- ja precs-taulukoissa pitää olla 12 kuukautta "
            "ensimmäisellä akselilla (axis=0)."
        )

    if temps.shape != precs.shape:
        raise ValueError(
            "temps- ja precs-taulukoiden dimensioiden pitää olla samat."
        )

    if np.any(precs < 0):
        raise ValueError("Sademäärät eivät voi olla negatiivisia.")

    # ==========================================================
    # 1. TMIN JA TMAX
    # ==========================================================
    #
    # Jos käytettävissä on vain kuukausittainen keskilämpötila,
    # estimoidaan kuukausittainen minimi- ja maksimilämpötila.
    #
    # Tämä osa voidaan myöhemmin korvata oikeilla Tmin/Tmax-rastereilla.
    #

    amplitude = (
        4.0
        + 0.12 * np.abs(temps)
        + 2.0 * np.exp(-precs / 100.0)
    )

    tmin = temps - amplitude
    tmax = temps + amplitude

    # ==========================================================
    # 2. KUUKAUSITTAINEN LÄMPÖTILAERO
    # ==========================================================

    monthly_range = tmax - tmin

    # ==========================================================
    # 3. BIO1 - Annual Mean Temperature
    # ==========================================================

    BIO1 = np.nanmean(temps, axis=0)

    # ==========================================================
    # 4. BIO2 - Mean Diurnal Range
    # ==========================================================

    BIO2 = np.nanmean(monthly_range, axis=0)

    # ==========================================================
    # 5. BIO3 - Isothermality
    # ==========================================================

    annual_temp_range = (
        np.nanmax(tmax, axis=0)
        - np.nanmin(tmin, axis=0)
    )

    BIO3 = np.divide(
        BIO2 * 100.0,
        annual_temp_range,
        out=np.zeros_like(BIO2),
        where=annual_temp_range != 0
    )

    # ==========================================================
    # 6. BIO4 - Temperature Seasonality
    # ==========================================================

    BIO4 = np.nanstd(temps, axis=0, ddof=0) * 100.0

    # ==========================================================
    # 7. BIO5 - Max Temperature of Warmest Month
    # ==========================================================

    BIO5 = np.nanmax(tmax, axis=0)

    # ==========================================================
    # 8. BIO6 - Min Temperature of Coldest Month
    # ==========================================================

    BIO6 = np.nanmin(tmin, axis=0)

    # ==========================================================
    # 9. BIO7 - Temperature Annual Range
    # ==========================================================

    BIO7 = BIO5 - BIO6

    # ==========================================================
    # 10. 3 KUUKAUDEN LIIKKUVAT JAKSOT
    # ==========================================================
    #
    # Kuukaudet:
    #   0 = tammikuu
    #   1 = helmikuu
    #   ...
    #   11 = joulukuu
    #
    # Luodaan 12 mahdollista 3 kk:n jaksoa:
    #
    # Jan-Feb-Mar
    # Feb-Mar-Apr
    # ...
    # Dec-Jan-Feb
    #
    # np.roll mahdollistaa vuodenvaihteen yli menevät kvartaalit.
    #

    quarter_prec = np.empty(
        (12,) + precs.shape[1:],
        dtype=float
    )

    quarter_temp = np.empty(
        (12,) + temps.shape[1:],
        dtype=float
    )

    for start in range(12):

        months = [
            start,
            (start + 1) % 12,
            (start + 2) % 12
        ]

        quarter_prec[start] = np.nansum(
            precs[months],
            axis=0
        )

        # Neljänneksen keskilämpötila.
        quarter_temp[start] = np.nanmean(
            temps[months],
            axis=0
        )

    # ==========================================================
    # 11. MÄRIN JA KUIVIN KVARTAALI
    # ==========================================================

    wettest_quarter_idx = np.nanargmax(
        quarter_prec,
        axis=0
    )

    driest_quarter_idx = np.nanargmin(
        quarter_prec,
        axis=0
    )

    # ==========================================================
    # 12. BIO8 - Mean Temperature of Wettest Quarter
    # ==========================================================

    BIO8 = np.take_along_axis(
        quarter_temp,
        wettest_quarter_idx[np.newaxis, ...],
        axis=0
    )[0]

    # ==========================================================
    # 13. BIO9 - Mean Temperature of Driest Quarter
    # ==========================================================

    BIO9 = np.take_along_axis(
        quarter_temp,
        driest_quarter_idx[np.newaxis, ...],
        axis=0
    )[0]

    # ==========================================================
    # 14. LÄMPIMIN JA KYLMIN KVARTAALI
    # ==========================================================

    warmest_quarter_idx = np.nanargmax(
        quarter_temp,
        axis=0
    )

    coldest_quarter_idx = np.nanargmin(
        quarter_temp,
        axis=0
    )

    # ==========================================================
    # 15. BIO10 - Mean Temperature of Warmest Quarter
    # ==========================================================

    BIO10 = np.take_along_axis(
        quarter_temp,
        warmest_quarter_idx[np.newaxis, ...],
        axis=0
    )[0]

    # ==========================================================
    # 16. BIO11 - Mean Temperature of Coldest Quarter
    # ==========================================================

    BIO11 = np.take_along_axis(
        quarter_temp,
        coldest_quarter_idx[np.newaxis, ...],
        axis=0
    )[0]

    # ==========================================================
    # 17. BIO12 - Annual Precipitation
    # ==========================================================

    BIO12 = np.nansum(precs, axis=0)

    # ==========================================================
    # 18. BIO13 - Precipitation of Wettest Month
    # ==========================================================

    BIO13 = np.nanmax(precs, axis=0)

    # ==========================================================
    # 19. BIO14 - Precipitation of Driest Month
    # ==========================================================

    BIO14 = np.nanmin(precs, axis=0)

    # ==========================================================
    # 20. BIO15 - Precipitation Seasonality
    # ==========================================================
    #
    # WorldClim-tyyppisesti:
    #
    # standard deviation / mean * 100
    #

    precip_mean = np.nanmean(precs, axis=0)
    precip_std = np.nanstd(precs, axis=0, ddof=0)

    BIO15 = np.divide(
        precip_std * 100.0,
        precip_mean,
        out=np.zeros_like(precip_std),
        where=precip_mean != 0
    )

    # ==========================================================
    # 21. BIO16 - Precipitation of Wettest Quarter
    # ==========================================================

    BIO16 = np.nanmax(
        quarter_prec,
        axis=0
    )

    # ==========================================================
    # 22. BIO17 - Precipitation of Driest Quarter
    # ==========================================================

    BIO17 = np.nanmin(
        quarter_prec,
        axis=0
    )

    # ==========================================================
    # 23. BIO18 - Precipitation of Warmest Quarter
    # ==========================================================

    BIO18 = np.take_along_axis(
        quarter_prec,
        warmest_quarter_idx[np.newaxis, ...],
        axis=0
    )[0]

    # ==========================================================
    # 24. BIO19 - Precipitation of Coldest Quarter
    # ==========================================================

    BIO19 = np.take_along_axis(
        quarter_prec,
        coldest_quarter_idx[np.newaxis, ...],
        axis=0
    )[0]

    # ==========================================================
    # 25. PALAUTUS
    # ==========================================================

    return {
        "BIO1": BIO1,
        "BIO2": BIO2,
        "BIO3": BIO3,
        "BIO4": BIO4,
        "BIO5": BIO5,
        "BIO6": BIO6,
        "BIO7": BIO7,
        "BIO8": BIO8,
        "BIO9": BIO9,
        "BIO10": BIO10,
        "BIO11": BIO11,
        "BIO12": BIO12,
        "BIO13": BIO13,
        "BIO14": BIO14,
        "BIO15": BIO15,
        "BIO16": BIO16,
        "BIO17": BIO17,
        "BIO18": BIO18,
        "BIO19": BIO19
    }



def calculate_population_density(npp):

   # Muuta NPP yksikköön kg/m²
   npp_kg_m2 = npp
   # Käytä annettua kaavaa
   detected_log10density = 9.6e-4 * npp_kg_m2 - 1.53
   detected_density = np.power(10, detected_log10density)
   return detected_density



import numpy as np
import heapq


def muodosta_kansat(
    population_in_pixels,
    relief,
    planet_radius,
    population_per_nation=100_000,
    min_distance_km=500.0,

    # ------------------------------------------------------------
    # Vuoristot / rinteet
    # ------------------------------------------------------------
    slope_scale=15.0,
    slope_power=2.0,

    # ------------------------------------------------------------
    # Kulttuurinen etäisyys
    # ------------------------------------------------------------
    culture_distance_scale_km=1500.0,
    culture_power=2.0,

    # ------------------------------------------------------------
    # Väestön vaikutus
    # ------------------------------------------------------------
    population_influence=0.35,
    population_scale=1000.0,

    # ------------------------------------------------------------
    # Jokilaaksojen vaikutus
    # ------------------------------------------------------------
    river_influence=0.30,
    river_valley_scale=500.0,
    river_valley_power=2.0,

    diagonal=True,
):
    """
    Muodostaa kansojen keskukset ja alueet.

    Reliefistä johdetaan automaattisesti jokilaaksojen
    kaltaisia kulkureittejä.

    Palauttaa:

        centers
        nation_map
        nation_populations

    Lisäksi funktio palauttaa debug-kentät:

        terrain_factor
        river_factor
        physical_distance
        total_cost
    """

    # ============================================================
    # 1. INPUT
    # ============================================================

    pop = np.asarray(
        population_in_pixels,
        dtype=np.float64,
    )

    relief = np.asarray(
        relief,
        dtype=np.float64,
    )

    if pop.ndim != 2:
        raise ValueError(
            "population_in_pixels pitää olla 2D-array."
        )

    if relief.shape != pop.shape:
        raise ValueError(
            "relief ja population_in_pixels pitää olla "
            "saman kokoisia."
        )

    if planet_radius <= 0:
        raise ValueError(
            "planet_radius pitää olla > 0."
        )

    height, width = pop.shape

    # ============================================================
    # 2. MAA / MERI
    # ============================================================

    land = (
        np.isfinite(relief)
        & (relief > 0)
    )

    valid_population = (
        np.isfinite(pop)
        & (pop > 0)
        & land
    )

    if not np.any(valid_population):
        return (
            [],
            np.zeros_like(
                pop,
                dtype=np.int32,
            ),
            np.array(
                [],
                dtype=np.float64,
            ),
            None,
            None,
            None,
            None,
        )

    total_population = (
        pop[valid_population].sum()
    )

    n_nations = int(
        total_population
        / population_per_nation
    )

    if n_nations < 1:
        return (
            [],
            np.zeros_like(
                pop,
                dtype=np.int32,
            ),
            np.array(
                [],
                dtype=np.float64,
            ),
            None,
            None,
            None,
            None,
        )

    # ============================================================
    # 3. LAT / LON
    # ============================================================

    dlat_deg = 180.0 / height
    dlon_deg = 360.0 / width

    lat_deg = (
        90.0
        - (np.arange(height) + 0.5)
        * dlat_deg
    )

    lon_deg = (
        -180.0
        + (np.arange(width) + 0.5)
        * dlon_deg
    )

    lat_rad = np.deg2rad(lat_deg)
    lon_rad = np.deg2rad(lon_deg)

    lat_grid, lon_grid = np.meshgrid(
        lat_rad,
        lon_rad,
        indexing="ij",
    )

    # ============================================================
    # 4. PALLOKOORDINAATIT
    # ============================================================

    cos_lat = np.cos(lat_grid)

    x = (
        cos_lat
        * np.cos(lon_grid)
    )

    y = (
        cos_lat
        * np.sin(lon_grid)
    )

    z = np.sin(lat_grid)

    # ============================================================
    # 5. KANSAKESKUKSET
    # ============================================================

    flat_valid = np.flatnonzero(
        valid_population.ravel()
    )

    xyz = np.column_stack(
        (
            x.ravel()[flat_valid],
            y.ravel()[flat_valid],
            z.ravel()[flat_valid],
        )
    )

    values = pop.ravel()[flat_valid]

    min_angle = (
        min_distance_km
        / planet_radius
    )

    cos_min_angle = np.cos(
        min_angle
    )

    available = np.ones(
        len(values),
        dtype=bool,
    )

    centers = []

    for _ in range(n_nations):

        if not np.any(available):
            break

        candidates = np.where(
            available,
            values,
            -np.inf,
        )

        idx = np.argmax(
            candidates
        )

        center = np.unravel_index(
            flat_valid[idx],
            pop.shape,
        )

        centers.append(center)

        center_xyz = xyz[idx]

        dot = xyz @ center_xyz

        too_close = (
            dot >= cos_min_angle
        )

        available[too_close] = False

    if not centers:
        return (
            [],
            np.zeros_like(
                pop,
                dtype=np.int32,
            ),
            np.array(
                [],
                dtype=np.float64,
            ),
            None,
            None,
            None,
            None,
        )

    # ============================================================
    # 6. KALTEVUUS
    # ============================================================

    elevation = np.maximum(
        relief,
        0.0,
    )

    dy_km = (
        planet_radius
        * np.deg2rad(dlat_deg)
    )

    dz_dy = (
        np.gradient(
            elevation,
            axis=0,
        )
        / dy_km
    )

    # Longitude wrap

    elevation_left = np.roll(
        elevation,
        1,
        axis=1,
    )

    elevation_right = np.roll(
        elevation,
        -1,
        axis=1,
    )

    dx_km = (
        planet_radius
        * np.cos(lat_grid)
        * np.deg2rad(dlon_deg)
    )

    dx_km = np.maximum(
        dx_km,
        1e-6,
    )

    dz_dx = (
        elevation_right
        - elevation_left
    ) / (
        2.0 * dx_km
    )

    gradient = np.sqrt(
        dz_dx ** 2
        + dz_dy ** 2
    )

    slope_rad = np.arctan(
        gradient
    )

    slope_deg = np.rad2deg(
        slope_rad
    )

    # ============================================================
    # 7. MAASTON KULKUKERROIN
    # ============================================================

    slope_factor = (
        slope_deg
        / slope_scale
    )

    terrain_factor = (
        1.0
        + slope_factor ** slope_power
    )

    terrain_factor[
        ~land
    ] = np.inf

    # ============================================================
    # 8. JOKILAAKSO-ARVIO
    # ============================================================
    #
    # Emme yritä "piirtää jokia".
    #
    # Sen sijaan muodostetaan jatkuva river_factor:
    #
    #     0 = ei jokilaakson vaikutusta
    #     1 = voimakas jokilaakso
    #
    # Perusteena käytetään sitä, kuinka paljon pikseli
    # on ympäristöään alempana.
    #
    # ------------------------------------------------------------
    # 3x3-ympäristön pienin korkeus
    # ------------------------------------------------------------

    padded = np.pad(
        elevation,
        1,
        mode="wrap",
    )

    local_min = np.full_like(
        elevation,
        np.inf,
    )

    for dr in range(3):
        for dc in range(3):

            if dr == 1 and dc == 1:
                continue

            local_min = np.minimum(
                local_min,
                padded[
                    dr:dr + height,
                    dc:dc + width,
                ],
            )

    # ------------------------------------------------------------
    # Kuinka paljon ympäristö on korkeammalla?
    #
    # Tämä korostaa painanteita.
    # ------------------------------------------------------------

    valley_depth = (
        local_min
        - elevation
    )

    valley_depth = np.maximum(
        valley_depth,
        0.0,
    )

    # ------------------------------------------------------------
    # Myös kaltevuuden pitää olla kohtuullinen.
    #
    # Jyrkkä vuorenrinne ei saa muuttua joeksi vain siksi,
    # että se on ympäristöään alempana.
    # ------------------------------------------------------------

    gentle_factor = np.exp(
        -(
            slope_deg / 20.0
        ) ** 2
    )

    river_strength = (
        valley_depth
        / (
            valley_depth
            + river_valley_scale
        )
    )

    river_strength *= (
        gentle_factor
    )

    # Merellä ei ole jokibonusta.

    river_strength[
        ~land
    ] = 0.0

    # ============================================================
    # 9. VÄESTÖN LÄSNÄOLO
    # ============================================================

    population_presence = (
        1.0
        - np.exp(
            -np.maximum(
                pop,
                0.0,
            )
            / max(
                population_scale,
                1e-12,
            )
        )
    )

    # ============================================================
    # 10. NAAPURIT
    # ============================================================

    if diagonal:

        neighbors = (
            (-1, -1),
            (-1,  0),
            (-1,  1),
            ( 0, -1),
            ( 0,  1),
            ( 1, -1),
            ( 1,  0),
            ( 1,  1),
        )

    else:

        neighbors = (
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
        )

    # ============================================================
    # 11. TULOSKENTÄT
    # ============================================================

    nation_map = np.zeros(
        pop.shape,
        dtype=np.int32,
    )

    physical_distance = np.full(
        pop.shape,
        np.inf,
        dtype=np.float64,
    )

    terrain_distance = np.full(
        pop.shape,
        np.inf,
        dtype=np.float64,
    )

    total_cost = np.full(
        pop.shape,
        np.inf,
        dtype=np.float64,
    )

    # ============================================================
    # 12. PRIORITY QUEUE
    # ============================================================

    heap = []

    for nation_id, (r, c) in enumerate(
        centers,
        start=1,
    ):

        physical_distance[r, c] = 0.0
        terrain_distance[r, c] = 0.0
        total_cost[r, c] = 0.0

        nation_map[r, c] = nation_id

        heapq.heappush(
            heap,
            (
                0.0,
                nation_id,
                r,
                c,
            ),
        )

    # ============================================================
    # 13. KANSOJEN LEVIÄMINEN
    # ============================================================

    while heap:

        (
            current_cost,
            nation_id,
            r,
            c,
        ) = heapq.heappop(heap)

        if current_cost != total_cost[r, c]:
            continue

        current_physical_distance = (
            physical_distance[r, c]
        )

        current_terrain_distance = (
            terrain_distance[r, c]
        )

        for dr, dc in neighbors:

            nr = r + dr
            nc = c + dc

            # Napojen yli ei wrapata.

            if nr < 0 or nr >= height:
                continue

            # Longitude wrapataan.

            nc %= width

            if not land[nr, nc]:
                continue

            # ====================================================
            # 14. PALLOPINTA-ASKEL
            # ====================================================

            if dr == 0:

                step_distance_km = (
                    planet_radius
                    * np.cos(
                        lat_rad[nr]
                    )
                    * np.deg2rad(
                        dlon_deg
                    )
                )

            elif dc == 0:

                step_distance_km = (
                    dy_km
                )

            else:

                dx = (
                    planet_radius
                    * np.cos(
                        lat_rad[nr]
                    )
                    * np.deg2rad(
                        dlon_deg
                    )
                )

                step_distance_km = np.sqrt(
                    dx * dx
                    + dy_km * dy_km
                )

            # ====================================================
            # 15. FYYSINEN ETÄISYYS
            # ====================================================

            new_physical_distance = (
                current_physical_distance
                + step_distance_km
            )

            # ====================================================
            # 16. MAASTO
            # ====================================================

            terrain_step = (
                step_distance_km
                * terrain_factor[nr, nc]
            )

            new_terrain_distance = (
                current_terrain_distance
                + terrain_step
            )

            # ====================================================
            # 17. JOKIBONUS
            # ====================================================
            #
            # Jokilaakso alentaa paikallista kustannusta.
            #
            # Esimerkiksi river_influence=0.30:
            #
            #     ei jokea:
            #         factor = 1.0
            #
            #     vahva jokilaakso:
            #         factor = 0.70
            #
            # ====================================================

            river_factor = (
                1.0
                - river_influence
                * river_strength[nr, nc]
            )

            # ====================================================
            # 18. VÄESTÖBONUS
            # ====================================================

            presence = (
                population_presence[nr, nc]
            )

            population_factor = (
                1.0
                / (
                    1.0
                    + population_influence
                    * presence
                )
            )

            # ====================================================
            # 19. KULTTUURINEN ETÄISYYS
            # ====================================================

            culture_ratio = (
                new_physical_distance
                / culture_distance_scale_km
            )

            culture_factor = (
                1.0
                + culture_ratio ** culture_power
            )

            # ====================================================
            # 20. LOPULLINEN KUSTANNUS
            # ====================================================
            #
            #     relief
            #        ×
            #     joki
            #        ×
            #     väestö
            #        ×
            #     kulttuurinen etäisyys
            #
            # ====================================================

            new_cost = (
                new_terrain_distance
                * river_factor
                * population_factor
                * culture_factor
            )

            # ====================================================
            # 21. PÄIVITYS
            # ====================================================

            if new_cost < total_cost[nr, nc]:

                total_cost[nr, nc] = (
                    new_cost
                )

                physical_distance[nr, nc] = (
                    new_physical_distance
                )

                terrain_distance[nr, nc] = (
                    new_terrain_distance
                )

                nation_map[nr, nc] = (
                    nation_id
                )

                heapq.heappush(
                    heap,
                    (
                        new_cost,
                        nation_id,
                        nr,
                        nc,
                    ),
                )

    # ============================================================
    # 22. KANSOJEN VÄKILUVUT
    # ============================================================

    nation_ids = nation_map[
        valid_population
    ]

    population_values = pop[
        valid_population
    ]

    nation_populations = np.bincount(
        nation_ids,
        weights=population_values,
        minlength=len(centers) + 1,
    )[1:]

    # ============================================================
    # 23. DEBUG-KENTÄT MUKAAN
    # ============================================================

    return (
        centers,
        nation_map,
        nation_populations,
        terrain_factor,
        river_strength,
        physical_distance,
        total_cost,
    )


def muodosta_kansat_07(
    population_in_pixels,
    relief,
    planet_radius,
    population_per_nation=100_000,
    min_distance_km=500.0,

    # Maasto
    slope_scale=15.0,
    slope_power=2.0,

    # Kulttuurinen etäisyys
    culture_distance_scale_km=1500.0,
    culture_power=2.0,

    # Väestön vaikutus kulkemiseen
    population_influence=0.35,
    population_scale=1000.0,

    diagonal=True,
):
    """
    Muodostaa kansojen keskukset ja alueet.

    population_in_pixels:
        2D numpy-array, shape = (height, width)

    relief:
        2D numpy-array, sama shape.
        0 = meri
        >0 = maa, metreinä

    planet_radius:
        Planeetan säde kilometreinä.

    population_per_nation:
        Keskimääräinen väestömäärä yhtä kansaa kohti.

    min_distance_km:
        Kansakeskusten minimietäisyys.

    slope_scale:
        Kaltevuuskulma asteina, jossa rinne alkaa
        kasvattaa kulkukustannusta.

    slope_power:
        Kaltevuuden epälineaarisuus.

    culture_distance_scale_km:
        Kulttuurisen etäisyyden mittakaava.

    culture_power:
        Kulttuurisen etäisyyden epälineaarisuus.

    population_influence:
        Kuinka voimakkaasti olemassa oleva väestö
        helpottaa kansan leviämistä.

        0.0 = ei vaikutusta
        0.35 = kohtalainen vaikutus
        1.0 = erittäin voimakas

    population_scale:
        Väestömäärä, jolla väestöbonus alkaa näkyä.

    diagonal:
        True  -> 8 naapuria
        False -> 4 naapuria


    PALAUTUKSET
    -----------

    centers
        Lista (row, col)-koordinaatteja.

    nation_map
        0 = meri
        1 = kansa 1
        2 = kansa 2
        ...

    nation_populations
        Kansojen väkiluvut.
    """

    # ============================================================
    # 1. INPUT
    # ============================================================

    pop = np.asarray(
        population_in_pixels,
        dtype=np.float64,
    )

    relief = np.asarray(
        relief,
        dtype=np.float64,
    )

    if pop.ndim != 2:
        raise ValueError(
            "population_in_pixels pitää olla 2D-array."
        )

    if relief.shape != pop.shape:
        raise ValueError(
            "relief ja population_in_pixels pitää olla "
            "saman kokoisia."
        )

    if planet_radius <= 0:
        raise ValueError(
            "planet_radius pitää olla > 0."
        )

    height, width = pop.shape

    # ============================================================
    # 2. MAA / MERI
    # ============================================================

    land = (
        np.isfinite(relief)
        & (relief > 0)
    )

    valid_population = (
        np.isfinite(pop)
        & (pop > 0)
        & land
    )

    if not np.any(valid_population):
        return (
            [],
            np.zeros_like(
                pop,
                dtype=np.int32,
            ),
            np.array(
                [],
                dtype=np.float64,
            ),
        )

    total_population = (
        pop[valid_population].sum()
    )

    n_nations = int(
        total_population
        / population_per_nation
    )

    if n_nations < 1:
        return (
            [],
            np.zeros_like(
                pop,
                dtype=np.int32,
            ),
            np.array(
                [],
                dtype=np.float64,
            ),
        )

    # ============================================================
    # 3. LAT / LON
    # ============================================================

    dlat_deg = 180.0 / height
    dlon_deg = 360.0 / width

    lat_deg = (
        90.0
        - (np.arange(height) + 0.5)
        * dlat_deg
    )

    lon_deg = (
        -180.0
        + (np.arange(width) + 0.5)
        * dlon_deg
    )

    lat_rad = np.deg2rad(lat_deg)
    lon_rad = np.deg2rad(lon_deg)

    lat_grid, lon_grid = np.meshgrid(
        lat_rad,
        lon_rad,
        indexing="ij",
    )

    # ============================================================
    # 4. PALLOKOORDINAATIT
    # ============================================================

    cos_lat = np.cos(lat_grid)

    x = (
        cos_lat
        * np.cos(lon_grid)
    )

    y = (
        cos_lat
        * np.sin(lon_grid)
    )

    z = np.sin(lat_grid)

    # ============================================================
    # 5. KANSAKESKUKSET
    # ============================================================

    flat_valid = np.flatnonzero(
        valid_population.ravel()
    )

    xyz = np.column_stack(
        (
            x.ravel()[flat_valid],
            y.ravel()[flat_valid],
            z.ravel()[flat_valid],
        )
    )

    values = pop.ravel()[flat_valid]

    min_angle = (
        min_distance_km
        / planet_radius
    )

    cos_min_angle = np.cos(
        min_angle
    )

    available = np.ones(
        len(values),
        dtype=bool,
    )

    centers = []

    for _ in range(n_nations):

        if not np.any(available):
            break

        candidates = np.where(
            available,
            values,
            -np.inf,
        )

        idx = np.argmax(
            candidates
        )

        center = np.unravel_index(
            flat_valid[idx],
            pop.shape,
        )

        centers.append(center)

        center_xyz = xyz[idx]

        dot = xyz @ center_xyz

        too_close = (
            dot >= cos_min_angle
        )

        available[too_close] = False

    if not centers:
        return (
            [],
            np.zeros_like(
                pop,
                dtype=np.int32,
            ),
            np.array(
                [],
                dtype=np.float64,
            ),
        )

    # ============================================================
    # 6. RELIEFIN KALTEVUUS
    # ============================================================

    elevation = np.maximum(
        relief,
        0.0,
    )

    dy_km = (
        planet_radius
        * np.deg2rad(dlat_deg)
    )

    dz_dy = (
        np.gradient(
            elevation,
            axis=0,
        )
        / dy_km
    )

    # Longitude wrap

    elevation_left = np.roll(
        elevation,
        1,
        axis=1,
    )

    elevation_right = np.roll(
        elevation,
        -1,
        axis=1,
    )

    dx_km = (
        planet_radius
        * np.cos(lat_grid)
        * np.deg2rad(dlon_deg)
    )

    dx_km = np.maximum(
        dx_km,
        1e-6,
    )

    dz_dx = (
        elevation_right
        - elevation_left
    ) / (
        2.0 * dx_km
    )

    gradient = np.sqrt(
        dz_dx ** 2
        + dz_dy ** 2
    )

    slope_rad = np.arctan(
        gradient
    )

    slope_deg = np.rad2deg(
        slope_rad
    )

    # ============================================================
    # 7. MAASTON KULKUKUSTANNUS
    # ============================================================

    slope_factor = (
        slope_deg
        / slope_scale
    )

    terrain_factor = (
        1.0
        + slope_factor ** slope_power
    )

    terrain_factor[
        ~land
    ] = np.inf

    # ============================================================
    # 8. VÄESTÖN NORMALISOINTI
    # ============================================================
    #
    # Väestöbonus kasvaa nopeasti aluksi ja tasaantuu.
    #
    # population = 0
    #       -> bonus 0
    #
    # paljon väestöä
    #       -> bonus lähestyy 1
    #
    # ============================================================

    positive_population = pop[
        pop > 0
    ]

    if len(positive_population) > 0:

        # Käytetään annettua skaalaa,
        # mutta estetään nollalla jako.

        pop_scale = max(
            population_scale,
            1e-12,
        )

        population_presence = (
            1.0
            - np.exp(
                -pop / pop_scale
            )
        )

    else:

        population_presence = (
            np.zeros_like(pop)
        )

    # ============================================================
    # 9. NAAPURIT
    # ============================================================

    if diagonal:

        neighbors = (
            (-1, -1),
            (-1,  0),
            (-1,  1),
            ( 0, -1),
            ( 0,  1),
            ( 1, -1),
            ( 1,  0),
            ( 1,  1),
        )

    else:

        neighbors = (
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
        )

    # ============================================================
    # 10. TULOSKENTÄT
    # ============================================================

    nation_map = np.zeros(
        pop.shape,
        dtype=np.int32,
    )

    # Todellinen fyysinen etäisyys km

    physical_distance = np.full(
        pop.shape,
        np.inf,
        dtype=np.float64,
    )

    # Terrain-kustannus

    terrain_distance = np.full(
        pop.shape,
        np.inf,
        dtype=np.float64,
    )

    # Lopullinen Dijkstra-kustannus

    total_cost = np.full(
        pop.shape,
        np.inf,
        dtype=np.float64,
    )

    # ============================================================
    # 11. PRIORITY QUEUE
    # ============================================================

    heap = []

    for nation_id, (r, c) in enumerate(
        centers,
        start=1,
    ):

        physical_distance[r, c] = 0.0
        terrain_distance[r, c] = 0.0
        total_cost[r, c] = 0.0

        nation_map[r, c] = nation_id

        heapq.heappush(
            heap,
            (
                0.0,
                nation_id,
                r,
                c,
            ),
        )

    # ============================================================
    # 12. KANSOJEN LEVIÄMINEN
    # ============================================================

    while heap:

        (
            current_cost,
            nation_id,
            r,
            c,
        ) = heapq.heappop(heap)

        if current_cost != total_cost[r, c]:
            continue

        current_physical_distance = (
            physical_distance[r, c]
        )

        current_terrain_distance = (
            terrain_distance[r, c]
        )

        for dr, dc in neighbors:

            nr = r + dr
            nc = c + dc

            # Ei wrapia napojen yli.

            if nr < 0 or nr >= height:
                continue

            # Longitude wrap.

            nc %= width

            if not land[nr, nc]:
                continue

            # ====================================================
            # 13. PALLOPINTA
            # ====================================================

            if dr == 0:

                step_distance_km = (
                    planet_radius
                    * np.cos(
                        lat_rad[nr]
                    )
                    * np.deg2rad(
                        dlon_deg
                    )
                )

            elif dc == 0:

                step_distance_km = (
                    dy_km
                )

            else:

                dx = (
                    planet_radius
                    * np.cos(
                        lat_rad[nr]
                    )
                    * np.deg2rad(
                        dlon_deg
                    )
                )

                step_distance_km = np.sqrt(
                    dx * dx
                    + dy_km * dy_km
                )

            # ====================================================
            # 14. FYYSINEN ETÄISYYS
            # ====================================================

            new_physical_distance = (
                current_physical_distance
                + step_distance_km
            )

            # ====================================================
            # 15. RELIEF
            # ====================================================

            terrain_step = (
                step_distance_km
                * terrain_factor[nr, nc]
            )

            new_terrain_distance = (
                current_terrain_distance
                + terrain_step
            )

            # ====================================================
            # 16. VÄESTÖBONUS
            # ====================================================
            #
            # Runsas väestö tekee alueesta kulttuurisesti
            # helpommin saavutettavan.
            #
            # Bonus ei koskaan tee kustannuksesta nollaa.
            #
            # ====================================================

            presence = (
                population_presence[nr, nc]
            )

            population_factor = (
                1.0
                / (
                    1.0
                    + population_influence
                    * presence
                )
            )

            # ====================================================
            # 17. KULTTUURINEN ETÄISYYS
            # ====================================================

            culture_ratio = (
                new_physical_distance
                / culture_distance_scale_km
            )

            culture_factor = (
                1.0
                + culture_ratio ** culture_power
            )

            # ====================================================
            # 18. KOKONAISKUSTANNUS
            # ====================================================
            #
            # Terrain:
            #
            #   vaikea maasto -> kallis
            #
            # Population:
            #
            #   asuttu alue -> hieman halvempi
            #
            # Culture:
            #
            #   kaukana keskuksesta -> kallis
            #
            # ====================================================

            new_cost = (
                new_terrain_distance
                * culture_factor
                * population_factor
            )

            if new_cost < total_cost[nr, nc]:

                total_cost[nr, nc] = (
                    new_cost
                )

                physical_distance[nr, nc] = (
                    new_physical_distance
                )

                terrain_distance[nr, nc] = (
                    new_terrain_distance
                )

                nation_map[nr, nc] = (
                    nation_id
                )

                heapq.heappush(
                    heap,
                    (
                        new_cost,
                        nation_id,
                        nr,
                        nc,
                    ),
                )

    # ============================================================
    # 19. KANSOJEN VÄKILUVUT
    # ============================================================

    nation_ids = nation_map[
        valid_population
    ]

    population_values = pop[
        valid_population
    ]

    nation_populations = np.bincount(
        nation_ids,
        weights=population_values,
        minlength=len(centers) + 1,
    )[1:]

    return (
        centers,
        nation_map,
        nation_populations,
    )


def muodosta_kansat_06(
    population_in_pixels,
    relief,
    planet_radius,
    population_per_nation=100_000,
    min_distance_km=500.0,
    slope_scale=15.0,
    slope_power=2.0,
    culture_distance_scale_km=1500.0,
    culture_power=2.0,
    diagonal=True,
):
    """
    Muodostaa kansojen keskukset ja alueet koko planeetalle.

    population_in_pixels
        2D-array, shape = (height, width)

    relief
        2D-array, sama shape.
        0 = meri
        >0 = maa, metreinä

    planet_radius
        Planeetan säde kilometreinä.

    population_per_nation
        Kuinka monta ihmistä vastaa yhtä kansakeskusta.

    min_distance_km
        Kansakeskusten minimietäisyys pallopinnalla.

    slope_scale
        Kaltevuuskulma asteina, jossa maaston vaikutus
        alkaa kasvaa voimakkaasti.

    slope_power
        Kaltevuuden epälineaarisuus.

    culture_distance_scale_km
        Fyysinen/tehokas etäisyys keskuksesta, jolla
        kulttuurinen etäisyys alkaa vaikuttaa.

    culture_power
        Kulttuurisen etäisyyden voimakkuus.

    diagonal
        True  -> 8 naapuria
        False -> 4 naapuria

    Palauttaa
    ---------

    centers
        [(row, col), ...]

    nation_map
        0 = meri
        1 = kansa 1
        2 = kansa 2
        ...

    nation_populations
        Kansojen väkiluvut.
    """

    # ============================================================
    # 1. INPUT
    # ============================================================

    pop = np.asarray(
        population_in_pixels,
        dtype=np.float64,
    )

    relief = np.asarray(
        relief,
        dtype=np.float64,
    )

    if pop.ndim != 2:
        raise ValueError(
            "population_in_pixels pitää olla 2D-array."
        )

    if relief.shape != pop.shape:
        raise ValueError(
            "relief ja population_in_pixels pitää olla "
            "saman kokoisia."
        )

    if planet_radius <= 0:
        raise ValueError(
            "planet_radius pitää olla positiivinen."
        )

    height, width = pop.shape

    # ============================================================
    # 2. MAA / MERI
    # ============================================================

    land = (
        np.isfinite(relief)
        & (relief > 0)
    )

    valid_population = (
        np.isfinite(pop)
        & (pop > 0)
        & land
    )

    if not np.any(valid_population):
        return (
            [],
            np.zeros(
                pop.shape,
                dtype=np.int32,
            ),
            np.array(
                [],
                dtype=np.float64,
            ),
        )

    total_population = (
        pop[valid_population].sum()
    )

    # Esim.
    #
    # 8 700 000 / 100 000 = 87 kansaa
    #
    n_nations = int(
        total_population
        / population_per_nation
    )

    if n_nations < 1:
        return (
            [],
            np.zeros(
                pop.shape,
                dtype=np.int32,
            ),
            np.array(
                [],
                dtype=np.float64,
            ),
        )

    # ============================================================
    # 3. LAT/LON
    # ============================================================

    dlat_deg = 180.0 / height
    dlon_deg = 360.0 / width

    lat_deg = (
        90.0
        - (np.arange(height) + 0.5)
        * dlat_deg
    )

    lon_deg = (
        -180.0
        + (np.arange(width) + 0.5)
        * dlon_deg
    )

    lat_rad = np.deg2rad(lat_deg)
    lon_rad = np.deg2rad(lon_deg)

    lat_grid, lon_grid = np.meshgrid(
        lat_rad,
        lon_rad,
        indexing="ij",
    )

    # ============================================================
    # 4. YKSIKKÖPALLON KOORDINAATIT
    # ============================================================

    cos_lat = np.cos(lat_grid)

    x = (
        cos_lat
        * np.cos(lon_grid)
    )

    y = (
        cos_lat
        * np.sin(lon_grid)
    )

    z = np.sin(lat_grid)

    # ============================================================
    # 5. KANSAKESKUSTEN ETSIMINEN
    # ============================================================

    flat_valid = np.flatnonzero(
        valid_population.ravel()
    )

    xyz = np.column_stack(
        (
            x.ravel()[flat_valid],
            y.ravel()[flat_valid],
            z.ravel()[flat_valid],
        )
    )

    values = pop.ravel()[flat_valid]

    min_angle = (
        min_distance_km
        / planet_radius
    )

    cos_min_angle = np.cos(
        min_angle
    )

    available = np.ones(
        len(values),
        dtype=bool,
    )

    centers = []
    center_indices = []

    for _ in range(n_nations):

        if not np.any(available):
            break

        candidate_values = np.where(
            available,
            values,
            -np.inf,
        )

        idx = np.argmax(
            candidate_values
        )

        center = np.unravel_index(
            flat_valid[idx],
            pop.shape,
        )

        centers.append(center)
        center_indices.append(idx)

        # Poista minimietäisyyden sisältä
        # kaikki mahdolliset uudet keskukset.

        center_xyz = xyz[idx]

        dot = xyz @ center_xyz

        too_close = (
            dot >= cos_min_angle
        )

        available[too_close] = False

    if not centers:
        return (
            [],
            np.zeros(
                pop.shape,
                dtype=np.int32,
            ),
            np.array(
                [],
                dtype=np.float64,
            ),
        )

    # ============================================================
    # 6. RELIEFIN KALTEVUUS
    # ============================================================

    elevation = np.maximum(
        relief,
        0.0,
    )

    # ------------------------------------------------------------
    # Pohjois-etelä
    # ------------------------------------------------------------

    dy_km = (
        planet_radius
        * np.deg2rad(dlat_deg)
    )

    dz_dy = (
        np.gradient(
            elevation,
            axis=0,
        )
        / dy_km
    )

    # ------------------------------------------------------------
    # Itä-länsi
    #
    # Roll tekee 180/-180 saumasta jatkuvan.
    # ------------------------------------------------------------

    elevation_left = np.roll(
        elevation,
        1,
        axis=1,
    )

    elevation_right = np.roll(
        elevation,
        -1,
        axis=1,
    )

    dx_km = (
        planet_radius
        * np.cos(lat_grid)
        * np.deg2rad(dlon_deg)
    )

    dx_km = np.maximum(
        dx_km,
        1e-6,
    )

    dz_dx = (
        elevation_right
        - elevation_left
    ) / (
        2.0 * dx_km
    )

    # ------------------------------------------------------------
    # Kaltevuus
    # ------------------------------------------------------------

    gradient = np.sqrt(
        dz_dx ** 2
        + dz_dy ** 2
    )

    slope_rad = np.arctan(
        gradient
    )

    slope_deg = np.rad2deg(
        slope_rad
    )

    # ============================================================
    # 7. MAASTON KULKUKERROIN
    # ============================================================

    slope_factor = (
        slope_deg
        / slope_scale
    )

    terrain_cost = (
        1.0
        + slope_factor ** slope_power
    )

    # Meri on absoluuttinen este.

    terrain_cost[
        ~land
    ] = np.inf

    # ============================================================
    # 8. NAAPURIT
    # ============================================================

    if diagonal:

        neighbors = (
            (-1, -1),
            (-1,  0),
            (-1,  1),
            ( 0, -1),
            ( 0,  1),
            ( 1, -1),
            ( 1,  0),
            ( 1,  1),
        )

    else:

        neighbors = (
            (-1,  0),
            (1,  0),
            (0, -1),
            (0,  1),
        )

    # ============================================================
    # 9. TULOSRASTERIT
    # ============================================================

    nation_map = np.zeros(
        pop.shape,
        dtype=np.int32,
    )

    # Tämä on todellinen fyysinen etäisyys
    # keskuksesta, EI terrain-cost.

    physical_distance = np.full(
        pop.shape,
        np.inf,
        dtype=np.float64,
    )

    # Tämä on varsinainen kokonaiskustannus,
    # jota Dijkstra optimoi.

    total_cost = np.full(
        pop.shape,
        np.inf,
        dtype=np.float64,
    )

    # ============================================================
    # 10. PRIORITY QUEUE
    # ============================================================

    heap = []

    for nation_id, (r, c) in enumerate(
        centers,
        start=1,
    ):

        physical_distance[r, c] = 0.0
        total_cost[r, c] = 0.0

        nation_map[r, c] = nation_id

        heapq.heappush(
            heap,
            (
                0.0,
                0.0,
                nation_id,
                r,
                c,
            ),
        )

    # ============================================================
    # 11. KANSOJEN KASVU
    # ============================================================

    while heap:

        current_total_cost, current_physical_distance, nation_id, r, c = (
            heapq.heappop(heap)
        )

        if current_total_cost != total_cost[r, c]:
            continue

        # --------------------------------------------------------
        # Naapurit
        # --------------------------------------------------------

        for dr, dc in neighbors:

            nr = r + dr
            nc = c + dc

            # Napoja ei wrapata.

            if nr < 0 or nr >= height:
                continue

            # Longitude wrapataan.

            nc %= width

            if not land[nr, nc]:
                continue

            # ----------------------------------------------------
            # Todellinen pallopinnan askelpituus
            # ----------------------------------------------------

            if dr == 0:

                # Itä-länsi.

                step_distance_km = (
                    planet_radius
                    * np.cos(
                        lat_rad[nr]
                    )
                    * np.deg2rad(
                        dlon_deg
                    )
                )

            elif dc == 0:

                # Pohjois-etelä.

                step_distance_km = dy_km

            else:

                # Diagonaalinen askel.

                dx = (
                    planet_radius
                    * np.cos(
                        lat_rad[nr]
                    )
                    * np.deg2rad(
                        dlon_deg
                    )
                )

                step_distance_km = np.sqrt(
                    dx * dx
                    + dy_km * dy_km
                )

            # ----------------------------------------------------
            # Maaston vaikutus
            # ----------------------------------------------------

            terrain_factor = (
                terrain_cost[nr, nc]
            )

            terrain_distance = (
                step_distance_km
                * terrain_factor
            )

            # ----------------------------------------------------
            # Fyysinen etäisyys
            #
            # Tätä käytetään kulttuurisen etäisyyden
            # laskemiseen.
            # ----------------------------------------------------

            new_physical_distance = (
                current_physical_distance
                + step_distance_km
            )

            # ----------------------------------------------------
            # Maaston aiheuttama kokonaiskustannus
            # ----------------------------------------------------

            new_terrain_cost = (
                current_total_cost
                + terrain_distance
            )

            # ----------------------------------------------------
            # Kulttuurinen etäisyys
            #
            # Tärkeää:
            #
            # tämä perustuu fyysiseen etäisyyteen,
            # ei terrain-kustannukseen.
            # ----------------------------------------------------

            culture_ratio = (
                new_physical_distance
                / culture_distance_scale_km
            )

            culture_factor = (
                1.0
                + culture_ratio ** culture_power
            )

            # ----------------------------------------------------
            # Lopullinen kustannus
            # ----------------------------------------------------
            #
            # Maaston kustannus * kulttuurinen etäisyys
            #
            # Keskuksen lähellä:
            #
            #     culture_factor ≈ 1
            #
            # Kauempana:
            #
            #     culture_factor kasvaa
            #
            # ----------------------------------------------------

            new_total_cost = (
                new_terrain_cost
                * culture_factor
            )

            # ----------------------------------------------------
            # Päivitä jos tämä reitti on parempi.
            # ----------------------------------------------------

            if new_total_cost < total_cost[nr, nc]:

                total_cost[nr, nc] = (
                    new_total_cost
                )

                physical_distance[nr, nc] = (
                    new_physical_distance
                )

                nation_map[nr, nc] = (
                    nation_id
                )

                heapq.heappush(
                    heap,
                    (
                        new_total_cost,
                        new_physical_distance,
                        nation_id,
                        nr,
                        nc,
                    ),
                )

    # ============================================================
    # 12. KANSOJEN VÄKILUVUT
    # ============================================================

    nation_ids = nation_map[
        valid_population
    ]

    population_values = pop[
        valid_population
    ]

    nation_populations = np.bincount(
        nation_ids,
        weights=population_values,
        minlength=len(centers) + 1,
    )[1:]

    return (
        centers,
        nation_map,
        nation_populations,
    )



def muodosta_kansat_05(
    population_in_pixels,
    relief,
    planet_radius,
    population_per_nation=100_000,
    min_distance_km=500.0,
    slope_scale=15.0,
    slope_power=2.0,
    culture_distance_scale_km=1500.0,
    culture_power=2.0,
    diagonal=True,
):
    """
    Muodostaa kansojen keskukset ja alueet koko planeetalle.

    PARAMETRIT
    ----------

    population_in_pixels : 2D ndarray
        Väestökenttä, shape = (height, width).

    relief : 2D ndarray
        Korkeuskenttä metreinä, sama shape kuin population.

        0 = meri
        >0 = maa

    planet_radius : float
        Planeetan säde kilometreinä.

    population_per_nation : float
        Keskimääräinen väkiluku yhtä kansaa kohti.

        Esim. 100_000.

    min_distance_km : float
        Kansakeskusten pienin sallittu etäisyys km.

    slope_scale : float
        Kaltevuuskulma asteina, jossa maaston kustannus
        alkaa kasvaa voimakkaasti.

    slope_power : float
        Kaltevuuden vaikutuksen epälineaarisuus.

    culture_distance_scale_km : float
        Mittakaava, jolla etäisyys kansan keskuksesta
        alkaa vaikuttaa voimakkaasti.

        Pieni arvo:
            paikallisia pieniä kansoja.

        Suuri arvo:
            laajalle leviäviä kansoja.

    culture_power : float
        Kulttuurisen etäisyyden epälineaarisuus.

    diagonal : bool
        True:
            8 naapuria

        False:
            4 naapuria


    PALAUTUKSET
    -----------

    centers :
        Lista:

            [(row, col), ...]

        Kansa 1 = centers[0]
        Kansa 2 = centers[1]
        jne.

    nation_map :
        2D ndarray.

            0 = meri
            1 = kansa 1
            2 = kansa 2
            ...

    nation_populations :
        1D ndarray.

            nation_populations[0] = kansa 1
            nation_populations[1] = kansa 2
            ...
    """

    # ============================================================
    # 1. INPUT
    # ============================================================

    pop = np.asarray(
        population_in_pixels,
        dtype=np.float64,
    )

    relief = np.asarray(
        relief,
        dtype=np.float64,
    )

    if pop.ndim != 2:
        raise ValueError(
            "population_in_pixels pitää olla 2D-array."
        )

    if relief.shape != pop.shape:
        raise ValueError(
            "relief ja population_in_pixels pitää olla "
            "saman kokoisia."
        )

    if planet_radius <= 0:
        raise ValueError(
            "planet_radius pitää olla > 0."
        )

    if population_per_nation <= 0:
        raise ValueError(
            "population_per_nation pitää olla > 0."
        )

    if min_distance_km < 0:
        raise ValueError(
            "min_distance_km ei voi olla negatiivinen."
        )

    if culture_distance_scale_km <= 0:
        raise ValueError(
            "culture_distance_scale_km pitää olla > 0."
        )

    height, width = pop.shape

    # ============================================================
    # 2. MAA / MERI
    # ============================================================

    land = (
        np.isfinite(relief)
        & (relief > 0)
    )

    valid_population = (
        np.isfinite(pop)
        & (pop > 0)
        & land
    )

    if not np.any(valid_population):
        return (
            [],
            np.zeros_like(
                pop,
                dtype=np.int32,
            ),
            np.array(
                [],
                dtype=np.float64,
            ),
        )

    total_population = (
        pop[valid_population].sum()
    )

    # Alussa sovittu periaate:
    #
    #   100 000 ihmistä -> yksi potentiaalinen kansa
    #
    n_nations = int(
        total_population
        / population_per_nation
    )

    if n_nations < 1:
        return (
            [],
            np.zeros_like(
                pop,
                dtype=np.int32,
            ),
            np.array(
                [],
                dtype=np.float64,
            ),
        )

    # ============================================================
    # 3. RASTERIN MAANTIETEELLISET MITAT
    # ============================================================

    dlat_deg = 180.0 / height
    dlon_deg = 360.0 / width

    lat_deg = (
        90.0
        - (np.arange(height) + 0.5)
        * dlat_deg
    )

    lon_deg = (
        -180.0
        + (np.arange(width) + 0.5)
        * dlon_deg
    )

    lat_rad = np.deg2rad(lat_deg)
    lon_rad = np.deg2rad(lon_deg)

    lat_grid, lon_grid = np.meshgrid(
        lat_rad,
        lon_rad,
        indexing="ij",
    )

    # ============================================================
    # 4. PALLOKOORDINAATIT
    # ============================================================

    cos_lat = np.cos(lat_grid)

    x = (
        cos_lat
        * np.cos(lon_grid)
    )

    y = (
        cos_lat
        * np.sin(lon_grid)
    )

    z = np.sin(lat_grid)

    # ============================================================
    # 5. KANSAKESKUSTEN ETSIMINEN
    # ============================================================

    flat_valid = np.flatnonzero(
        valid_population.ravel()
    )

    xyz = np.column_stack(
        (
            x.ravel()[flat_valid],
            y.ravel()[flat_valid],
            z.ravel()[flat_valid],
        )
    )

    values = pop.ravel()[flat_valid]

    # ------------------------------------------------------------
    # Minimietäisyys pallopinnalla
    # ------------------------------------------------------------

    min_angle = (
        min_distance_km
        / planet_radius
    )

    cos_min_angle = np.cos(
        min_angle
    )

    available = np.ones(
        len(values),
        dtype=bool,
    )

    centers = []
    center_indices = []

    for _ in range(n_nations):

        if not np.any(available):
            break

        candidate_values = np.where(
            available,
            values,
            -np.inf,
        )

        idx = np.argmax(
            candidate_values
        )

        center = np.unravel_index(
            flat_valid[idx],
            pop.shape,
        )

        centers.append(center)
        center_indices.append(idx)

        # --------------------------------------------------------
        # Kaikki liian lähellä olevat pisteet pois.
        # --------------------------------------------------------

        center_xyz = xyz[idx]

        dot = xyz @ center_xyz

        too_close = (
            dot >= cos_min_angle
        )

        available[too_close] = False

    if not centers:
        return (
            [],
            np.zeros_like(
                pop,
                dtype=np.int32,
            ),
            np.array(
                [],
                dtype=np.float64,
            ),
        )

    # ============================================================
    # 6. KALTEVUUDEN LASKENTA
    # ============================================================

    elevation = np.maximum(
        relief,
        0.0,
    )

    # ------------------------------------------------------------
    # Pohjois-eteläsuuntainen matka
    # ------------------------------------------------------------

    dy_km = (
        planet_radius
        * np.deg2rad(dlat_deg)
    )

    dz_dy = (
        np.gradient(
            elevation,
            axis=0,
        )
        / dy_km
    )

    # ------------------------------------------------------------
    # Itä-länsisuuntainen gradientti.
    #
    # Longitude-wrap:
    #
    # vasemman reunan naapuri =
    # oikean reunan pikseli.
    # ------------------------------------------------------------

    elevation_left = np.roll(
        elevation,
        1,
        axis=1,
    )

    elevation_right = np.roll(
        elevation,
        -1,
        axis=1,
    )

    dx_km = (
        planet_radius
        * np.cos(lat_grid)
        * np.deg2rad(dlon_deg)
    )

    dx_km = np.maximum(
        dx_km,
        1e-6,
    )

    dz_dx = (
        elevation_right
        - elevation_left
    ) / (
        2.0 * dx_km
    )

    # ------------------------------------------------------------
    # Kokonaiskaltevuus
    # ------------------------------------------------------------

    gradient = np.sqrt(
        dz_dx ** 2
        + dz_dy ** 2
    )

    slope_rad = np.arctan(
        gradient
    )

    slope_deg = np.rad2deg(
        slope_rad
    )

    # ============================================================
    # 7. MAASTON KULKUKUSTANNUS
    # ============================================================

    slope_factor = (
        slope_deg
        / slope_scale
    )

    terrain_cost = (
        1.0
        + slope_factor ** slope_power
    )

    # Meri = ehdoton este.

    terrain_cost[
        ~land
    ] = np.inf

    # ============================================================
    # 8. NAAPURIT
    # ============================================================

    if diagonal:

        neighbors = (
            (-1, -1),
            (-1,  0),
            (-1,  1),
            ( 0, -1),
            ( 0,  1),
            ( 1, -1),
            ( 1,  0),
            ( 1,  1),
        )

    else:

        neighbors = (
            (-1,  0),
            ( 1,  0),
            ( 0, -1),
            ( 0,  1),
        )

    # ============================================================
    # 9. KANSARASTERI
    # ============================================================

    nation_map = np.zeros(
        pop.shape,
        dtype=np.int32,
    )

    # ============================================================
    # 10. DIJKSTRAN ETÄISYYS
    # ============================================================

    distance = np.full(
        pop.shape,
        np.inf,
        dtype=np.float64,
    )

    # ============================================================
    # 11. KULTTUURINEN ETÄISYYS
    # ============================================================
    #
    # distance_km kertoo fyysisen least-cost-etäisyyden
    # kansan keskuksesta.
    #
    # Kulttuurinen kustannus kasvaa sen mukana.
    #
    # Esimerkiksi:
    #
    #   d = 0
    #       culture = 1
    #
    #   d = scale
    #       culture = 2
    #
    #   d = 2*scale
    #       culture = 5
    #
    # ============================================================

    def culture_factor(distance_km):
        """
        Etäisyydestä riippuva kulttuurinen kerroin.
        """

        x = (
            distance_km
            / culture_distance_scale_km
        )

        return (
            1.0
            + x ** culture_power
        )

    # ============================================================
    # 12. PRIORITY QUEUE
    # ============================================================

    heap = []

    for nation_id, (r, c) in enumerate(
        centers,
        start=1,
    ):

        distance[r, c] = 0.0

        nation_map[r, c] = nation_id

        heapq.heappush(
            heap,
            (
                0.0,
                nation_id,
                r,
                c,
            ),
        )

    # ============================================================
    # 13. KANSOJEN LEVIÄMINEN
    # ============================================================

    while heap:

        current_cost, nation_id, r, c = (
            heapq.heappop(heap)
        )

        if current_cost != distance[r, c]:
            continue

        for dr, dc in neighbors:

            nr = r + dr
            nc = c + dc

            # ----------------------------------------------------
            # Pohjois / etelä:
            # ei wrapia.
            # ----------------------------------------------------

            if nr < 0 or nr >= height:
                continue

            # ----------------------------------------------------
            # Longitude:
            # wrapataan.
            # ----------------------------------------------------

            nc %= width

            if not land[nr, nc]:
                continue

            # ====================================================
            # 14. PALLON PINTAMATKA
            # ====================================================

            if dr == 0 and dc != 0:

                step_distance_km = (
                    planet_radius
                    * np.cos(
                        lat_rad[nr]
                    )
                    * np.deg2rad(
                        dlon_deg
                    )
                )

            elif dc == 0 and dr != 0:

                step_distance_km = (
                    dy_km
                )

            else:

                dx = (
                    planet_radius
                    * np.cos(
                        lat_rad[nr]
                    )
                    * np.deg2rad(
                        dlon_deg
                    )
                )

                step_distance_km = np.sqrt(
                    dx * dx
                    + dy_km * dy_km
                )

            # ====================================================
            # 15. RELIEF
            # ====================================================

            local_terrain_cost = (
                terrain_cost[nr, nc]
            )

            terrain_step = (
                step_distance_km
                * local_terrain_cost
            )

            # ====================================================
            # 16. KULTTUURINEN ETÄISYYS
            # ====================================================
            #
            # Arvioidaan ensin fyysinen etäisyys.
            # ====================================================

            physical_distance = (
                current_cost
                + terrain_step
            )

            culture = culture_factor(
                physical_distance
            )

            # ====================================================
            # 17. KOKONAISKUSTANNUS
            # ====================================================
            #
            # Perusajatus:
            #
            #   maasto
            #       *
            #   kulttuurinen etäisyys
            #
            # ====================================================

            new_cost = (
                physical_distance
                * culture
            )

            if new_cost < distance[nr, nc]:

                distance[nr, nc] = new_cost

                nation_map[nr, nc] = (
                    nation_id
                )

                heapq.heappush(
                    heap,
                    (
                        new_cost,
                        nation_id,
                        nr,
                        nc,
                    ),
                )

    # ============================================================
    # 18. KANSOJEN VÄKILUVUT
    # ============================================================

    nation_ids = nation_map[
        valid_population
    ]

    population_values = pop[
        valid_population
    ]

    nation_populations = np.bincount(
        nation_ids,
        weights=population_values,
        minlength=len(centers) + 1,
    )[1:]

    return (
        centers,
        nation_map,
        nation_populations,
    )





def muodosta_kansat_04(
    population_in_pixels,
    relief,
    planet_radius,
    population_per_nation=100_000,
    min_distance_km=500.0,
    slope_scale=15.0,
    slope_power=2.0,
    diagonal=True,
):
    """
    Muodostaa kansojen keskukset ja alueet planeetan pinnalla.

    population_in_pixels:
        2D numpy-array, shape (height, width)

    relief:
        2D numpy-array, sama shape.
        Metrejä.
        0 = meri
        >0 = maa

    planet_radius:
        Planeetan säde kilometreinä.

    population_per_nation:
        Keskimääräinen väestömäärä yhtä kansaa kohti.

    min_distance_km:
        Kansakeskusten minimietäisyys planeetan pinnalla.

    slope_scale:
        Kaltevuuskulma asteina, jonka kohdalla rinne alkaa
        vaikuttaa voimakkaasti kulkukustannukseen.

    slope_power:
        Kaltevuuskustannuksen epälineaarisuus.

    diagonal:
        Sallitaanko 8-suuntainen liikkuminen.

    Returns
    -------
    centers
        Lista:
            [(row, col), ...]

    nation_map
        0 = meri / ei aluetta
        1 = kansa 1
        2 = kansa 2
        ...

    nation_populations
        Kansojen lopulliset väkiluvut.
    """

    # ============================================================
    # 1. Tarkistukset
    # ============================================================

    pop = np.asarray(
        population_in_pixels,
        dtype=np.float64,
    )

    relief = np.asarray(
        relief,
        dtype=np.float64,
    )

    if pop.ndim != 2:
        raise ValueError(
            "population_in_pixels pitää olla 2D-array."
        )

    if relief.shape != pop.shape:
        raise ValueError(
            "relief ja population_in_pixels pitää olla "
            "saman kokoisia."
        )

    if planet_radius <= 0:
        raise ValueError(
            "planet_radius pitää olla positiivinen."
        )

    height, width = pop.shape

    # ============================================================
    # 2. Maa / meri
    # ============================================================

    land = (
        np.isfinite(relief)
        & (relief > 0)
    )

    valid_population = (
        np.isfinite(pop)
        & (pop > 0)
        & land
    )

    total_population = pop[valid_population].sum()

    n_nations = int(
        total_population / population_per_nation
    )

    if n_nations < 1:
        return (
            [],
            np.zeros_like(pop, dtype=np.int32),
            np.array([], dtype=np.float64),
        )

    # ============================================================
    # 3. Maapallon koordinaatit
    # ============================================================

    dlat_deg = 180.0 / height
    dlon_deg = 360.0 / width

    lat_deg = (
        90.0
        - (np.arange(height) + 0.5) * dlat_deg
    )

    lon_deg = (
        -180.0
        + (np.arange(width) + 0.5) * dlon_deg
    )

    lat_rad = np.deg2rad(lat_deg)
    lon_rad = np.deg2rad(lon_deg)

    lat_grid, lon_grid = np.meshgrid(
        lat_rad,
        lon_rad,
        indexing="ij",
    )

    # ============================================================
    # 4. Pallopisteet
    # ============================================================

    cos_lat = np.cos(lat_grid)

    x = (
        cos_lat
        * np.cos(lon_grid)
    )

    y = (
        cos_lat
        * np.sin(lon_grid)
    )

    z = np.sin(lat_grid)

    # ============================================================
    # 5. Etsi kansojen keskukset
    # ============================================================

    flat_valid = np.flatnonzero(
        valid_population.ravel()
    )

    xyz = np.column_stack((
        x.ravel()[flat_valid],
        y.ravel()[flat_valid],
        z.ravel()[flat_valid],
    ))

    values = pop.ravel()[flat_valid]

    # Pallopinnan kulmaetäisyys.

    min_angle = (
        min_distance_km
        / planet_radius
    )

    cos_min_angle = np.cos(
        min_angle
    )

    available = np.ones(
        len(values),
        dtype=bool,
    )

    centers = []
    center_indices = []

    for _ in range(n_nations):

        if not np.any(available):
            break

        candidate_values = np.where(
            available,
            values,
            -np.inf,
        )

        idx = np.argmax(
            candidate_values
        )

        center = np.unravel_index(
            flat_valid[idx],
            pop.shape,
        )

        centers.append(center)
        center_indices.append(idx)

        # Estetään seuraavat keskukset liian läheltä.

        center_xyz = xyz[idx]

        dot = xyz @ center_xyz

        too_close = (
            dot >= cos_min_angle
        )

        available[too_close] = False

    # ============================================================
    # 6. Lasketaan maaston kaltevuus
    # ============================================================
    #
    # Korkeus itsessään EI aiheuta kustannusta.
    #
    # Ainoastaan korkeuden muutos eli rinne.
    #
    # ============================================================

    elevation = np.maximum(
        relief,
        0.0,
    )

    # ------------------------------------------------------------
    # Pohjois-eteläsuuntainen gradientti
    # ------------------------------------------------------------

    dy_km = (
        planet_radius
        * np.deg2rad(dlat_deg)
    )

    dz_dy = (
        np.gradient(
            elevation,
            axis=0,
        )
        / dy_km
    )

    # ------------------------------------------------------------
    # Itä-länsisuuntainen gradientti
    #
    # Longitude on syytä wrapata.
    # ------------------------------------------------------------

    elevation_left = np.roll(
        elevation,
        1,
        axis=1,
    )

    elevation_right = np.roll(
        elevation,
        -1,
        axis=1,
    )

    # Pikselin itä-länsisuuntainen fyysinen leveys.

    dx_km = (
        planet_radius
        * np.cos(lat_grid)
        * np.deg2rad(dlon_deg)
    )

    dx_km = np.maximum(
        dx_km,
        1e-6,
    )

    dz_dx = (
        elevation_right
        - elevation_left
    ) / (
        2.0 * dx_km
    )

    # ------------------------------------------------------------
    # Kokonaiskaltevuus
    # ------------------------------------------------------------

    gradient = np.sqrt(
        dz_dx ** 2
        + dz_dy ** 2
    )

    slope_rad = np.arctan(
        gradient
    )

    slope_deg = np.rad2deg(
        slope_rad
    )

    # ============================================================
    # 7. Kulkukustannus
    # ============================================================
    #
    # Tasanko:
    #
    #     cost = 1
    #
    # Rinne:
    #
    #     cost = 1 + (slope / slope_scale)^power
    #
    # ============================================================

    slope_factor = (
        slope_deg
        / slope_scale
    )

    movement_cost = (
        1.0
        + slope_factor ** slope_power
    )

    # Meri on kokonaan läpipääsemätön.

    movement_cost[
        ~land
    ] = np.inf

    # ============================================================
    # 8. Naapurit
    # ============================================================

    if diagonal:
        neighbors = (
            (-1, -1),
            (-1,  0),
            (-1,  1),
            ( 0, -1),
            ( 0,  1),
            ( 1, -1),
            ( 1,  0),
            ( 1,  1),
        )
    else:
        neighbors = (
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
        )

    # ============================================================
    # 9. Monilähde-Dijkstra
    # ============================================================

    nation_map = np.zeros(
        pop.shape,
        dtype=np.int32,
    )

    distance = np.full(
        pop.shape,
        np.inf,
        dtype=np.float64,
    )

    heap = []

    for nation_id, (r, c) in enumerate(
        centers,
        start=1,
    ):

        distance[r, c] = 0.0

        nation_map[r, c] = nation_id

        heapq.heappush(
            heap,
            (
                0.0,
                nation_id,
                r,
                c,
            ),
        )

    # ============================================================
    # 10. Kansojen leviäminen
    # ============================================================

    while heap:

        current_cost, nation_id, r, c = (
            heapq.heappop(heap)
        )

        if current_cost != distance[r, c]:
            continue

        for dr, dc in neighbors:

            nr = r + dr
            nc = c + dc

            # Napoja ei wrapata.

            if nr < 0 or nr >= height:
                continue

            # Päivämääräraja wrapataan.

            nc %= width

            if not land[nr, nc]:
                continue

            # ----------------------------------------------------
            # Todellinen pallopinnan askelpituus
            # ----------------------------------------------------

            if dr == 0 and dc != 0:

                # Itä-länsisuunta.

                step_distance_km = (
                    planet_radius
                    * np.cos(lat_rad[nr])
                    * np.deg2rad(dlon_deg)
                )

            elif dc == 0 and dr != 0:

                # Pohjois-eteläsuunta.

                step_distance_km = dy_km

            else:

                # Diagonaali.
                #
                # Likimääräinen yhdistelmä.

                dx = (
                    planet_radius
                    * np.cos(lat_rad[nr])
                    * np.deg2rad(dlon_deg)
                )

                step_distance_km = np.sqrt(
                    dx * dx
                    + dy_km * dy_km
                )

            # ----------------------------------------------------
            # Maaston kustannus
            # ----------------------------------------------------

            terrain_cost = (
                movement_cost[nr, nc]
            )

            # Normalisoidaan kilometreiksi.
            #
            # Tasangolla cost = 1,
            # joten todellinen etäisyys säilyy.
            #
            step_cost = (
                step_distance_km
                * terrain_cost
            )

            new_cost = (
                current_cost
                + step_cost
            )

            # ----------------------------------------------------
            # Päivitä jos tämä kansa pääsee halvemmalla.
            # ----------------------------------------------------

            if new_cost < distance[nr, nc]:

                distance[nr, nc] = new_cost

                nation_map[nr, nc] = nation_id

                heapq.heappush(
                    heap,
                    (
                        new_cost,
                        nation_id,
                        nr,
                        nc,
                    ),
                )

    # ============================================================
    # 11. Kansojen väkiluvut
    # ============================================================

    nation_ids = nation_map[
        valid_population
    ]

    population_values = pop[
        valid_population
    ]

    nation_populations = np.bincount(
        nation_ids,
        weights=population_values,
        minlength=len(centers) + 1,
    )[1:]

    return (
        centers,
        nation_map,
        nation_populations,
    )



def muodosta_kansat_02(
    population_in_pixels,
    relief,
    planet_radius,
    population_per_nation=100_000,
    min_distance_km=500.0,
    elevation_scale=3000.0,
    elevation_power=2.0,
    slope_scale=15.0,
    slope_power=2.0,
    diagonal=True,
):
    """
    Muodostaa kansojen keskukset ja alueet koko planeetan
    väestö- ja korkeusrasterista.

    population_in_pixels
        Väestörasteri, shape = (height, width).

    relief
        Korkeusrasteri metreinä, sama shape kuin population.

        0 = meri
        >0 = maa

    planet_radius
        Planeetan säde kilometreinä.

    population_per_nation
        Keskimääräinen väestömäärä yhtä kansakeskusta kohti.

    min_distance_km
        Kansakeskusten pienin pallopintaetäisyys.

    elevation_scale
        Korkeus, jolla korkeuden vaikutus alkaa olla
        merkittävä.

    elevation_power
        Korkeusvaikutuksen voimakkuus.

    slope_scale
        Kaltevuuskulma asteina, jolla kaltevuuden vaikutus
        alkaa olla merkittävä.

    slope_power
        Kaltevuusvaikutuksen voimakkuus.

    diagonal
        True  -> 8-suuntaiset naapurit
        False -> 4-suuntaiset naapurit

    Returns
    -------
    centers
        Lista muodossa:

            [(row, col), ...]

    nation_map
        0 = meri
        1 = kansa 1
        2 = kansa 2
        ...

    nation_populations
        Kunkin kansan alueelle jäävä väestö.
    """

    # ============================================================
    # 1. Tarkistukset
    # ============================================================

    pop = np.asarray(
        population_in_pixels,
        dtype=np.float64,
    )

    relief = np.asarray(
        relief,
        dtype=np.float64,
    )

    if pop.ndim != 2:
        raise ValueError(
            "population_in_pixels pitää olla 2D-array."
        )

    if relief.shape != pop.shape:
        raise ValueError(
            "relief ja population_in_pixels pitää olla "
            "saman kokoisia."
        )

    if planet_radius <= 0:
        raise ValueError(
            "planet_radius pitää olla > 0."
        )

    if population_per_nation <= 0:
        raise ValueError(
            "population_per_nation pitää olla > 0."
        )

    height, width = pop.shape

    # ============================================================
    # 2. Maa / meri
    # ============================================================

    land = (
        np.isfinite(relief)
        & (relief > 0)
    )

    valid_population = (
        np.isfinite(pop)
        & (pop > 0)
        & land
    )

    total_population = pop[valid_population].sum()

    n_nations = int(
        total_population / population_per_nation
    )

    if n_nations < 1:
        return (
            [],
            np.zeros_like(pop, dtype=np.int32),
            np.array([], dtype=np.float64),
        )

    # ============================================================
    # 3. Pikselien latitude
    # ============================================================

    dlat = 180.0 / height
    dlon = 360.0 / width

    lat_deg = (
        90.0
        - (np.arange(height) + 0.5) * dlat
    )

    lat_rad = np.deg2rad(lat_deg)

    # ============================================================
    # 4. Pallokoordinaatit
    # ============================================================

    lon_deg = (
        -180.0
        + (np.arange(width) + 0.5) * dlon
    )

    lon_rad = np.deg2rad(lon_deg)

    lat_grid, lon_grid = np.meshgrid(
        lat_rad,
        lon_rad,
        indexing="ij",
    )

    cos_lat = np.cos(lat_grid)

    x = (
        cos_lat
        * np.cos(lon_grid)
    )

    y = (
        cos_lat
        * np.sin(lon_grid)
    )

    z = np.sin(lat_grid)

    # ============================================================
    # 5. Etsi kansakeskukset
    # ============================================================

    flat_valid = np.flatnonzero(
        valid_population.ravel()
    )

    xyz = np.column_stack((
        x.ravel()[flat_valid],
        y.ravel()[flat_valid],
        z.ravel()[flat_valid],
    ))

    values = pop.ravel()[flat_valid]

    # Minimietäisyys pallon pinnalla.

    min_angle = (
        min_distance_km
        / planet_radius
    )

    cos_min_angle = np.cos(
        min_angle
    )

    available = np.ones(
        len(values),
        dtype=bool,
    )

    centers = []
    center_indices = []

    for _ in range(n_nations):

        if not np.any(available):
            break

        candidate_values = np.where(
            available,
            values,
            -np.inf,
        )

        idx = np.argmax(
            candidate_values
        )

        center = np.unravel_index(
            flat_valid[idx],
            pop.shape,
        )

        centers.append(center)
        center_indices.append(idx)

        center_xyz = xyz[idx]

        # Kulmaetäisyys pistetulon avulla.

        dot = xyz @ center_xyz

        too_close = (
            dot >= cos_min_angle
        )

        available[too_close] = False

    # ============================================================
    # 6. Korkeuden aiheuttama kustannus
    # ============================================================

    elevation = np.maximum(
        relief,
        0.0,
    )

    elevation_factor = (
        elevation
        / elevation_scale
    )

    elevation_cost = (
        elevation_factor
        ** elevation_power
    )

    # ============================================================
    # 7. Lasketaan kaltevuus
    # ============================================================
    #
    # Kaltevuus ei saa käyttää tavallista dx/dy-etäisyyttä,
    # koska pikselin fyysinen leveys riippuu leveysasteesta.
    #
    # Käytetään pallon pintamatkaa.
    #
    # Pohjois-eteläsuunnassa:
    #
    #     distance = R * dlat
    #
    # Itä-länsisuunnassa:
    #
    #     distance = R * cos(lat) * dlon
    #
    # ============================================================

    lat_abs = np.abs(
        lat_grid
    )

    # Meridiaalisuuntainen pikselimatka.

    dy_km = (
        planet_radius
        * np.deg2rad(dlat)
    )

    # Itä-länsisuuntainen pikselimatka.
    #
    # Napa-alueella tämä lähestyy nollaa.

    dx_km = (
        planet_radius
        * np.cos(lat_abs)
        * np.deg2rad(dlon)
    )

    dx_km = np.maximum(
        dx_km,
        1e-6,
    )

    # Gradientit metreinä / km.

    dz_dy = np.gradient(
        elevation,
        axis=0,
    ) / dy_km

    # Longitude-gradientti täytyy wrapata.

    elevation_wrapped = np.concatenate(
        (
            elevation[:, -1:],
            elevation,
            elevation[:, :1],
        ),
        axis=1,
    )

    dz_dx = (
        elevation_wrapped[:, 2:]
        - elevation_wrapped[:, :-2]
    ) / (
        2.0 * dx_km
    )

    # Kokonaisgradientti.

    gradient = np.sqrt(
        dz_dx ** 2
        + dz_dy ** 2
    )

    # Gradientti on metrejä / km.
    #
    # Muutetaan se kulmaksi:
    #
    # slope = atan(gradient)

    slope_rad = np.arctan(
        gradient
    )

    slope_deg = np.rad2deg(
        slope_rad
    )

    # ============================================================
    # 8. Kaltevuuskustannus
    # ============================================================

    slope_factor = (
        slope_deg
        / slope_scale
    )

    slope_cost = (
        slope_factor
        ** slope_power
    )

    # ============================================================
    # 9. Kokonaiskustannus
    # ============================================================

    # Perusliikkumiskustannus = 1
    #
    # Tämän päälle tulee:
    #
    #   korkeus
    #   +
    #   kaltevuus
    #
    movement_cost = (
        1.0
        + elevation_cost
        + slope_cost
    )

    # Meri on täysin läpipääsemätön.

    movement_cost[
        ~land
    ] = np.inf

    # ============================================================
    # 10. Naapurit
    # ============================================================

    if diagonal:
        neighbors = (
            (-1, -1),
            (-1,  0),
            (-1,  1),
            ( 0, -1),
            ( 0,  1),
            ( 1, -1),
            ( 1,  0),
            ( 1,  1),
        )
    else:
        neighbors = (
            (-1,  0),
            ( 1,  0),
            ( 0, -1),
            ( 0,  1),
        )

    # ============================================================
    # 11. Monilähde-Dijkstra
    # ============================================================

    nation_map = np.zeros(
        pop.shape,
        dtype=np.int32,
    )

    distance = np.full(
        pop.shape,
        np.inf,
        dtype=np.float64,
    )

    heap = []

    # Kaikki keskukset aloittavat yhtä aikaa.

    for nation_id, (r, c) in enumerate(
        centers,
        start=1,
    ):

        distance[r, c] = 0.0

        nation_map[r, c] = nation_id

        heapq.heappush(
            heap,
            (
                0.0,
                nation_id,
                r,
                c,
            ),
        )

    # ============================================================
    # 12. Alueiden kasvattaminen
    # ============================================================

    while heap:

        current_cost, nation_id, r, c = (
            heapq.heappop(heap)
        )

        if current_cost != distance[r, c]:
            continue

        for dr, dc in neighbors:

            nr = r + dr
            nc = c + dc

            # Pohjois/etelä eivät wrapaa.

            if nr < 0 or nr >= height:
                continue

            # Longitude wrapaa.

            nc %= width

            if not land[nr, nc]:
                continue

            # ----------------------------------------------------
            # Liikkumisen geometrinen kustannus
            # ----------------------------------------------------

            if dr != 0 and dc != 0:
                # Diagonaalinen liike.
                #
                # Käytetään likimääräistä sqrt(2)-kerrointa.

                geometric_factor = np.sqrt(2.0)
            else:
                geometric_factor = 1.0

            step_cost = (
                movement_cost[nr, nc]
                * geometric_factor
            )

            new_cost = (
                current_cost
                + step_cost
            )

            # ----------------------------------------------------
            # Päivitä jos tämä kansa pääsee halvemmalla.
            # ----------------------------------------------------

            if new_cost < distance[nr, nc]:

                distance[nr, nc] = new_cost

                nation_map[nr, nc] = nation_id

                heapq.heappush(
                    heap,
                    (
                        new_cost,
                        nation_id,
                        nr,
                        nc,
                    ),
                )

    # ============================================================
    # 13. Kansojen väkiluvut
    # ============================================================

    nation_ids = (
        nation_map[valid_population]
    )

    population_values = (
        pop[valid_population]
    )

    nation_populations = np.bincount(
        nation_ids,
        weights=population_values,
        minlength=len(centers) + 1,
    )[1:]

    return (
        centers,
        nation_map,
        nation_populations,
    )



def muodosta_kansat_01(
    population_in_pixels,
    relief,
    planet_radius,
    population_per_nation=100_000,
    min_distance_km=500.0,
    relief_scale=1000.0,
    relief_power=2.0,
):
    """
    Muodostaa kansojen keskukset ja alueet koko planeetan
    population- ja relief-rasterista.

    population_in_pixels
        2D-väestörasteri muodossa (height, width).

    relief
        2D-korkeusrasteri samassa koossa.

        0 = meri
        >0 = maa metreinä

    planet_radius
        Planeetan säde kilometreinä.

    population_per_nation
        Yksi kansa / tämä määrä ihmisiä.

    min_distance_km
        Kansakeskusten minimietäisyys pallopinnalla.

    relief_scale
        Korkeus metreinä, jolla reliefin vaikutus alkaa
        olla merkittävä.

        Esim. 1000 tarkoittaa, että 1000 m korkeudessa
        kustannus kasvaa selvästi.

    relief_power
        Relief-kustannuksen epälineaarisuus.

        1 = lineaarinen
        2 = voimakkaasti kasvava
        3 = erittäin voimakas

    Palauttaa
    ----------
    centers
        [(row, col), ...]

    nation_map
        0 = meri
        1 = kansa 1
        2 = kansa 2
        ...

    nation_populations
        Kansojen lopulliset väkiluvut.
    """

    pop = np.asarray(
        population_in_pixels,
        dtype=np.float64,
    )

    relief = np.asarray(
        relief,
        dtype=np.float64,
    )

    if pop.ndim != 2:
        raise ValueError(
            "population_in_pixels pitää olla 2D-array."
        )

    if relief.shape != pop.shape:
        raise ValueError(
            "relief pitää olla täsmälleen saman kokoinen "
            "kuin population_in_pixels."
        )

    if planet_radius <= 0:
        raise ValueError(
            "planet_radius pitää olla > 0."
        )

    if population_per_nation <= 0:
        raise ValueError(
            "population_per_nation pitää olla > 0."
        )

    height, width = pop.shape

    # ------------------------------------------------------------
    # 1. Maa / meri
    # ------------------------------------------------------------

    land = (
        np.isfinite(relief)
        & (relief > 0)
    )

    valid_population = (
        np.isfinite(pop)
        & (pop > 0)
        & land
    )

    total_population = pop[valid_population].sum()

    n_nations = int(
        total_population / population_per_nation
    )

    if n_nations < 1:
        return (
            [],
            np.zeros_like(pop, dtype=np.int32),
            np.array([], dtype=np.float64),
        )

    # ------------------------------------------------------------
    # 2. Pallokoordinaatit
    # ------------------------------------------------------------

    dlat = 180.0 / height
    dlon = 360.0 / width

    lat = (
        90.0
        - (np.arange(height) + 0.5) * dlat
    )

    lon = (
        -180.0
        + (np.arange(width) + 0.5) * dlon
    )

    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)

    lat_grid, lon_grid = np.meshgrid(
        lat_rad,
        lon_rad,
        indexing="ij",
    )

    cos_lat = np.cos(lat_grid)

    x = cos_lat * np.cos(lon_grid)
    y = cos_lat * np.sin(lon_grid)
    z = np.sin(lat_grid)

    flat_valid = np.flatnonzero(
        valid_population.ravel()
    )

    xyz = np.column_stack((
        x.ravel()[flat_valid],
        y.ravel()[flat_valid],
        z.ravel()[flat_valid],
    ))

    values = pop.ravel()[flat_valid]

    # ------------------------------------------------------------
    # 3. Kansakeskusten minimietäisyys
    # ------------------------------------------------------------

    min_angle = (
        min_distance_km / planet_radius
    )

    cos_min_angle = np.cos(
        min_angle
    )

    available = np.ones(
        len(values),
        dtype=bool,
    )

    centers = []
    center_indices = []

    for _ in range(n_nations):

        if not np.any(available):
            break

        candidate_values = np.where(
            available,
            values,
            -np.inf,
        )

        idx = np.argmax(
            candidate_values
        )

        center = np.unravel_index(
            flat_valid[idx],
            pop.shape,
        )

        centers.append(center)
        center_indices.append(idx)

        center_xyz = xyz[idx]

        dot = xyz @ center_xyz

        too_close = (
            dot >= cos_min_angle
        )

        available[too_close] = False

    # ------------------------------------------------------------
    # 4. Relief-kustannus
    # ------------------------------------------------------------

    # Maanpinnan korkeuden aiheuttama liikkumiskustannus.

    relief_positive = np.maximum(
        relief,
        0.0,
    )

    movement_cost = (
        1.0
        + (
            relief_positive
            / relief_scale
        ) ** relief_power
    )

    # Meri on täysin läpipääsemätön.

    movement_cost[
        ~land
    ] = np.inf

    # ------------------------------------------------------------
    # 5. Monilähde-Dijkstra
    # ------------------------------------------------------------
    #
    # Kaikki kansakeskukset aloittavat yhtä aikaa.
    #
    # Jokainen pikseli saa sen kansan numeron,
    # joka saavuttaa sen pienimmällä kustannuksella.
    #
    # ------------------------------------------------------------

    nation_map = np.zeros(
        pop.shape,
        dtype=np.int32,
    )

    distance = np.full(
        pop.shape,
        np.inf,
        dtype=np.float64,
    )

    heap = []

    # Keskukset lähteiksi.
    for nation_id, center in enumerate(
        centers,
        start=1,
    ):

        r, c = center

        distance[r, c] = 0.0

        nation_map[r, c] = nation_id

        heapq.heappush(
            heap,
            (0.0, nation_id, r, c),
        )

    # ------------------------------------------------------------
    # 6. Kasvata kansojen alueet
    # ------------------------------------------------------------

    # 8-suuntaiset naapurit.
    #
    # dx/dy painotetaan myöhemmin tarvittaessa
    # leveysasteen mukaan.
    #
    neighbors = (
        (-1, -1),
        (-1,  0),
        (-1,  1),
        ( 0, -1),
        ( 0,  1),
        ( 1, -1),
        ( 1,  0),
        ( 1,  1),
    )

    while heap:

        current_cost, nation_id, r, c = (
            heapq.heappop(heap)
        )

        if current_cost != distance[r, c]:
            continue

        # --------------------------------------------------------
        # Naapurit
        # --------------------------------------------------------

        for dr, dc in neighbors:

            nr = r + dr
            nc = c + dc

            # Pohjois/eteläsuuntaisen reunan ulkopuolelle
            # ei voi mennä.
            if nr < 0 or nr >= height:
                continue

            # ----------------------------------------------------
            # Pituusasteen wrap-around.
            #
            # Tämä tekee:
            #
            # col = -1 -> width - 1
            # col = width -> 0
            #
            # ----------------------------------------------------

            nc %= width

            if not land[nr, nc]:
                continue

            # ----------------------------------------------------
            # Relief-kustannus.
            #
            # Käytetään uuden pikselin korkeutta.
            # ----------------------------------------------------

            step_cost = movement_cost[nr, nc]

            new_cost = (
                current_cost
                + step_cost
            )

            if new_cost < distance[nr, nc]:

                distance[nr, nc] = new_cost

                nation_map[nr, nc] = nation_id

                heapq.heappush(
                    heap,
                    (
                        new_cost,
                        nation_id,
                        nr,
                        nc,
                    ),
                )

    # ------------------------------------------------------------
    # 7. Laske kansojen väkiluvut
    # ------------------------------------------------------------

    valid_nation_values = (
        nation_map[valid_population]
    )

    valid_pop_values = (
        pop[valid_population]
    )

    nation_populations = np.bincount(
        valid_nation_values,
        weights=valid_pop_values,
        minlength=len(centers) + 1,
    )[1:]

    return (
        centers,
        nation_map,
        nation_populations,
    )


def muodosta_kansat_00(
    population_in_pixels,
    planet_radius,
    population_per_nation=100_000,
    min_distance_km=500.0,
):
    """
    Muodostaa kansojen keskukset ja kansojen alueet
    koko planeetan väestörasterista.

    Parameters
    ----------
    population_in_pixels : np.ndarray
        2D-array muodossa (height, width).

        Rasterin oletetaan olevan tasavälisessä
        latitude/longitude-koordinaatistossa:

            rivi 0       = +90°
            viimeinen rivi = -90°
            sarake 0     = -180°
            viimeinen sarake < +180°

        Vasen ja oikea reuna muodostavat yhdessä jatkuvan
        planeetan pinnan.

    planet_radius : float
        Planeetan säde kilometreinä.

    population_per_nation : float
        Kuinka monta ihmistä vastaa yhtä kansaa.

        Esimerkiksi 100_000.

    min_distance_km : float
        Kansakeskusten pienin sallittu etäisyys planeetan
        pintaa pitkin kilometreinä.

    Returns
    -------
    centers : list[tuple[int, int]]
        Kansojen keskusten rasterikoordinaatit:

            [(row, col), ...]

        Listan ensimmäinen on kansa 1 jne.

    nation_map : np.ndarray
        Saman kokoinen rasteri kuin population_in_pixels.

            0 = ei kuulu kansaan
            1 = kansa 1
            2 = kansa 2
            ...

    nation_populations : np.ndarray
        Kansojen lopulliset väkiluvut.

            nation_populations[0] = kansa 1
            nation_populations[1] = kansa 2
            ...
    """

    pop = np.asarray(population_in_pixels, dtype=np.float64)

    if pop.ndim != 2:
        raise ValueError(
            "population_in_pixels pitää olla 2-ulotteinen."
        )

    if planet_radius <= 0:
        raise ValueError(
            "planet_radius pitää olla positiivinen."
        )

    if population_per_nation <= 0:
        raise ValueError(
            "population_per_nation pitää olla positiivinen."
        )

    if min_distance_km < 0:
        raise ValueError(
            "min_distance_km ei voi olla negatiivinen."
        )

    height, width = pop.shape

    # ------------------------------------------------------------
    # 1. Käyttökelpoiset pikselit
    # ------------------------------------------------------------

    valid = (
        np.isfinite(pop)
        & (pop > 0)
    )

    if not np.any(valid):
        return (
            [],
            np.zeros_like(pop, dtype=np.int32),
            np.array([], dtype=np.float64),
        )

    total_population = pop[valid].sum()

    # Kansojen lukumäärä.
    n_nations = int(
        total_population / population_per_nation
    )

    if n_nations < 1:
        return (
            [],
            np.zeros_like(pop, dtype=np.int32),
            np.array([], dtype=np.float64),
        )

    # ------------------------------------------------------------
    # 2. Rasteripikselien latitude/longitude
    # ------------------------------------------------------------

    # Pikselin keskipisteet.
    #
    # Tärkeää:
    # käytetään pikselikeskuksia, ei rasterin reunoja.

    dlat = 180.0 / height
    dlon = 360.0 / width

    lat = (
        90.0
        - (np.arange(height) + 0.5) * dlat
    )

    lon = (
        -180.0
        + (np.arange(width) + 0.5) * dlon
    )

    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)

    lat_grid, lon_grid = np.meshgrid(
        lat_rad,
        lon_rad,
        indexing="ij",
    )

    # ------------------------------------------------------------
    # 3. Pallopinnan 3D-koordinaatit
    # ------------------------------------------------------------
    #
    # Planeetan säde voidaan jättää pois tästä vaiheesta:
    # yksikköpallon koordinaatit riittävät kulmaetäisyyksien
    # laskemiseen.
    #

    cos_lat = np.cos(lat_grid)

    x = cos_lat * np.cos(lon_grid)
    y = cos_lat * np.sin(lon_grid)
    z = np.sin(lat_grid)

    # ------------------------------------------------------------
    # 4. Kerätään käyttökelpoiset pikselit
    # ------------------------------------------------------------

    flat_valid = np.flatnonzero(valid.ravel())

    xyz = np.column_stack((
        x.ravel()[flat_valid],
        y.ravel()[flat_valid],
        z.ravel()[flat_valid],
    ))

    values = pop.ravel()[flat_valid]

    # ------------------------------------------------------------
    # 5. Keskusten minimietäisyys
    # ------------------------------------------------------------

    # Pallopinnan etäisyys:
    #
    # distance = R * angle
    #
    # joten:
    #
    # angle = distance / R

    min_angle = (
        min_distance_km / planet_radius
    )

    # Jos halutaan minimietäisyys d,
    # pisteiden pistetulon pitää olla alle:
    #
    # cos(d / R)
    #
    cos_min_angle = np.cos(min_angle)

    # ------------------------------------------------------------
    # 6. Etsi kansakeskukset
    # ------------------------------------------------------------

    available = np.ones(
        len(values),
        dtype=bool,
    )

    centers = []
    center_indices = []

    for _ in range(n_nations):

        if not np.any(available):
            break

        # Suurin jäljellä oleva väestö.
        candidate_values = np.where(
            available,
            values,
            -np.inf,
        )

        idx = np.argmax(candidate_values)

        centers.append(
            tuple(
                np.unravel_index(
                    flat_valid[idx],
                    pop.shape,
                )
            )
        )

        center_indices.append(idx)

        # --------------------------------------------------------
        # Estetään liian lähellä olevat seuraavat keskukset.
        # --------------------------------------------------------

        center_xyz = xyz[idx]

        dot = xyz @ center_xyz

        too_close = (
            dot >= cos_min_angle
        )

        available[too_close] = False

    # ------------------------------------------------------------
    # 7. Liitä jokainen pikseli lähimpään kansakeskukseen
    # ------------------------------------------------------------

    center_xyz = xyz[center_indices]

    nation_ids = np.zeros(
        len(values),
        dtype=np.int32,
    )

    # Ei muodosteta koko
    #
    # pixel_count × nation_count
    #
    # matriisia kerralla.

    block_size = 100_000

    for start in range(
        0,
        len(values),
        block_size,
    ):

        end = min(
            start + block_size,
            len(values),
        )

        block = xyz[start:end]

        # Suurin pistetulo = pienin kulmaetäisyys.

        similarity = (
            block @ center_xyz.T
        )

        nearest = np.argmax(
            similarity,
            axis=1,
        )

        nation_ids[start:end] = (
            nearest + 1
        )

    # ------------------------------------------------------------
    # 8. Muodostetaan kansarasteri
    # ------------------------------------------------------------

    nation_map = np.zeros(
        pop.shape,
        dtype=np.int32,
    )

    nation_map.ravel()[flat_valid] = nation_ids

    # ------------------------------------------------------------
    # 9. Lasketaan kansojen väkiluvut
    # ------------------------------------------------------------

    nation_populations = np.bincount(
        nation_ids,
        weights=values,
        minlength=len(centers) + 1,
    )[1:]

    return (
        centers,
        nation_map,
        nation_populations,
    )




"""
Monthly Forest Fire Risk Index
--------------------------------

Prototype for gridded monthly climate data.

Inputs:
    temp  : temperature [12, H, W], deg C
    prec  : precipitation [12, H, W], mm/month
    wind  : wind speed [12, H, W], m/s
    npp   : NPP [12, H, W] or [H, W]

Output:
    fire_risk : [12, H, W], approximately 0-1

This is NOT the official Canadian FWI.
It is a simple, transparent prototype intended
for development and validation.
"""


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

def percentile_normalize(x, low=5, high=95):
    """
    Normalize raster to 0-1 using percentiles.

    This is more robust than min-max normalization because
    extreme raster values don't dominate the result.
    """
    lo = np.nanpercentile(x, low)
    hi = np.nanpercentile(x, high)

    result = (x - lo) / (hi - lo + 1e-9)

    return np.clip(result, 0, 1)


def monthly_percentile_normalize(x):
    """
    Normalize each month separately.

    x shape:
        [12, H, W]
    """
    result = np.zeros_like(x, dtype=float)

    for m in range(12):
        result[m] = percentile_normalize(x[m])

    return result


# ---------------------------------------------------------
# Climate components
# ---------------------------------------------------------

def calculate_heat_score(temp):
    """
    High temperature -> higher fire-weather stress.

    Rough prototype:
        <= 10 C  -> 0
        >= 30 C  -> 1
    """

    return np.clip((temp - 10.0) / 20.0, 0, 1)


def calculate_rain_dryness(prec):
    """
    Convert monthly precipitation into dryness.

    This alone is deliberately simple.
    The important part is that we later add memory
    from previous months.
    """

    # ~100 mm/month treated as very wet
    wetness = np.clip(prec / 100.0, 0, 1)

    return 1.0 - wetness


def calculate_drought_memory(prec, memory=0.65):
    """
    Create a simple multi-month drought memory.

    Previous dryness is retained with the 'memory' parameter.

    memory = 0.0  -> only current month matters
    memory = 0.9  -> strong influence from previous months
    """

    n_months = prec.shape[0]

    drought = np.zeros_like(prec, dtype=float)

    current_dryness = calculate_rain_dryness(prec)

    drought[0] = current_dryness[0]

    for m in range(1, n_months):
        drought[m] = (
            memory * drought[m - 1]
            + (1.0 - memory) * current_dryness[m]
        )

    return np.clip(drought, 0, 1)


def calculate_wind_score(wind):
    """
    Wind contribution.

    Rough prototype:
        <= 2 m/s  -> 0
        >= 15 m/s -> 1

    Change these limits depending on your dataset.
    """

    return np.clip((wind - 2.0) / 13.0, 0, 1)


# ---------------------------------------------------------
# NPP / fuel component
# ---------------------------------------------------------

def calculate_fuel_score(npp):
    """
    Convert NPP into relative fuel availability.

    Supports:

        npp.shape == [12, H, W]
        npp.shape == [H, W]

    For monthly NPP, normalization is performed separately
    for each month.
    """

    if npp.ndim == 3:
        return monthly_percentile_normalize(npp)

    elif npp.ndim == 2:
        return percentile_normalize(npp)

    else:
        raise ValueError("NPP must have shape [H,W] or [12,H,W]")


# ---------------------------------------------------------
# Main model
# ---------------------------------------------------------

def calculate_fire_risk(
    temp,
    prec,
    wind,
    npp,
    w_drought=0.40,
    w_heat=0.25,
    w_wind=0.20,
    w_fuel=0.15,
    drought_memory=0.65,
):
    """
    Calculate monthly fire risk.

    Parameters
    ----------
    temp : ndarray
        [12, H, W], deg C

    prec : ndarray
        [12, H, W], mm/month

    wind : ndarray
        [12, H, W], m/s

    npp : ndarray
        [12, H, W] or [H, W]

    Returns
    -------
    fire_risk : ndarray
        [12, H, W], 0-1

    components : dict
        Individual model components
    """

    if temp.shape != prec.shape or temp.shape != wind.shape:
        raise ValueError(
            "temp, prec and wind must have identical shapes"
        )

    if temp.ndim != 3 or temp.shape[0] != 12:
        raise ValueError(
            "Climate rasters must have shape [12, H, W]"
        )

    # -----------------------------------------------------
    # Components
    # -----------------------------------------------------

    heat = calculate_heat_score(temp)

    drought = calculate_drought_memory(
        prec,
        memory=drought_memory
    )

    wind_score = calculate_wind_score(wind)
    fuel=np.copy(npp)*0+1
    #fuel = calculate_fuel_score(npp)

    # If NPP is static [H,W], expand it to all months
    if fuel.ndim == 2:
        fuel = np.broadcast_to(
            fuel,
            temp.shape
        )

    # -----------------------------------------------------
    # Weighted risk
    # -----------------------------------------------------

    total_weight = (
        w_drought
        + w_heat
        + w_wind
        + w_fuel
    )

    fire_risk = (
        w_drought * drought
        + w_heat * heat
        + w_wind * wind_score
        + w_fuel * fuel
    ) / total_weight

    fire_risk = np.clip(fire_risk, 0, 1)

    components = {
        "drought": drought,
        "heat": heat,
        "wind": wind_score,
        "fuel": fuel,
    }

    return fire_risk, components


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




def calculate_hydrology_00(
    relief,
    precip_annual,
    pet,
    cell_size,
    river_threshold=1_000_000.0,
    min_lake_inflow=500_000.0,
    min_lake_depth=1.5,
    runoff_coefficient=0.65,
    lake_strength=1.0,
):
    """
    Kevyt mutta hydrologisesti uskottava maa/joki/järvi-malli.

    DEM:
        meri = 0
        maa > 0

    Vesimäärät:
        m3 / vuosi

    Parameters
    ----------
    relief : np.ndarray
        DEM metreinä. Meri = 0.

    precip_annual : np.ndarray
        Vuotuinen sadanta mm/vuosi.

    pet : np.ndarray
        Potentiaalinen haihdunta mm/vuosi.

    cell_size : float
        Rasterisolun koko metreinä.

    river_threshold : float
        Vuosittainen virtaama m3/vuosi,
        jonka yläpuolella solu luokitellaan joeksi.

    min_lake_inflow : float
        Minimi vuosittainen valunta m3/vuosi,
        jotta painanne voi muodostaa järven.

    min_lake_depth : float
        Järven minimisyvyys metreinä.

    runoff_coefficient : float
        Kuinka suuri osa P-PET-vesiylijäämästä
        päätyy pintavalunnaksi.

        0.0 = ei pintavaluntaa
        1.0 = kaikki muuttuu pintavalunnaksi

    lake_strength : float
        Säätää järven täyttymisen tehokkuutta.

    Returns
    -------
    result : dict

        rivers
        lakes
        lake_id
        lake_depth
        lake_area
        lake_volume
        lake_water_level
        lake_spill
        flow_to
        accumulation
        runoff
        sink_id
    """

    # =========================================================
    # 0. INPUT
    # =========================================================

    relief = np.asarray(relief, dtype=np.float64)
    precip_annual = np.asarray(
        precip_annual,
        dtype=np.float64
    )
    pet = np.asarray(
        pet,
        dtype=np.float64
    )

    if not (
        relief.shape ==
        precip_annual.shape ==
        pet.shape
    ):
        raise ValueError(
            "relief, precip_annual ja pet "
            "pitää olla samanmuotoisia."
        )

    height, width = relief.shape
    size = height * width

    cell_area = float(cell_size) ** 2

    landmask = relief > 0.0

    # =========================================================
    # 1. EFFECTIVE RUNOFF
    # =========================================================
    #
    # P - PET antaa vesiylijäämän.
    #
    # Runoff coefficient estää sitä, että kaikki
    # vesiylijäämä muuttuisi välittömästi pintavalunnaksi.
    #
    # Lopullinen yksikkö:
    #
    # m3 / vuosi / solu
    #

    water_surplus = np.maximum(
        precip_annual - pet,
        0.0
    )

    runoff_mm = (
        water_surplus *
        runoff_coefficient
    )

    runoff = (
        runoff_mm / 1000.0
    ) * cell_area

    runoff[~landmask] = 0.0

    # =========================================================
    # 2. D8 FLOW DIRECTION
    # =========================================================

    flow_to = np.full(
        (height, width, 2),
        -1,
        dtype=np.int32
    )

    neighbors = [
        (-1, -1),
        (-1,  0),
        (-1,  1),
        ( 0, -1),
        ( 0,  1),
        ( 1, -1),
        ( 1,  0),
        ( 1,  1),
    ]

    # ---------------------------------------------------------
    # D8
    # ---------------------------------------------------------

    for y in range(height):

        for x in range(width):

            if not landmask[y, x]:
                continue

            elevation = relief[y, x]

            best_slope = 0.0
            best_neighbor = None

            for dy, dx in neighbors:

                ny = y + dy
                nx = x + dx

                if not (
                    0 <= ny < height and
                    0 <= nx < width
                ):
                    continue

                distance = (
                    cell_size *
                    np.sqrt(2.0)
                    if dy != 0 and dx != 0
                    else cell_size
                )

                dz = (
                    elevation -
                    relief[ny, nx]
                )

                slope = dz / distance

                if slope > best_slope:

                    best_slope = slope
                    best_neighbor = (
                        ny,
                        nx
                    )

            if best_neighbor is not None:

                flow_to[y, x] = best_neighbor

    # =========================================================
    # 3. TOPOLOGICAL FLOW ORDER
    # =========================================================
    #
    # Korkeimmat ensin.
    #
    # Tämän ansiosta vesi voidaan siirtää alaspäin
    # yhdellä läpikäynnillä.
    #

    order = np.argsort(
        relief.ravel()
    )[::-1]

    # =========================================================
    # 4. BASIN / SINK IDENTIFICATION
    # =========================================================

    sink_id = np.full(
        size,
        -1,
        dtype=np.int32
    )

    sink_count = 0

    for index in order:

        y = index // width
        x = index % width

        if not landmask[y, x]:
            continue

        ny, nx = flow_to[y, x]

        # Ei alempaa naapuria.
        #
        # Tämä on topografinen sinkki.
        #

        if ny < 0:

            sink_id[index] = sink_count
            sink_count += 1

        else:

            downstream = (
                ny * width +
                nx
            )

            sink_id[index] = sink_id[
                downstream
            ]

    sink_id_2d = sink_id.reshape(
        height,
        width
    )

    # =========================================================
    # 5. BASIN WATER
    # =========================================================

    valid = sink_id >= 0

    basin_water = np.bincount(
        sink_id[valid],
        weights=runoff.ravel()[valid],
        minlength=sink_count
    )

    basin_area_cells = np.bincount(
        sink_id[valid],
        minlength=sink_count
    )

    basin_area = (
        basin_area_cells *
        cell_area
    )

    # =========================================================
    # 6. FIND SPILL ELEVATION
    # =========================================================
    #
    # Jokaiselle altaalle etsitään alin kohta,
    # jonka kautta vesi voisi poistua.
    #

    spill_height = np.full(
        sink_count,
        np.inf,
        dtype=np.float64
    )

    spill_y = np.full(
        sink_count,
        -1,
        dtype=np.int32
    )

    spill_x = np.full(
        sink_count,
        -1,
        dtype=np.int32
    )

    for dy, dx in neighbors:

        y0 = max(0, -dy)
        y1 = min(height, height - dy)

        x0 = max(0, -dx)
        x1 = min(width, width - dx)

        src = sink_id_2d[
            y0:y1,
            x0:x1
        ]

        nbr_sink = sink_id_2d[
            y0 + dy:y1 + dy,
            x0 + dx:x1 + dx
        ]

        nbr_height = relief[
            y0 + dy:y1 + dy,
            x0 + dx:x1 + dx
        ]

        boundary = (
            (src >= 0) &
            (nbr_sink != src)
        )

        if not np.any(boundary):
            continue

        ids = src[boundary]
        heights = nbr_height[boundary]

        # Pienin poistumiskorkeus.
        #

        np.minimum.at(
            spill_height,
            ids,
            heights
        )

    # =========================================================
    # 7. SINK POSITIONS
    # =========================================================

    sink_y = np.full(
        sink_count,
        -1,
        dtype=np.int32
    )

    sink_x = np.full(
        sink_count,
        -1,
        dtype=np.int32
    )

    for index in np.flatnonzero(valid):

        sid = sink_id[index]

        if sink_y[sid] < 0:

            sink_y[sid] = (
                index // width
            )

            sink_x[sid] = (
                index % width
            )

    # =========================================================
    # 8. LAKE CANDIDATES
    # =========================================================

    lake_candidate = (
        (basin_water >= min_lake_inflow) &
        np.isfinite(spill_height)
    )

    # =========================================================
    # 9. ESTIMATE LAKE WATER LEVEL
    # =========================================================
    #
    # Tässä käytetään altaan topografiaa.
    #
    # Mitä enemmän vettä, sitä korkeammalle vedenpinta
    # voi nousta.
    #
    # Tämä on paljon realistisempi kuin:
    #
    #     volume / basin_area
    #
    # koska järven pinta-ala määräytyy topografiasta.
    #

    lake_water_level = np.zeros(
        sink_count,
        dtype=np.float64
    )

    lake_volume = np.zeros(
        sink_count,
        dtype=np.float64
    )

    lake_area = np.zeros(
        sink_count,
        dtype=np.float64
    )

    lake_depth_sink = np.zeros(
        sink_count,
        dtype=np.float64
    )

    # ---------------------------------------------------------
    # Tämä osa tehdään vain järvikandidaateille.
    #
    # Se on tarkoituksella hitaampi kuin varsinainen
    # D8-laskenta, mutta järvikandidaatteja on yleensä
    # huomattavasti vähemmän kuin rasterisoluja.
    # ---------------------------------------------------------

    candidate_ids = np.flatnonzero(
        lake_candidate
    )

    for sid in candidate_ids:

        sy = sink_y[sid]
        sx = sink_x[sid]

        if sy < 0:
            continue

        basin_mask = (
            sink_id_2d == sid
        )

        elevations = relief[
            basin_mask
        ]

        if elevations.size == 0:
            continue

        # -----------------------------------------------------
        # Sortataan altaan korkeudet.
        #
        # Näin voidaan arvioida kuinka suuri tilavuus
        # tarvitaan tietyn vedenkorkeuden saavuttamiseen.
        # -----------------------------------------------------

        elevations = np.sort(
            elevations
        )

        # Järven pohjan korkeus.
        bottom = elevations[0]

        spill = spill_height[sid]

        # Veden määrä, jota altaan täyttäminen
        # spill-korkeuteen vaatisi.
        #

        below_spill = (
            elevations < spill
        )

        if not np.any(below_spill):
            continue

        low = elevations[
            below_spill
        ]

        # -----------------------------------------------------
        # Arvioidaan vedenkorkeus.
        #
        # Vesi täyttää ensin alimmat solut.
        #
        # Tämä on "hypsometric fill".
        # -----------------------------------------------------

        available_volume = (
            basin_water[sid] *
            lake_strength
        )

        if available_volume <= 0:
            continue

        # Vedenkorkeus aloitetaan pohjasta.

        level = bottom

        volume_used = 0.0

        # Käydään korkeustasot läpi.
        #
        # Rasteri on tyypillisesti suhteellisen pieni
        # verrattuna koko maailman dataan, ja tämä tehdään
        # vain järvikandidaateille.
        #

        previous_level = low[0]

        for elevation in low:

            if elevation <= previous_level:
                continue

            # Solujen määrä joiden pohja on tällä tasolla
            # tai sen alapuolella.

            active_cells = np.searchsorted(
                elevations,
                previous_level,
                side="right"
            )

            if active_cells <= 0:
                active_cells = 1

            dz = (
                elevation -
                previous_level
            )

            required = (
                active_cells *
                cell_area *
                dz
            )

            if (
                volume_used +
                required
                >= available_volume
            ):

                remaining = (
                    available_volume -
                    volume_used
                )

                level = (
                    previous_level +
                    remaining /
                    (
                        active_cells *
                        cell_area
                    )
                )

                break

            volume_used += required

            previous_level = elevation
            level = elevation

        # -----------------------------------------------------
        # Vedenpinta ei saa ylittää spill-korkeutta.
        # -----------------------------------------------------

        level = min(
            level,
            spill
        )

        depth = (
            level -
            bottom
        )

        if depth < min_lake_depth:
            continue

        # -----------------------------------------------------
        # Todellinen järvipinta.
        # -----------------------------------------------------

        lake_mask = (
            basin_mask &
            (relief < level)
        )

        area_cells = np.count_nonzero(
            lake_mask
        )

        area = (
            area_cells *
            cell_area
        )

        # -----------------------------------------------------
        # Tilavuus vedenpinnan alla.
        # -----------------------------------------------------

        volume = np.sum(
            np.maximum(
                level -
                relief[lake_mask],
                0.0
            )
        ) * cell_area

        lake_water_level[sid] = level
        lake_depth_sink[sid] = depth
        lake_area[sid] = area
        lake_volume[sid] = volume

    # =========================================================
    # 10. LAKE MASK
    # =========================================================

    lakes = np.zeros(
        (height, width),
        dtype=bool
    )

    lake_depth = np.zeros(
        (height, width),
        dtype=np.float64
    )

    lake_id = np.zeros(
        (height, width),
        dtype=np.int32
    )

    # ---------------------------------------------------------
    # Muodostetaan järvet.
    # ---------------------------------------------------------

    for sid in np.flatnonzero(
        lake_water_level > 0
    ):

        level = lake_water_level[sid]

        mask = (
            (sink_id_2d == sid) &
            (relief < level) &
            landmask
        )

        if not np.any(mask):
            continue

        lakes[mask] = True

        lake_id[mask] = (
            sid + 1
        )

        lake_depth[mask] = (
            level -
            relief[mask]
        )

    # =========================================================
    # 11. FINAL WATER ACCUMULATION
    # =========================================================
    #
    # Nyt tehdään varsinainen jokiverkko.
    #
    # Järvet toimivat varastoina:
    #
    #     yläpuolinen runoff
    #            ↓
    #          järvi
    #            ↓
    #        spill outlet
    #            ↓
    #          joki
    #

    accumulation = runoff.copy()

    # ---------------------------------------------------------
    # Lisää järven ulosvirtaus sen spill-pisteeseen.
    #
    # Yksinkertainen vuositasoinen vesitase:
    #
    #     lake input
    #     + lake precipitation
    #     - lake evaporation
    #
    # Jos järvi on spillannut, ylimääräinen vesi jatkaa
    # alavirtaan.
    # ---------------------------------------------------------

    lake_outflow = np.zeros(
        sink_count,
        dtype=np.float64
    )

    for sid in np.flatnonzero(
        lake_water_level > 0
    ):

        level = lake_water_level[sid]

        # Altaan valuma-alueen kokonaisvesimäärä.
        inflow = basin_water[sid]

        # -----------------------------------------------------
        # Järven oma sadanta.
        # -----------------------------------------------------

        mask = (
            lake_id == sid + 1
        )

        if not np.any(mask):
            continue

        lake_precip = np.sum(
            precip_annual[mask] /
            1000.0 *
            cell_area
        )

        # -----------------------------------------------------
        # Järven haihdunta.
        # -----------------------------------------------------

        lake_evap = np.sum(
            pet[mask] /
            1000.0 *
            cell_area
        )

        net_lake_water = (
            inflow +
            lake_precip -
            lake_evap
        )

        # -----------------------------------------------------
        # Jos järvi on saavuttanut spill-korkeuden,
        # vesi poistuu alavirtaan.
        #
        # Muuten oletetaan, että vesi jää järveen.
        # -----------------------------------------------------

        if level >= spill_height[sid] - 1e-9:

            lake_outflow[sid] = max(
                net_lake_water,
                0.0
            )

    # =========================================================
    # 12. REMOVE BASIN INTERNAL FLOW
    # =========================================================
    #
    # Järven sisällä oleva valuma ei saa tulla jokiverkostoon
    # normaalina D8-virtauksena.
    #
    # Järvi korvaa sen yhtenä hydrologisena solmuna.
    #

    accumulation[lakes] = 0.0

    # =========================================================
    # 13. ACCUMULATE LAND FLOW
    # =========================================================

    for index in order:

        y = index // width
        x = index % width

        if not landmask[y, x]:
            continue

        # Järven solu ei välitä normaalia D8-virtausta.
        #

        if lakes[y, x]:
            continue

        ny, nx = flow_to[y, x]

        if ny >= 0:

            accumulation[ny, nx] += (
                accumulation[y, x]
            )

    # =========================================================
    # 14. ADD LAKE OUTFLOWS
    # =========================================================

    for sid in np.flatnonzero(
        lake_outflow > 0
    ):

        sy = sink_y[sid]
        sx = sink_x[sid]

        level = lake_water_level[sid]

        # Etsitään järven reunan alin solu,
        # joka on järven ulkopuolella.
        #

        best = None
        best_height = np.inf

        mask = (
            lake_id ==
            sid + 1
        )

        ys, xs = np.where(mask)

        for y, x in zip(ys, xs):

            for dy, dx in neighbors:

                ny = y + dy
                nx = x + dx

                if not (
                    0 <= ny < height and
                    0 <= nx < width
                ):
                    continue

                if mask[ny, nx]:
                    continue

                h = relief[ny, nx]

                if h < best_height:

                    best_height = h
                    best = (
                        ny,
                        nx
                    )

        if best is not None:

            oy, ox = best

            accumulation[oy, ox] += (
                lake_outflow[sid]
            )

    # =========================================================
    # 15. RIVERS
    # =========================================================

    rivers = (
        landmask &
        (~lakes) &
        (accumulation >= river_threshold)
    )

    # =========================================================
    # 16. RETURN
    # =========================================================

    return {
        "rivers": rivers,

        "lakes": lakes,

        "lake_id": lake_id,

        "lake_depth": lake_depth,

        "lake_area": lake_area,

        "lake_volume": lake_volume,

        "lake_water_level": lake_water_level,

        "lake_spill": spill_height,

        "lake_outflow": lake_outflow,

        "flow_to": flow_to,

        "accumulation": accumulation,

        "runoff": runoff,

        "sink_id": sink_id_2d,
    }




import numpy as np


def calculate_hydrology(
    relief,
    precip_annual,
    pet,
    cell_size,
    river_threshold=1_000_000.0,
    min_lake_inflow=500_000.0,
    min_lake_depth=1.5,
    runoff_coefficient=0.65,
    lake_strength=1.0,
):
    """
    Kevyt mutta hydrologisesti uskottava joki- ja järvimalli.

    Kaikki vesimäärät ovat m3/vuosi.

    DEM:
        meri = 0
        maa > 0

    Palauttaa
    ---------
    result : dict
        rivers
        lakes
        lake_id
        lake_depth
        lake_area
        lake_volume
        lake_water_level
        lake_spill
        lake_outflow
        flow_to
        accumulation
        runoff
        sink_id
        lake_outlet
    """

    # =========================================================
    # 0. INPUT
    # =========================================================

    relief = np.asarray(relief, dtype=np.float64)
    precip_annual = np.asarray(
        precip_annual,
        dtype=np.float64
    )
    pet = np.asarray(
        pet,
        dtype=np.float64
    )

    if not (
        relief.shape ==
        precip_annual.shape ==
        pet.shape
    ):
        raise ValueError(
            "relief, precip_annual ja pet "
            "pitää olla samanmuotoisia."
        )

    height, width = relief.shape
    size = height * width

    cell_area = float(cell_size) ** 2

    landmask = relief > 0.0

    # =========================================================
    # 1. EFFECTIVE RUNOFF
    # =========================================================
    #
    # P - PET = ilmastollinen vesiylijäämä.
    #
    # Runoff coefficient kuvaa infiltraatiota,
    # maaperää jne. ilman raskasta maaperämallia.
    #

    water_surplus = np.maximum(
        precip_annual - pet,
        0.0
    )

    runoff_mm = (
        water_surplus *
        runoff_coefficient
    )

    runoff = (
        runoff_mm / 1000.0
    ) * cell_area

    runoff[~landmask] = 0.0

    # =========================================================
    # 2. D8 FLOW
    # =========================================================

    flow_to = np.full(
        (height, width, 2),
        -1,
        dtype=np.int32
    )

    neighbors = [
        (-1, -1),
        (-1,  0),
        (-1,  1),
        ( 0, -1),
        ( 0,  1),
        ( 1, -1),
        ( 1,  0),
        ( 1,  1),
    ]

    diagonal = np.sqrt(2.0)

    for y in range(height):

        for x in range(width):

            if not landmask[y, x]:
                continue

            h = relief[y, x]

            best_slope = 0.0
            best = None

            for dy, dx in neighbors:

                ny = y + dy
                nx = x + dx

                if not (
                    0 <= ny < height and
                    0 <= nx < width
                ):
                    continue

                distance = (
                    cell_size * diagonal
                    if dy != 0 and dx != 0
                    else cell_size
                )

                slope = (
                    h - relief[ny, nx]
                ) / distance

                if slope > best_slope:

                    best_slope = slope
                    best = (ny, nx)

            if best is not None:
                flow_to[y, x] = best

    # =========================================================
    # 3. TOPOLOGICAL ORDER
    # =========================================================
    #
    # Korkeimmat ensin.
    #

    order = np.argsort(
        relief.ravel()
    )[::-1]

    # =========================================================
    # 4. FIND TOPOGRAPHIC SINKS
    # =========================================================

    sink_id = np.full(
        size,
        -1,
        dtype=np.int32
    )

    sink_count = 0

    for index in order:

        y = index // width
        x = index % width

        if not landmask[y, x]:
            continue

        ny, nx = flow_to[y, x]

        if ny < 0:

            sink_id[index] = sink_count
            sink_count += 1

        else:

            downstream = (
                ny * width +
                nx
            )

            sink_id[index] = (
                sink_id[downstream]
            )

    sink_id_2d = sink_id.reshape(
        height,
        width
    )

    valid = sink_id >= 0

    # =========================================================
    # 5. BASIN WATER
    # =========================================================

    basin_water = np.bincount(
        sink_id[valid],
        weights=runoff.ravel()[valid],
        minlength=sink_count
    )

    basin_area_cells = np.bincount(
        sink_id[valid],
        minlength=sink_count
    )

    basin_area = (
        basin_area_cells *
        cell_area
    )

    # =========================================================
    # 6. SPILL ELEVATION
    # =========================================================

    spill_height = np.full(
        sink_count,
        np.inf,
        dtype=np.float64
    )

    # Spill-pisteen koordinaatit.
    #
    # Piste on altaan solu, jonka kautta vesi ylittää
    # matalimman altaan reunan.
    #

    spill_y = np.full(
        sink_count,
        -1,
        dtype=np.int32
    )

    spill_x = np.full(
        sink_count,
        -1,
        dtype=np.int32
    )

    for dy, dx in neighbors:

        y0 = max(0, -dy)
        y1 = min(height, height - dy)

        x0 = max(0, -dx)
        x1 = min(width, width - dx)

        src = sink_id_2d[
            y0:y1,
            x0:x1
        ]

        nbr_sink = sink_id_2d[
            y0 + dy:y1 + dy,
            x0 + dx:x1 + dx
        ]

        nbr_height = relief[
            y0 + dy:y1 + dy,
            x0 + dx:x1 + dx
        ]

        boundary = (
            (src >= 0) &
            (nbr_sink != src)
        )

        if not np.any(boundary):
            continue

        ids = src[boundary]
        heights = nbr_height[boundary]

        # Minimi spill-korkeus.
        np.minimum.at(
            spill_height,
            ids,
            heights
        )

    # =========================================================
    # 7. FIND EXACT SPILL CELLS
    # =========================================================
    #
    # Etsitään jokaiselle altaalle solu, jonka naapurina
    # on spill_height.
    #

    for dy, dx in neighbors:

        y0 = max(0, -dy)
        y1 = min(height, height - dy)

        x0 = max(0, -dx)
        x1 = min(width, width - dx)

        src = sink_id_2d[
            y0:y1,
            x0:x1
        ]

        nbr_sink = sink_id_2d[
            y0 + dy:y1 + dy,
            x0 + dx:x1 + dx
        ]

        nbr_height = relief[
            y0 + dy:y1 + dy,
            x0 + dx:x1 + dx
        ]

        boundary = (
            (src >= 0) &
            (nbr_sink != src) &
            np.isfinite(nbr_height)
        )

        if not np.any(boundary):
            continue

        ys, xs = np.where(boundary)

        for iy, ix in zip(ys, xs):

            sid = src[iy, ix]
            h = nbr_height[iy, ix]

            if (
                h <=
                spill_height[sid] + 1e-9
            ):

                spill_y[sid] = iy + y0
                spill_x[sid] = ix + x0

    # =========================================================
    # 8. SINK POSITIONS
    # =========================================================

    sink_y = np.full(
        sink_count,
        -1,
        dtype=np.int32
    )

    sink_x = np.full(
        sink_count,
        -1,
        dtype=np.int32
    )

    for index in np.flatnonzero(valid):

        sid = sink_id[index]

        if sink_y[sid] < 0:

            sink_y[sid] = (
                index // width
            )

            sink_x[sid] = (
                index % width
            )

    # =========================================================
    # 9. LAKE ARRAYS
    # =========================================================

    lake_water_level = np.zeros(
        sink_count,
        dtype=np.float64
    )

    lake_depth_sink = np.zeros(
        sink_count,
        dtype=np.float64
    )

    lake_area = np.zeros(
        sink_count,
        dtype=np.float64
    )

    lake_volume = np.zeros(
        sink_count,
        dtype=np.float64
    )

    lake_outflow = np.zeros(
        sink_count,
        dtype=np.float64
    )

    # =========================================================
    # 10. LAKE CANDIDATES
    # =========================================================

    lake_candidates = (
        (basin_water >= min_lake_inflow) &
        np.isfinite(spill_height)
    )

    # =========================================================
    # 11. BUILD LAKES
    # =========================================================
    #
    # Järven vedenpinta määräytyy topografiasta.
    #
    # Ei:
    #
    #     basin_water / basin_area
    #
    # vaan etsitään korkeus, jonka alle käytettävissä oleva
    # vesimäärä mahtuu.
    #

    candidate_ids = np.flatnonzero(
        lake_candidates
    )

    for sid in candidate_ids:

        sy = sink_y[sid]
        sx = sink_x[sid]

        if sy < 0:
            continue

        basin_mask = (
            sink_id_2d == sid
        )

        elevations = relief[
            basin_mask
        ]

        if elevations.size == 0:
            continue

        # Matalimmasta korkeimmalle.
        elevations = np.sort(
            elevations
        )

        spill = spill_height[sid]

        below = (
            elevations < spill
        )

        if not np.any(below):
            continue

        elevations = elevations[below]

        bottom = elevations[0]

        available_volume = (
            basin_water[sid] *
            lake_strength
        )

        if available_volume <= 0:
            continue

        # -----------------------------------------------------
        # Hypsometrinen täyttö
        # -----------------------------------------------------

        level = bottom
        previous = bottom
        volume_used = 0.0

        n = elevations.size

        # Solumäärät korkeustasoittain.
        unique_h, counts = np.unique(
            elevations,
            return_counts=True
        )

        active_cells = 0

        for h, count in zip(
            unique_h,
            counts
        ):

            if h > previous:

                dz = (
                    h -
                    previous
                )

                required = (
                    active_cells *
                    cell_area *
                    dz
                )

                if (
                    volume_used +
                    required
                    >= available_volume
                ):

                    remaining = (
                        available_volume -
                        volume_used
                    )

                    if active_cells > 0:

                        level = (
                            previous +
                            remaining /
                            (
                                active_cells *
                                cell_area
                            )
                        )

                    break

                volume_used += required

                level = h

            active_cells += count
            previous = h

        level = min(
            level,
            spill
        )

        depth = (
            level -
            bottom
        )

        if depth < min_lake_depth:
            continue

        # -----------------------------------------------------
        # Todellinen järvimaski
        # -----------------------------------------------------

        lake_mask = (
            basin_mask &
            (relief < level) &
            landmask
        )

        area_cells = np.count_nonzero(
            lake_mask
        )

        if area_cells == 0:
            continue

        area = (
            area_cells *
            cell_area
        )

        volume = np.sum(
            np.maximum(
                level -
                relief[lake_mask],
                0.0
            )
        ) * cell_area

        lake_water_level[sid] = level
        lake_depth_sink[sid] = depth
        lake_area[sid] = area
        lake_volume[sid] = volume

    # =========================================================
    # 12. LAKE MASK
    # =========================================================

    lakes = np.zeros(
        (height, width),
        dtype=bool
    )

    lake_depth = np.zeros(
        (height, width),
        dtype=np.float64
    )

    lake_id = np.zeros(
        (height, width),
        dtype=np.int32
    )

    lake_sids = np.flatnonzero(
        lake_water_level > 0
    )

    for sid in lake_sids:

        level = lake_water_level[sid]

        mask = (
            (sink_id_2d == sid) &
            (relief < level) &
            landmask
        )

        lakes[mask] = True

        lake_id[mask] = (
            sid + 1
        )

        lake_depth[mask] = (
            level -
            relief[mask]
        )

    # =========================================================
    # 13. LAKE WATER BALANCE
    # =========================================================
    #
    # Qout =
    #
    #     basin runoff
    #     + lake precipitation
    #     - lake evaporation
    #
    # Jos vedenpinta ei yllä spilliin, Qout = 0.
    #

    for sid in lake_sids:

        level = lake_water_level[sid]

        mask = (
            lake_id ==
            sid + 1
        )

        if not np.any(mask):
            continue

        lake_precip = np.sum(
            precip_annual[mask]
            / 1000.0
            * cell_area
        )

        lake_evap = np.sum(
            pet[mask]
            / 1000.0
            * cell_area
        )

        net_water = (
            basin_water[sid]
            + lake_precip
            - lake_evap
        )

        # Spillaa vain jos vedenpinta saavuttaa
        # altaan reunan.
        #

        if (
            level >=
            spill_height[sid] - 1e-9
        ):

            lake_outflow[sid] = max(
                net_water,
                0.0
            )

    # =========================================================
    # 14. LAKE OUTLET
    # =========================================================
    #
    # outlet = ensimmäinen solu järven ulkopuolella,
    # johon spill tapahtuu.
    #

    lake_outlet = np.full(
        (height, width, 2),
        -1,
        dtype=np.int32
    )

    for sid in lake_sids:

        sy = spill_y[sid]
        sx = spill_x[sid]

        if sy < 0:
            continue

        spill_h = spill_height[sid]

        best = None
        best_h = np.inf

        # Spill-solun ympäristö.
        for dy, dx in neighbors:

            ny = sy + dy
            nx = sx + dx

            if not (
                0 <= ny < height and
                0 <= nx < width
            ):
                continue

            # Naapurin pitää olla järven ulkopuolella.
            if lake_id[ny, nx] == sid + 1:
                continue

            h = relief[ny, nx]

            if h < best_h:

                best_h = h
                best = (
                    ny,
                    nx
                )

        if best is not None:

            oy, ox = best

            lake_outlet[
                sy,
                sx
            ] = (
                oy,
                ox
            )

    # =========================================================
    # 15. FINAL FLOW NETWORK
    # =========================================================
    #
    # Järven sisäiset D8-yhteydet poistetaan.
    #
    # Spill-solu -> lake outlet -> normaali D8.
    #

    final_flow_to = flow_to.copy()

    ys, xs = np.where(lakes)

    for y, x in zip(ys, xs):

        final_flow_to[y, x] = (
            -1,
            -1
        )

    # Spill point on järven viimeinen solu.
    # Se ohjataan outlet-soluun.
    #

    for sid in lake_sids:

        sy = spill_y[sid]
        sx = spill_x[sid]

        if sy < 0:
            continue

        oy, ox = lake_outlet[
            sy,
            sx
        ]

        if oy >= 0:

            final_flow_to[
                sy,
                sx
            ] = (
                oy,
                ox
            )

    # =========================================================
    # 16. ACCUMULATION
    # =========================================================
    #
    # Ensin tavallinen maa-valunta.
    #

    accumulation = runoff.copy()

    # Järvien sisäinen runoff ei saa kulkea
    # normaalisti läpi jokaista järvisolua.
    #

    accumulation[lakes] = 0.0

    # ---------------------------------------------------------
    # Maa -> maa
    # ---------------------------------------------------------

    for index in order:

        y = index // width
        x = index % width

        if not landmask[y, x]:
            continue

        # Järvisolut ohitetaan tässä.
        #

        if lakes[y, x]:
            continue

        ny, nx = final_flow_to[y, x]

        if ny >= 0:

            accumulation[
                ny,
                nx
            ] += accumulation[
                y,
                x
            ]

    # =========================================================
    # 17. LAKE OUTFLOW INTO NETWORK
    # =========================================================
    #
    # Järven koko vuosivirta siirtyy outlet-soluun.
    #

    for sid in lake_sids:

        q = lake_outflow[sid]

        if q <= 0:
            continue

        sy = spill_y[sid]
        sx = spill_x[sid]

        if sy < 0:
            continue

        oy, ox = lake_outlet[
            sy,
            sx
        ]

        if oy < 0:
            continue

        accumulation[
            oy,
            ox
        ] += q

    # =========================================================
    # 18. JÄRVEN OMA VEDEN MÄÄRÄ
    # =========================================================
    #
    # Järvi saa olla myös "hydrologinen solmu".
    #
    # Sen pinta-ala, tilavuus ja vedenkorkeus jäävät
    # erillisiksi tiedoiksi.
    #

    # =========================================================
    # 19. RIVERS
    # =========================================================

    rivers = (
        landmask &
        (~lakes) &
        (
            accumulation >=
            river_threshold
        )
    )

    # =========================================================
    # 20. RETURN
    # =========================================================

    return {
        "rivers": rivers,

        "lakes": lakes,

        "lake_id": lake_id,

        "lake_depth": lake_depth,

        "lake_area": lake_area,

        "lake_volume": lake_volume,

        "lake_water_level":
            lake_water_level,

        "lake_spill":
            spill_height,

        "lake_outflow":
            lake_outflow,

        "lake_outlet":
            lake_outlet,

        "flow_to":
            final_flow_to,

        "accumulation":
            accumulation,

        "runoff":
            runoff,

        "sink_id":
            sink_id_2d,
    }





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
        #print("vuoden_kulma, rata_kulma, aurinko_deklinaatio, aurinko_y, aurinkoetaisyys")
        #print(vuoden_kulma, rata_kulma, aurinko_deklinaatio, aurinko_y, aurinkoetaisyys)

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
        ) * (6.5+1.5*math.log10(planet_atmosphere_pressure))  *1.0*gee_earths
  
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
                -25.0 * # -25
                (
                    abs_y -
               
                    0.68 ## 0.60
                ) ** 2
            )
        )

        # --------------------------------------------------------
        # 3. Kylmien alueiden sade
        # --------------------------------------------------------

        kylma_sade = (
            150.0 * ## 150x
            np.exp(
                -3.0 *  ##5x
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
            0.3,
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
        
        meriefekti=np.where(meriefekti<0.5,0.5,meriefekti)
        
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
            ##kosteus *
            sade_leveysaste
            #* orografinen_efekti
            #*korkeus_efekti
            *sadekatve 
            #*meriefekti
            #+
            #1 ## 40.0
        )

        # ========================================================
        # 9. KUUKAUSISADE
        # ========================================================

        ## kk_sade[kk] = np.clip(sade_pohja /12.0 *P,0.0,None)
        #kk_sade[kk] = sade_leveysaste*kosteus*korkeus_efekti*sadekatve
        kk_sade[kk] = sade_pohja*0.5
    # ============================================================
    # PALAUTUS
    # ============================================================
    #kk_temp=kk_temp-5
    #kk_sade=kk_sade*1.25
    kk_temp=kk_temp-12
    kk_sade=kk_sade*1.25
    kk_tuuli_x=kk_tuuli_x*4 ##  about 33 in land, in sea 7.75
    kk_tuuli_y=kk_tuuli_x*4
    kk_tuuli_z=kk_tuuli_x*4
    # delta_t?
    kk_tuuli_x=math.sqrt(planet_radius_re)*kk_tuuli_x/planet_atmosphere_pressure ##  about 33 in land, in sea 7.75
    kk_tuuli_y=math.sqrt(planet_radius_re)*kk_tuuli_x/planet_atmosphere_pressure ## delta_t ?
    kk_tuuli_z=math.sqrt(planet_radius_re)*kk_tuuli_x/planet_atmosphere_pressure
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




def luo_biomikartta(dem, temp_annual, precip_annual, sealevel=0.5):
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
    T = temp_annual
    P = precip_annual
    
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

def distance_to_point(target_lon, target_lat, height, width, bounds, planet_radius=6371.0):
    """
    Laskee etäisyyden kilometreinä pallon pinnalla kohdepisteeseen koko rasterille.
    
    bounds: [min_lon, max_lon, min_lat, max_lat] -> esim. [-180, 180, -90, 90]
    planet_radius: Planeetan säde kilometreinä (Maalla n. 6371 km)
    """
    min_lon, max_lon, min_lat, max_lat = bounds
    
    # 1. Luodaan x- ja y-akselien koordinaatit rasterin koon mukaan
    # Haetaan jokaisen ruudun keskipisteen koordinaatti
    lon_akseli = np.linspace(min_lon, max_lon, width)
    lat_akseli = np.linspace(max_lat, min_lat, height) # Y-akseli alkaa yleensä ylhäältä (max_lat) alaspäin
    
    # 2. Tehdään akseleista koko rasterin kokoiset 2D-ruudukot (lon- ja lat-rasterit)
    lon_ruudukko, lat_ruudukko = np.meshgrid(lon_akseli, lat_akseli)
    
    # 3. Muutetaan kaikki asteet radiaaneiksi (Haversine-kaava vaatii radiaanit)
    lon1 = np.radians(lon_ruudukko)
    lat1 = np.radians(lat_ruudukko)
    lon2 = np.radians(target_lon)
    lat2 = np.radians(target_lat)
    
    # 4. Haversine-kaava (Pallonpinnan etäisyys)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    # np.clip varmistaa, ettei pyöristysvirheiden takia luku hyppää yli 1.0 (mikä rikkoisi arcsin-funktion)
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1))) 
    
    # 5. Kerrotaan planeetan säteellä, jotta saadaan kilometrit
    distance_raster = planet_radius * c
    
    return distance_raster




def laske_etaisyys_korkeuteen_km(relief, target_height, planet_radius, tolerance=10):
    """ distange to certain height in relief or dem
    Laskee etäisyysrasterin (kilometreinä) lähimpään pisteeseen, joka vastaa tavoitekorkeutta.
    
    Parametrit:
    - relief: 2D numpy array [height, width]
    - target_height: Tavoitekorkeus (samassa yksikössä kuin relief, esim. metreinä)
    - planet_radius: Planeetan säde kilometreinä
    - tolerance: Sallittu poikkeama tavoitekorkeudesta (oletus 10), jotta pisteitä löytyy
    """
    nrows, ncols = relief.shape
    
    # 1. Luodaan koordinaattiverkko (lasketaan solujen keskipisteet)
    # Huom: Alkuperäinen koordinaatisto kattaa pituusasteet [-180, 180] ja leveysasteet [-90, 90]
    lons = np.linspace(-180, 180, ncols)
    lats = np.linspace(-90, 90, nrows)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # Muunnetaan asteet radiaaneiksi Haversine-kaavaa varten
    lon_rad = np.radians(lon_grid)
    lat_rad = np.radians(lat_grid)
    
    # 2. Esitellään tyhjä etäisyysrasteri täynnä ääretöntä
    distance_raster = np.full_like(relief, fill_value=np.inf, dtype=np.float64)
    
    # 3. Esitellään kohdepisteet, jotka ovat riittävän lähellä tavoitekorkeutta
    mask = np.abs(relief - target_height) <= tolerance
    
    # Jos yhtään pistettä ei löydy toleranssin rajoissa, otetaan kaikkein lähin yksittäinen piste
    if not np.any(mask):
        mask = np.abs(relief - target_height) == np.min(np.abs(relief - target_height))
        
    target_lats_rad = lat_rad[mask]
    target_lons_rad = lon_rad[mask]
    
    # 4. Lasketaan jokaiselle rasterin solulle etäisyys lähimpään kohdepisteeseen
    # Vektoroidaan operaatio käymällä kohdepisteet läpi (tehokkaampi kuin solujen läpikäynti)
    for t_lat, t_lon in zip(target_lats_rad, target_lons_rad):
        # Haversine-kaava
        dlat = lat_rad - t_lat
        dlon = lon_rad - t_lon
        
        a = np.sin(dlat / 2)**2 + np.cos(lat_rad) * np.cos(t_lat) * np.sin(dlon / 2)**2
        # Varmistetaan numeerinen vakaus (a ei saa ylittää 1.0)
        a = np.clip(a, 0.0, 1.0)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        d = planet_radius * c
        
        # Päivitetään rasteriin aina lyhin löydetty etäisyys
        distance_raster = np.minimum(distance_raster, d)
        
    return distance_raster



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
        #if uusi_indeksi <= 5:  # Näytetään esim. 5 suurinta
        #    print(f"Manner {uusi_indeksi}: Pinta-ala = {manner_alat_km2[vanha_id]:,.1f} km², Osuus planeetasta = {osuus*100:.2f}%")

    return osuus_taulukko, jarjestys_taulukko



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



def laske_albedoluokat(landmask, temp_annual, precip_annual):

    # Alustetaan albedorasteri NaN-arvoilla (float64, jotta tukee NaN-arvoja)
    #albedo = np.full_like(landmask.shape, np.nan, dtype=np.float64)
    albedo = np.copy(landmask)
    albedo = np.where(albedo==0,np.nan,0)   

    # Tehdään maski vain maa-alueille, joissa on dataa
    maa_maski = np.copy(landmask) == 1

    # Haetaan helpompaa käsittelyä varten vain maa-alueiden arvot
    T = temp_annual[maa_maski]
    P = precip_annual[maa_maski]

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




def calculate_gdd_above_and_below(
    temps,
    t_base=5.0,
    orbital_period_in_days=365.25, rotation_period_in_hours=24
):
    """
    Laskee lämpösumman kynnyslämpötilan ylä- ja alapuolelta.

    Parameters
    ----------
    temps : np.ndarray
        12 kuukauden lämpötilarasteri.
        Shape: (12, rows, cols)
        Yksikkö: °C, kuukausikeskiarvo.

    t_base : float
        Kynnyslämpötila °C.

    rotation_period : float
        Planeetan vuoden pituus Maan päivinä.
        Maa = 365 päivää.

    Returns
    -------
    gdd_above : np.ndarray
        Lämpösumma yli t_base-arvon, °C·päivää.

    gdd_below : np.ndarray
        Kylmäsumma alle t_base-arvon, °C·päivää.
    """

    temps = np.asarray(temps, dtype=np.float32)

    if temps.shape[0] != 12:
        raise ValueError(
            f"temps pitää sisältää 12 kuukautta, mutta shape on {temps.shape}"
        )

    # Keskimääräinen kuukauden pituus.
    # Maalle oletuksena 30.5 päivää/kk.
    days_in_month = 30.5 * (orbital_period_in_days/ 365.0)*(rotation_period_in_hours/ 24.0)

    # Lämpötila kynnysarvon yläpuolella
    above = np.maximum(temps - t_base, 0)

    # Lämpötila kynnysarvon alapuolella
    below = np.maximum(t_base - temps, 0)

    # Kuukausikeskiarvo -> °C-päivät
    gdd_above = np.nansum(
        above * days_in_month,
        axis=0
    )

    gdd_below = np.nansum(
        below * days_in_month,
        axis=0
    )

    return gdd_above, gdd_below


from scipy.ndimage import map_coordinates





def calculate_pet(
    temperatures,
    dem,
    landmask,
    lat_min=-90.0,
    lat_max=90.0,
    lapse_rate=0.0065,

    # Planeetan pyöriminen
    rotation_period_h=24.0,

    # Planeetan akseli
    tilt=23.44,

    # Planeetan rata
    orbital_period_days=365.25,
    eccentricity=0.0167,
    mvelp=102.94,

    # Tähden säteily planeetan keskietäisyydellä
    stellar_flux=1361.0,

    # Kalenteri
    days_in_month=None,

    # Kuukausiakseli
    axis=0,
):
    """
    Laskee kuukausittaisista lämpötiloista vuosittaisen PET:n.

    Planeetan astronomiset parametrit:
        - tilt
        - rotation_period_h
        - orbital_period_days
        - eccentricity
        - mvelp

    Tähden säteily:
        stellar_flux = tähden säteilyvuo [W/m²]
        planeetan keskimääräisellä kiertorataetäisyydellä.

    Parameters
    ----------
    temperatures : np.ndarray
        Kuukausittaiset keskilämpötilat °C.

        Shape esimerkiksi:
            (12, height, width)

        Jos kuukausiakseli on muualla, käytä axis-parametria.

    dem : np.ndarray
        Korkeus metreinä.
        Shape:
            (height, width)

    landmask : np.ndarray
        1 = maa
        0 = meri

    lat_min : float
        Rasterin eteläisin leveysaste.

    lat_max : float
        Rasterin pohjoisin leveysaste.

    lapse_rate : float
        Lämpötilan pystysuuntainen gradientti °C/m.

    rotation_period_h : float
        Planeetan pyörähdysaika tunneissa.

    tilt : float
        Planeetan akselikallistuma asteina.

    orbital_period_days : float
        Planeetan kiertoaika päivinä.

    eccentricity : float
        Radankeskipakottomuus.

        e = 0:
            ympyrärata

        e > 0:
            elliptinen rata

    mvelp : float
        Perihelionin pituusaste suhteessa
        kevättasauspisteeseen.

    stellar_flux : float
        Tähden säteilyvuo W/m² planeetan
        keskietäisyydellä tähdestä.

    days_in_month : array-like or None
        Kuukausien pituudet päivinä.

    axis : int
        temperatures-arrayn kuukausiakseli.

    Returns
    -------
    np.ndarray
        Vuosittainen PET [mm/v].
    """

    # =========================================================
    # Tarkistukset
    # =========================================================

    if dem.shape != landmask.shape:
        raise ValueError(
            "dem ja landmask eivät ole saman kokoisia."
        )

    if temperatures.ndim != 3:
        raise ValueError(
            "temperatures pitää olla 3-ulotteinen."
        )

    temperatures = np.moveaxis(
        temperatures,
        axis,
        0
    )

    n_months, height, width = temperatures.shape

    if n_months != 12:
        raise ValueError(
            "Thornthwaite-laskenta odottaa 12 kuukautta."
        )

    if dem.shape != (height, width):
        raise ValueError(
            "temperatures-rasterin spatial dimensions "
            "eivät vastaa dem-rasteria."
        )

    if not 0.0 <= eccentricity < 1.0:
        raise ValueError(
            "eccentricity pitää olla välillä 0 <= e < 1."
        )

    if rotation_period_h <= 0:
        raise ValueError(
            "rotation_period_h pitää olla > 0."
        )

    if orbital_period_days <= 0:
        raise ValueError(
            "orbital_period_days pitää olla > 0."
        )

    if stellar_flux < 0:
        raise ValueError(
            "stellar_flux pitää olla >= 0."
        )

    # =========================================================
    # Kuukausien pituudet
    # =========================================================

    if days_in_month is None:

        days = np.array([
            31, 28, 31, 30, 31, 30,
            31, 31, 30, 31, 30, 31
        ], dtype=float)

    else:

        days = np.asarray(
            days_in_month,
            dtype=float
        )

        if days.shape != (12,):
            raise ValueError(
                "days_in_month pitää sisältää 12 arvoa."
            )

        if np.any(days <= 0):
            raise ValueError(
                "days_in_month sisältää virheellisiä arvoja."
            )

    # =========================================================
    # Kuukausien keskikohdat
    # =========================================================

    month_start = np.concatenate([
        [0.0],
        np.cumsum(days)[:-1]
    ])

    month_day = (
        month_start
        + days / 2.0
    )

    # =========================================================
    # Leveysaste
    # =========================================================

    latitudes = np.linspace(
        lat_max,
        lat_min,
        height
    )

    latitude = latitudes[:, np.newaxis]

    phi = np.radians(latitude)

    # =========================================================
    # Korkeussäätö
    # =========================================================

    temp = (
        temperatures
        - lapse_rate * dem[np.newaxis, :, :]
    )

    # =========================================================
    # Thornthwaite lämpöindeksi
    # =========================================================

    I = np.zeros(
        (height, width),
        dtype=np.float64
    )

    for i in range(12):

        T = temp[i]

        positive = T > 0

        I += np.where(
            positive,
            (T / 5.0) ** 1.514,
            0.0
        )

    # =========================================================
    # Thornthwaiten a
    # =========================================================

    a = (
        6.75e-7 * I**3
        - 7.71e-5 * I**2
        + 1.792e-2 * I
        + 0.49239
    )

    safe_I = np.where(
        I > 0,
        I,
        1.0
    )

    # =========================================================
    # PET
    # =========================================================

    pet_annual = np.zeros(
        (height, width),
        dtype=np.float64
    )

    tilt_rad = np.radians(tilt)

    # =========================================================
    # Kuukausittainen laskenta
    # =========================================================

    for i in range(12):

        T = temp[i]

        positive = T > 0

        pet_month = np.zeros(
            (height, width),
            dtype=np.float64
        )

        # -----------------------------------------------------
        # Thornthwaite
        # -----------------------------------------------------

        pet_month[positive] = (
            16.0
            * (
                10.0
                * T[positive]
                / safe_I[positive]
            ) ** a[positive]
        )

        # =====================================================
        # ORBITAALINEN GEOMETRIA
        # =====================================================

        # Mean anomaly
        M = (
            2.0
            * np.pi
            * month_day[i]
            / orbital_period_days
        )

        # -----------------------------------------------------
        # Keplerin yhtälö
        #
        # M = E - e sin(E)
        # -----------------------------------------------------

        E = M.copy()

        for _ in range(10):

            E -= (
                E
                - eccentricity * np.sin(E)
                - M
            ) / (
                1.0
                - eccentricity * np.cos(E)
            )

        # -----------------------------------------------------
        # Todellinen anomalía
        # -----------------------------------------------------

        true_anomaly = (
            2.0
            * np.arctan2(
                np.sqrt(1.0 + eccentricity)
                * np.sin(E / 2.0),

                np.sqrt(1.0 - eccentricity)
                * np.cos(E / 2.0)
            )
        )

        # -----------------------------------------------------
        # Etäisyys suhteessa keskietäisyyteen
        #
        # r/a = (1 - e²) / (1 + e cos(nu))
        # -----------------------------------------------------

        relative_distance = (
            (1.0 - eccentricity**2)
            / (
                1.0
                + eccentricity
                * np.cos(true_anomaly)
            )
        )

        # =====================================================
        # AURINGON SUUNTA
        # =====================================================

        solar_longitude = (
            np.degrees(true_anomaly)
            - mvelp
        ) % 360.0

        solar_longitude_rad = np.radians(
            solar_longitude
        )

        # =====================================================
        # AURINGON DEKLINAAATIO
        # =====================================================

        declination = np.arcsin(
            np.sin(tilt_rad)
            * np.sin(solar_longitude_rad)
        )

        # =====================================================
        # AURINGONLASKUKULMA
        # =====================================================

        x = (
            -np.tan(phi)
            * np.tan(declination)
        )

        polar_day = x <= -1.0
        polar_night = x >= 1.0

        x_clipped = np.clip(
            x,
            -1.0,
            1.0
        )

        sunset_angle = np.arccos(
            x_clipped
        )

        # =====================================================
        # VALOISAN AJAN OSUUS
        # =====================================================

        daylight_fraction = (
            sunset_angle / np.pi
        )

        daylight_fraction = np.where(
            polar_day,
            1.0,
            daylight_fraction
        )

        daylight_fraction = np.where(
            polar_night,
            0.0,
            daylight_fraction
        )

        # =====================================================
        # PÄIVÄN PITUUS
        # =====================================================

        daylight_hours = (
            rotation_period_h
            * daylight_fraction
        )

        day_length_factor = (
            daylight_hours / 12.0
        )

        # =====================================================
        # STELLAR FLUX
        # =====================================================

        # stellar_flux on flux keskietäisyydellä.
        #
        # Koska säteily noudattaa 1/r²-skaalausta:
        #
        # F(r) = F_mean / (r/a)²

        orbital_flux_factor = (
            1.0
            / relative_distance**2
        )

        instantaneous_stellar_flux = (
            stellar_flux
            * orbital_flux_factor
        )

        # =====================================================
        # NORMALISOINTI
        # =====================================================

        # Thornthwaiten alkuperäinen kaava on empiirinen
        # ja sen 16 mm-kerroin on Maan olosuhteisiin
        # sidottu.
        #
        # Tässä stellar_flux skaalataan Maan
        # referenssifluxiin 1361 W/m².
        #
        # Maa:
        #   1361 / 1361 = 1

        stellar_flux_factor = (
            instantaneous_stellar_flux
            / 1361.0
        )

        # =====================================================
        # KUUKAUDEN PITUUS
        # =====================================================

        month_factor = (
            days[i] / 30.0
        )

        # =====================================================
        # LOPULLINEN PET
        # =====================================================

        pet_month *= (
            day_length_factor
            * month_factor
            * stellar_flux_factor
        )

        pet_annual += pet_month

    # =========================================================
    # LANDMASK
    # =========================================================

    pet_annual = np.where(
        landmask == 1,
        pet_annual,
        np.nan
    )

    return pet_annual





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


def laske_npp_miami_12(lampotila_matriisi, sade_matriisi, nodata_arvo=None):
    """
    Laskee vuotuisen NPP:n (g C / m² / vuosi) Miamin mallilla kuukausidatasta.
    
    Parametrit:
    - lampotila_matriisi: NumPy-taulukko, jossa on kuukausittaiset keskilämpötilat (esim. muotoa [12, Y, X])
    - sade_matriisi: NumPy-taulukko, jossa on kuukausittaiset sademäärät millimetreinä (sääntö sama kuin yllä)
    - nodata_arvo: Arvo, joka jätetään laskennan ulkopuolelle (esim. meri-alueet)
    """
    # Muunnetaan datatyypit liukuluvuiksi laskentaa varten
    T_kuukaudet = np.array(lampotila_matriisi, dtype=float)
    P_kuukaudet = np.array(sade_matriisi, dtype=float)
    
    # Luodaan maski puuttuvalle datalle (NoData)
    if nodata_arvo is not None:
        maski = (T_kuukaudet == nodata_arvo) | (P_kuukaudet == nodata_arvo)
        T_kuukaudet[maski] = np.nan
        P_kuukaudet[maski] = np.nan
        
    # 1. Lasketaan vuotuiset aggregaatit kuukausitason datasta (akseli 0 on yleensä kuukausi-akseli)
    # Käytetään nan-funktioita, jotta NoData-ruudut eivät sotke naapurikoordinaatteja
    T_vuosi = np.nanmean(T_kuukaudet, axis=0) # Vuoden keskilämpötila (°C)
    P_vuosi = np.nansum(P_kuukaudet, axis=0)  # Vuoden kokonaissademäärä (mm)
    
    # 2. Miamin mallin kaavat
    # Lämpötilarajoitteinen NPP
    npp_T = 3000 / (1 + np.exp(1.315 - 0.119 * T_vuosi))
    
    # Sademäärärajoitteinen NPP
    npp_P = 3000 * (1 - np.exp(-0.000664 * P_vuosi))
    
    # 3. NPP on näistä kahdesta minimi (Liebigin minimilaki)
    npp_lopputulos = np.minimum(npp_T, npp_P)
    
    # Palautetaan NoData-arvot takaisin alkuperäisille paikoilleen, jos sellainen oli määritelty
    if nodata_arvo is not None:
        npp_lopputulos[np.isnan(npp_lopputulos)] = nodata_arvo
        
    return npp_lopputulos


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





def calc_monthly_human_habitability(temp, precip):
    """
    temp:   (12, H, W), °C
    precip: (12, H, W), mm/kk

    Palauttaa:
        (12, H, W), 0-100
    """

    # -------------------------
    # 1. LÄMPÖTILA
    # -------------------------

    temp_temperate = np.exp(
        -((temp - 13) ** 2) / (2 * 4 ** 2)
    )

    temp_tropical = np.exp(
        -((temp - 22) ** 2) / (2 * 3 ** 2)
    )

    temp_score = np.maximum(
        temp_temperate,
        temp_tropical * 0.8
    )

    # Äärilämpötilojen rangaistus
    temp_score = np.where(
        (temp > 29) | (temp < -5),
        temp_score * 0.1,
        temp_score
    )

    # -------------------------
    # 2. SADE
    # -------------------------

    rain_score = np.exp(
        -((precip - 80) ** 2) / (2 * 60 ** 2)
    )

    # Erittäin kuiva
    rain_score = np.where(
        precip < 20,
        rain_score * 0.2,
        rain_score
    )

    # -------------------------
    # 3. KUUKAUSI-INDEKSI
    # -------------------------

    score = temp_score * rain_score

    return np.clip(score * 100, 0, 100)

def laske_human_habitability_index_annual(temp, rain):
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


import numpy as np


# ============================================================
# INPUT
# ============================================================
#
# temps12:
#   shape = (12, height, width)
#   kuukausittainen keskilämpötila °C
#
# precips12:
#   shape = (12, height, width)
#   kuukausittainen sademäärä mm/kk
#
# population:
#   shape = (height, width)
#   henkilömäärä / pikseli
#
# Rasterin extent:
#   [-180, 180, -90, 90]
#
# ============================================================


def pixel_area_km2(height, width):
    """
    Laskee jokaisen pikselin pinta-alan km².
    Oletetaan tasavälinen lon/lat-rasteri.

    Palauttaa:
        shape = (height, 1)

    jolloin sitä voidaan broadcastata koko rasterille.
    """

    # Leveysasteen pikselikeskukset
    lat_edges = np.linspace(-90, 90, height + 1)
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2

    # Maapallon säde km
    R = 6371.0088

    # Pikselin leveys- ja pituusasteet radiaaneina
    dlat = np.deg2rad(180 / height)
    dlon = np.deg2rad(360 / width)

    # Pallopinnan pinta-ala:
    #
    # A = R² * dlon * (sin(lat2) - sin(lat1))
    #
    lat1 = np.deg2rad(lat_edges[:-1])
    lat2 = np.deg2rad(lat_edges[1:])

    area = R**2 * dlon * (np.sin(lat2) - np.sin(lat1))

    return area[:, None]


def temperature_suitability(temp):
    """
    Lämpötilan soveltuvuus maataloudelle.

    Maksimi noin 20 °C:ssa.
    Hyvin kylmä ja hyvin kuuma ilmasto saa pienen arvon.

    Palauttaa arvot 0...1.
    """

    # Käytetään Gaussian-tyyppistä käyrää.
    #
    # 20 °C -> 1
    # 10 °C -> ~0.61
    # 30 °C -> ~0.61
    #
    suitability = np.exp(
        -((temp - 20.0) / 12.0) ** 2
    )

    return np.clip(suitability, 0, 1)


def precipitation_suitability(precip):
    """
    Kuukausittaisen sateen soveltuvuus.

    Alle ~40 mm/kk alkaa rajoittaa kasvua.
    Noin 100-200 mm/kk on hyvä.
    Erittäin suuri sade ei automaattisesti lisää tuotantoa.
    """

    low = np.clip(precip / 100.0, 0, 1)

    # Liiallinen sade alkaa heikentää soveltuvuutta
    excess = np.clip((precip - 250.0) / 300.0, 0, 1)

    suitability = low * (1.0 - 0.5 * excess)

    return np.clip(suitability, 0, 1)


def calculate_carrying_capacity(
    temps12,
    precips12,
    population,
    arable_fraction=None,
    food_kcal_per_person_day=2500,
    crop_efficiency=0.25,
):
    """
    Laskee globaalin maatalouden kantokykymallin.

    Parametrit
    ----------
    temps12 : ndarray
        (12, H, W), °C

    precips12 : ndarray
        (12, H, W), mm/kk

    population : ndarray
        (H, W), henkilöä/pikseli

    arable_fraction : ndarray tai None
        (H, W), viljelykelpoisen maan osuus 0...1.
        Jos None, käytetään hyvin karkeaa ilmastoperusteista arviota.

    food_kcal_per_person_day : float
        Keskimääräinen energiantarve.

    crop_efficiency : float
        Karkea hyötysuhde 0...1.

    Palauttaa dictin, jossa on kaikki olennaiset rasterit.
    """

    # --------------------------------------------------------
    # CHECKS
    # --------------------------------------------------------

    temps12 = np.asarray(temps12, dtype=np.float32)
    precips12 = np.asarray(precips12, dtype=np.float32)
    population = np.asarray(population, dtype=np.float32)

    if temps12.ndim != 3:
        raise ValueError("temps12 pitää olla muotoa (12, height, width)")

    if precips12.shape != temps12.shape:
        raise ValueError("temps12 ja precips12 pitää olla saman kokoiset")

    if temps12.shape[0] != 12:
        raise ValueError("Ensimmäisen dimension pitää sisältää 12 kuukautta")

    height, width = temps12.shape[1:]

    if population.shape != (height, width):
        raise ValueError(
            "population pitää olla muotoa (height, width)"
        )

    # --------------------------------------------------------
    # PIXEL AREA
    # --------------------------------------------------------

    area = pixel_area_km2(height, width)

    # --------------------------------------------------------
    # TEMPERATURE
    # --------------------------------------------------------

    temp_suitability_monthly = temperature_suitability(temps12)

    # Kuukausittainen lämpötilasoveltuvuus
    temp_suitability = np.mean(
        temp_suitability_monthly,
        axis=0
    )

    # Kasvukauden pituus.
    #
    # Tässä yksinkertainen määritelmä:
    # kuukausi kuuluu kasvukauteen jos
    # lämpötila > 5 °C.
    growing_months = np.sum(
        temps12 > 5.0,
        axis=0
    )

    # --------------------------------------------------------
    # PRECIPITATION
    # --------------------------------------------------------

    precip_suitability_monthly = precipitation_suitability(
        precips12
    )

    precip_suitability = np.mean(
        precip_suitability_monthly,
        axis=0
    )

    annual_precip = np.sum(
        precips12,
        axis=0
    )

    # --------------------------------------------------------
    # WATER STRESS
    # --------------------------------------------------------

    # Jos vuosittainen sademäärä jää alle 300 mm,
    # syntyy voimakas vesirajoite.
    water_factor = np.clip(
        annual_precip / 800.0,
        0,
        1
    )

    # --------------------------------------------------------
    # GROWING SEASON
    # --------------------------------------------------------

    growing_factor = np.clip(
        growing_months / 7.0,
        0,
        1
    )

    # --------------------------------------------------------
    # CLIMATE SUITABILITY
    # --------------------------------------------------------

    climate_suitability = (
        temp_suitability
        * precip_suitability
        * water_factor
        * growing_factor
    )

    climate_suitability = np.clip(
        climate_suitability,
        0,
        1
    )

    # --------------------------------------------------------
    # ARABLE LAND
    # --------------------------------------------------------

    if arable_fraction is None:

        # Hyvin karkea proxy.
        #
        # Oikeassa mallissa tähän kannattaa antaa esim.
        # maankäyttörasteri.

        arable_fraction = (
            climate_suitability ** 1.5
        )

        # Vähintään erittäin kylmät/kuivat alueet pois
        arable_fraction = np.where(
            (growing_months >= 3) & (annual_precip >= 250),
            arable_fraction,
            0
        )

    else:
        arable_fraction = np.asarray(
            arable_fraction,
            dtype=np.float32
        )

        if arable_fraction.shape != (height, width):
            raise ValueError(
                "arable_fraction pitää olla (height, width)"
            )

    arable_fraction = np.clip(
        arable_fraction,
        0,
        1
    )

    # --------------------------------------------------------
    # AGRICULTURAL PRODUCTIVITY
    # --------------------------------------------------------

    # Teoreettinen perustuotanto.
    #
    # climate_suitability = 1
    # -> 10 000 kg biomassaa / ha / vuosi
    #
    # Tämä on parametrisoitava myöhemmin.

    max_yield_kg_ha = 10000.0

    yield_kg_ha = (
        max_yield_kg_ha
        * climate_suitability
        * crop_efficiency
    )

    # --------------------------------------------------------
    # TOTAL AGRICULTURAL LAND
    # --------------------------------------------------------

    # 1 km² = 100 ha
    agricultural_area_ha = (
        area
        * arable_fraction
        * 100.0
    )

    # --------------------------------------------------------
    # FOOD PRODUCTION
    # --------------------------------------------------------

    food_kg = (
        agricultural_area_ha
        * yield_kg_ha
    )

    # Muutetaan ruoan määrä energiaksi.
    #
    # Oletetaan erittäin karkeasti:
    # 1 kg syötävää kasviperäistä biomassaa = 2500 kcal

    kcal_per_kg = 2500.0

    food_kcal_year = (
        food_kg
        * kcal_per_kg
    )

    # --------------------------------------------------------
    # HUMAN CARRYING CAPACITY
    # --------------------------------------------------------

    kcal_person_year = (
        food_kcal_per_person_day
        * 365.25
    )

    carrying_capacity = (
        food_kcal_year
        / kcal_person_year
    )

    carrying_capacity = np.maximum(
        carrying_capacity,
        0
    )

    # --------------------------------------------------------
    # POPULATION DENSITY
    # --------------------------------------------------------

    population_density = (
        population / area
    )

    # --------------------------------------------------------
    # CAPACITY RATIO
    # --------------------------------------------------------

    # >1 = kapasiteettia enemmän kuin väestöä
    # <1 = väestö ylittää mallin kapasiteetin

    capacity_ratio = (
        carrying_capacity
        / np.maximum(population, 1e-6)
    )

    # --------------------------------------------------------
    # POPULATION PRESSURE
    # --------------------------------------------------------

    population_pressure = (
        population
        / np.maximum(carrying_capacity, 1e-6)
    )

    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

    return {
        "pixel_area_km2": area,

        "annual_precip_mm": annual_precip,

        "temperature_suitability": temp_suitability,

        "precipitation_suitability": precip_suitability,

        "growing_months": growing_months,

        "water_factor": water_factor,

        "growing_factor": growing_factor,

        "climate_suitability": climate_suitability,

        "arable_fraction": arable_fraction,

        "agricultural_area_ha": agricultural_area_ha,

        "yield_kg_ha": yield_kg_ha,

        "food_production_kg": food_kg,

        "carrying_capacity": carrying_capacity,

        "population_density": population_density,

        "capacity_ratio": capacity_ratio,

        "population_pressure": population_pressure,
    }



# ============================================================
# BIOCLIM-TYYPPISET MUUTTUJAT
# ============================================================

def derive_bio(temps12, precips12):
    """
    temps12  : (12, H, W), °C
    precips12: (12, H, W), mm/kk
    """

    t = np.asarray(temps12, dtype=np.float32)
    p = np.asarray(precips12, dtype=np.float32)

    # Vuotuinen keskilämpötila
    bio1 = np.mean(t, axis=0)

    # Lämpötilan vaihtelu
    bio4 = np.std(t, axis=0)

    # Lämpimin ja kylmin kuukausi
    bio5 = np.max(t, axis=0)
    bio6 = np.min(t, axis=0)

    # 3 kk liukuvat keskiarvot
    tq = np.stack([
        (
            t[i]
            + t[(i + 1) % 12]
            + t[(i + 2) % 12]
        ) / 3
        for i in range(12)
    ])

    bio10 = np.max(tq, axis=0)
    bio11 = np.min(tq, axis=0)

    # Vuosisade
    bio12 = np.sum(p, axis=0)

    # Märin ja kuivin kuukausi
    bio13 = np.max(p, axis=0)
    bio14 = np.min(p, axis=0)

    # Sateen kausivaihtelu
    bio15 = (
        np.std(p, axis=0)
        / np.maximum(np.mean(p, axis=0), 0.01)
        * 100
    )

    # 3 kk sademäärät
    pq = np.stack([
        (
            p[i]
            + p[(i + 1) % 12]
            + p[(i + 2) % 12]
        )
        for i in range(12)
    ])

    bio16 = np.max(pq, axis=0)
    bio17 = np.min(pq, axis=0)

    return {
        "bio1": bio1,
        "bio4": bio4,
        "bio5": bio5,
        "bio6": bio6,
        "bio10": bio10,
        "bio11": bio11,
        "bio12": bio12,
        "bio13": bio13,
        "bio14": bio14,
        "bio15": bio15,
        "bio16": bio16,
        "bio17": bio17,
    }


# ============================================================
# PEHMEÄ SOPIVUUSFUNKTION
# ============================================================

def gaussian_score(x, optimum, width):
    """
    0...1 oleva pehmeä soveltuvuus.
    """

    return np.exp(
        -0.5 * ((x - optimum) / width) ** 2
    ).astype(np.float32)


def range_score(x, low, high, width):
    """
    0...1 oleva soveltuvuus.
    1 alueella low...high,
    jonka ulkopuolella arvo pienenee pehmeästi.
    """

    score_low = np.ones_like(x, dtype=np.float32)
    score_high = np.ones_like(x, dtype=np.float32)

    mask = x < low
    score_low[mask] = np.exp(
        -0.5 * ((x[mask] - low) / width) ** 2
    )

    mask = x > high
    score_high[mask] = np.exp(
        -0.5 * ((x[mask] - high) / width) ** 2
    )

    return score_low * score_high


# ============================================================
# KASVIKOHTAINEN ILMASTOSOPIVUUS
# ============================================================

def crop_suitability(bio, crop):
    """
    Palauttaa kasvin ilmastosopivuuden 0...1.
    """

    b1 = bio["bio1"]
    b5 = bio["bio5"]
    b6 = bio["bio6"]

    b10 = bio["bio10"]
    b11 = bio["bio11"]

    b12 = bio["bio12"]
    b14 = bio["bio14"]
    b17 = bio["bio17"]

    # --------------------------------------------------------
    # KASVIKOHTAISET PARAMETRIT
    #
    # Nämä ovat alustavia, eivät lopullisia agronomisia arvoja.
    # --------------------------------------------------------

    params = {

        "maize": {
            "temp": 20,
            "temp_width": 8,
            "hot_max": 35,
            "cold_min": 5,
            "precip": 700,
            "precip_width": 500,
            "dry_month": 40,
            "dry_quarter": 120,
        },

        "potato": {
            "temp": 14,
            "temp_width": 7,
            "hot_max": 30,
            "cold_min": 2,
            "precip": 600,
            "precip_width": 400,
            "dry_month": 30,
            "dry_quarter": 100,
        },

        "wheat": {
            "temp": 15,
            "temp_width": 8,
            "hot_max": 32,
            "cold_min": -5,
            "precip": 500,
            "precip_width": 350,
            "dry_month": 25,
            "dry_quarter": 80,
        },

        "rice": {
            "temp": 24,
            "temp_width": 6,
            "hot_max": 38,
            "cold_min": 10,
            "precip": 1200,
            "precip_width": 600,
            "dry_month": 70,
            "dry_quarter": 200,
        },

        "soy": {
            "temp": 22,
            "temp_width": 7,
            "hot_max": 35,
            "cold_min": 5,
            "precip": 700,
            "precip_width": 450,
            "dry_month": 40,
            "dry_quarter": 120,
        },
    }

    q = params[crop]

    # --------------------------------------------------------
    # LÄMPÖTILA
    # --------------------------------------------------------

    score_temp = gaussian_score(
        b1,
        q["temp"],
        q["temp_width"]
    )

    # Liian kuuma
    score_hot = np.ones_like(b5)

    mask = b5 > q["hot_max"]

    score_hot[mask] = np.exp(
        -0.5
        * ((b5[mask] - q["hot_max"]) / 5) ** 2
    )

    # Liian kylmä
    score_cold = np.ones_like(b6)

    mask = b6 < q["cold_min"]

    score_cold[mask] = np.exp(
        -0.5
        * ((b6[mask] - q["cold_min"]) / 5) ** 2
    )

    # --------------------------------------------------------
    # KASVUKAUDEN LÄMPÖTILA
    # --------------------------------------------------------

    score_growing_temp = gaussian_score(
        b10,
        q["temp"],
        q["temp_width"]
    )

    # --------------------------------------------------------
    # SADE
    # --------------------------------------------------------

    score_precip = gaussian_score(
        b12,
        q["precip"],
        q["precip_width"]
    )

    # Kuivin kuukausi
    score_dry_month = range_score(
        b14,
        q["dry_month"],
        np.inf,
        40
    )

    # Kuivin neljännes
    score_dry_quarter = range_score(
        b17,
        q["dry_quarter"],
        np.inf,
        100
    )

    # --------------------------------------------------------
    # YHDISTÄ
    # --------------------------------------------------------

    score = (
        score_temp
        * score_hot
        * score_cold
        * score_growing_temp
        * score_precip
        * score_dry_month
        * score_dry_quarter
    )

    return np.clip(score, 0, 1)


# ============================================================
# KOKO MAATALOUSPOTENTIAALI
# ============================================================

def agricultural_potential(
    temps12,
    precips12,
    relief=None,
    rivers1=None,
    population=None,
):
    """
    Ensimmäinen kokeellinen globaali maatalousmalli.

    relief ja rivers1 otetaan mukaan rajapintaan,
    mutta niitä EI vielä käytetä suoraan rajoittamaan
    tuotantoa.

    Tämä on tarkoituksellista:
    emme halua tehdä oletuksia ennen kuin tiedämme,
    miten niitä halutaan käyttää.
    """

    bio = derive_bio(
        temps12,
        precips12
    )

    crops = [
        "maize",
        "potato",
        "wheat",
        "rice",
        "soy",
    ]

    suitability = {}

    for crop in crops:
        suitability[crop] = crop_suitability(
            bio,
            crop
        )

    # --------------------------------------------------------
    # PARAS KASVI
    # --------------------------------------------------------

    crop_stack = np.stack([
        suitability[c]
        for c in crops
    ])

    best_index = np.argmax(
        crop_stack,
        axis=0
    )

    best_suitability = np.max(
        crop_stack,
        axis=0
    )

    best_crop = np.empty(
        best_index.shape,
        dtype=object
    )

    for i, crop in enumerate(crops):
        best_crop[best_index == i] = crop

    # --------------------------------------------------------
    # KASVUKAUSI
    # --------------------------------------------------------

    growing_months = np.sum(
        temps12 > 5,
        axis=0
    )

    # --------------------------------------------------------
    # VESISTRESSI
    # --------------------------------------------------------

    annual_precip = bio["bio12"]

    # Tämä ei vielä ole "kastelun puute",
    # vaan luonnollisen veden saatavuuden karkea indeksi.
    water_score = np.clip(
        annual_precip / 800,
        0,
        1
    )

    # --------------------------------------------------------
    # KOKONAISPOTENTIAALI
    # --------------------------------------------------------

    agricultural_score = (
        best_suitability
        * water_score
    )

    return {
        "bio": bio,

        "maize_suitability": suitability["maize"],
        "potato_suitability": suitability["potato"],
        "wheat_suitability": suitability["wheat"],
        "rice_suitability": suitability["rice"],
        "soy_suitability": suitability["soy"],

        "best_crop": best_crop,
        "best_suitability": best_suitability,

        "growing_months": growing_months,

        "annual_precip": annual_precip,
        "water_score": water_score,

        "agricultural_score": agricultural_score,
    }



import numpy as np
from scipy.ndimage import distance_transform_edt


# ============================================================
# PIXEL AREA
# ============================================================

def pixel_area_km2(height, width):
    """
    Maapallon lon/lat-rasterin pikselipinta-ala km².

    Oletus:
        extent = [-180, 180, -90, 90]
    """

    R = 6371.0088

    lat_edges = np.linspace(
        -90,
        90,
        height + 1
    )

    lat1 = np.deg2rad(lat_edges[:-1])
    lat2 = np.deg2rad(lat_edges[1:])

    dlon = np.deg2rad(360 / width)

    area = (
        R**2
        * dlon
        * (np.sin(lat2) - np.sin(lat1))
    )

    return area[:, None].astype(np.float32)


# ============================================================
# BIOCLIM
# ============================================================

def derive_bio(temps12, precips12):

    t = np.asarray(
        temps12,
        dtype=np.float32
    )

    p = np.asarray(
        precips12,
        dtype=np.float32
    )

    # -----------------------------
    # Temperature
    # -----------------------------

    bio1 = np.mean(t, axis=0)

    bio4 = np.std(t, axis=0)

    bio5 = np.max(t, axis=0)

    bio6 = np.min(t, axis=0)

    tq = np.stack([
        (
            t[i]
            + t[(i + 1) % 12]
            + t[(i + 2) % 12]
        ) / 3
        for i in range(12)
    ])

    bio10 = np.max(tq, axis=0)
    bio11 = np.min(tq, axis=0)

    # -----------------------------
    # Precipitation
    # -----------------------------

    bio12 = np.sum(p, axis=0)

    bio13 = np.max(p, axis=0)

    bio14 = np.min(p, axis=0)

    bio15 = (
        np.std(p, axis=0)
        / np.maximum(
            np.mean(p, axis=0),
            0.01
        )
        * 100
    )

    pq = np.stack([
        (
            p[i]
            + p[(i + 1) % 12]
            + p[(i + 2) % 12]
        )
        for i in range(12)
    ])

    bio16 = np.max(pq, axis=0)
    bio17 = np.min(pq, axis=0)

    return {
        "bio1": bio1,
        "bio4": bio4,
        "bio5": bio5,
        "bio6": bio6,
        "bio10": bio10,
        "bio11": bio11,

        "bio12": bio12,
        "bio13": bio13,
        "bio14": bio14,
        "bio15": bio15,
        "bio16": bio16,
        "bio17": bio17,
    }


# ============================================================
# HELPERS
# ============================================================

def gaussian_score(x, optimum, width):

    return np.exp(
        -0.5
        * ((x - optimum) / width) ** 2
    ).astype(np.float32)


def lower_limit_score(x, limit, width):
    """
    1 kun x >= limit.
    Alle rajan score pienenee pehmeästi.
    """

    score = np.ones_like(
        x,
        dtype=np.float32
    )

    mask = x < limit

    score[mask] = np.exp(
        -0.5
        * ((x[mask] - limit) / width) ** 2
    )

    return score


def upper_limit_score(x, limit, width):
    """
    1 kun x <= limit.
    Yli rajan score pienenee.
    """

    score = np.ones_like(
        x,
        dtype=np.float32
    )

    mask = x > limit

    score[mask] = np.exp(
        -0.5
        * ((x[mask] - limit) / width) ** 2
    )

    return score


# ============================================================
# CROP MODEL
# ============================================================

CROPS = {

    "maize": {
        "temp_opt": 20,
        "temp_width": 8,
        "cold_min": 5,
        "hot_max": 35,

        "precip_opt": 700,
        "precip_width": 500,

        "dry_month_min": 30,
        "dry_quarter_min": 100,

        # kg/ha/year hyvässä ympäristössä
        "yield": 8000,

        # Ruokakalorit / kg
        "kcal_kg": 3600,
    },

    "potato": {
        "temp_opt": 14,
        "temp_width": 7,
        "cold_min": 2,
        "hot_max": 30,

        "precip_opt": 600,
        "precip_width": 400,

        "dry_month_min": 30,
        "dry_quarter_min": 100,

        "yield": 20000,
        "kcal_kg": 770,
    },

    "wheat": {
        "temp_opt": 15,
        "temp_width": 8,
        "cold_min": -5,
        "hot_max": 32,

        "precip_opt": 500,
        "precip_width": 350,

        "dry_month_min": 20,
        "dry_quarter_min": 70,

        "yield": 5000,
        "kcal_kg": 3400,
    },

    "rice": {
        "temp_opt": 24,
        "temp_width": 6,
        "cold_min": 10,
        "hot_max": 38,

        "precip_opt": 1200,
        "precip_width": 600,

        "dry_month_min": 70,
        "dry_quarter_min": 200,

        "yield": 6000,
        "kcal_kg": 3600,
    },

    "soy": {
        "temp_opt": 22,
        "temp_width": 7,
        "cold_min": 5,
        "hot_max": 35,

        "precip_opt": 700,
        "precip_width": 450,

        "dry_month_min": 40,
        "dry_quarter_min": 120,

        "yield": 3500,
        "kcal_kg": 4400,
    },
}


def crop_suitability(bio, crop):

    q = CROPS[crop]

    score_temp = gaussian_score(
        bio["bio1"],
        q["temp_opt"],
        q["temp_width"]
    )

    score_hot = upper_limit_score(
        bio["bio5"],
        q["hot_max"],
        5
    )

    score_cold = lower_limit_score(
        bio["bio6"],
        q["cold_min"],
        5
    )

    score_growing_temp = gaussian_score(
        bio["bio10"],
        q["temp_opt"],
        q["temp_width"]
    )

    score_precip = gaussian_score(
        bio["bio12"],
        q["precip_opt"],
        q["precip_width"]
    )

    score_dry_month = lower_limit_score(
        bio["bio14"],
        q["dry_month_min"],
        40
    )

    score_dry_quarter = lower_limit_score(
        bio["bio17"],
        q["dry_quarter_min"],
        100
    )

    # Geometrinen yhdistelmä:
    #
    # jos yksi tekijä on todella huono,
    # se rajoittaa kokonaisuutta.

    score = (
        score_temp
        * score_hot
        * score_cold
        * score_growing_temp
        * score_precip
        * score_dry_month
        * score_dry_quarter
    )

    return np.clip(
        score,
        0,
        1
    ).astype(np.float32)


# ============================================================
# RIVER DISTANCE
# ============================================================

def river_distance_km(
    rivers1,
    width
):
    """
    Laskee etäisyyden lähimpään jokipikseliin.

    rivers1:
        1 = joki
        0 = ei jokea

    Huom:
        lon/lat-rasterissa yhden pikselin leveys
        riippuu leveysasteesta.

    Tämä on tässä tarkoituksella vain karkea
    approksimaatio.
    """

    river = np.asarray(
        rivers1
    ) > 0

    # Pikselietäisyys lähimpään jokeen
    distance_pixels = distance_transform_edt(
        ~river
    )

    # Keskimääräinen pikselikoko.
    # Tarkempi versio voidaan tehdä myöhemmin
    # leveysastekohtaisella etäisyydellä.

    km_per_pixel = 111.32 * (
        360 / width
    )

    return (
        distance_pixels
        * km_per_pixel
    ).astype(np.float32)


# ============================================================
# IRRIGATION POTENTIAL
# ============================================================

def irrigation_potential(
    rivers1,
    width
):
    """
    Joen läheisyyteen perustuva kastelupotentiaali.

    Tämä EI tarkoita että kaikki joen lähellä oleva
    maa todella kastellaan.

    Se tarkoittaa:
        "vesi voisi olla suhteellisen helposti
         saavutettavissa."
    """

    distance = river_distance_km(
        rivers1,
        width
    )

    # 20 km mittakaava
    access = np.exp(
        -distance / 20
    )

    return (
        distance,
        access.astype(np.float32)
    )


# ============================================================
# MAIN AGRICULTURAL MODEL
# ============================================================

def agricultural_carrying_capacity(
    temps12,
    precips12,
    relief,
    rivers1,
    population,
    arable_fraction=0.5,

    # Teknologinen kastelun hyödyntämisaste
    irrigation_technology=0.30,

    # Kuinka paljon tuotannosta päätyy ihmisten
    # käytettäväksi ruoaksi
    food_fraction=0.70,

    # kcal / henkilö / päivä
    kcal_person_day=2500,
):
    """
    Globaali kokeellinen maatalouden kantokykymalli.

    INPUT
    -----

    temps12:
        (12,H,W), °C

    precips12:
        (12,H,W), mm/kk

    relief:
        (H,W), metriä

    rivers1:
        (H,W), jokimaski

    population:
        (H,W), henkilöä/pikseli

    arable_fraction:
        oletettu viljelykelpoisen maan osuus.

    irrigation_technology:
        kuinka tehokkaasti jokien tarjoamaa
        vettä voidaan hyödyntää.

    OUTPUT
    ------

    dict, jossa useita rastereita.
    """

    t = np.asarray(
        temps12,
        dtype=np.float32
    )

    p = np.asarray(
        precips12,
        dtype=np.float32
    )

    pop = np.asarray(
        population,
        dtype=np.float32
    )

    relief = np.asarray(
        relief,
        dtype=np.float32
    )

    H, W = t.shape[1:]

    # --------------------------------------------------------
    # AREA
    # --------------------------------------------------------

    area_km2 = pixel_area_km2(
        H,
        W
    )

    area_ha = (
        area_km2
        * 100
    )

    # --------------------------------------------------------
    # BIO
    # --------------------------------------------------------

    bio = derive_bio(
        t,
        p
    )

    # --------------------------------------------------------
    # CROP SUITABILITY
    # --------------------------------------------------------

    crop_scores = {}

    for crop in CROPS:

        crop_scores[crop] = crop_suitability(
            bio,
            crop
        )

    crop_stack = np.stack(
        list(crop_scores.values())
    )

    crop_names = list(
        crop_scores.keys()
    )

    best_index = np.argmax(
        crop_stack,
        axis=0
    )

    best_score = np.max(
        crop_stack,
        axis=0
    )

    best_crop = np.empty(
        best_index.shape,
        dtype=object
    )

    for i, crop in enumerate(crop_names):

        best_crop[
            best_index == i
        ] = crop

    # --------------------------------------------------------
    # GROWING SEASON
    # --------------------------------------------------------

    growing_months = np.sum(
        t > 5,
        axis=0
    )

    growing_factor = np.clip(
        growing_months / 6,
        0,
        1
    )

    # --------------------------------------------------------
    # RIVER / IRRIGATION
    # --------------------------------------------------------

    river_distance, irrigation_access = (
        irrigation_potential(
            rivers1,
            W
        )
    )

    # Kastelu voi parantaa kuivien alueiden
    # tuotantopotentiaalia.
    #
    # Se ei kuitenkaan tee aavikosta automaattisesti
    # yhtä hyvää kuin hyvästä sadeviljelyalueesta.

    natural_water = np.clip(
        bio["bio12"] / 800,
        0,
        1
    )

    water_with_irrigation = (
        natural_water
        + (
            (1 - natural_water)
            * irrigation_access
            * irrigation_technology
        )
    )

    water_with_irrigation = np.clip(
        water_with_irrigation,
        0,
        1
    )

    # --------------------------------------------------------
    # AGRICULTURAL SCORE
    # --------------------------------------------------------

    agricultural_score = (
        best_score
        * growing_factor
        * water_with_irrigation
    )

    # --------------------------------------------------------
    # AVAILABLE AGRICULTURAL LAND
    # --------------------------------------------------------

    agricultural_area_ha = (
        area_ha
        * arable_fraction
        * agricultural_score
    )

    # --------------------------------------------------------
    # FOOD PRODUCTION
    # --------------------------------------------------------

    # Kasvikohtainen maksimisato.
    # Otetaan jokaiselle pikselille sen parhaan kasvin
    # vastaava sato.

    yield_kg_ha = np.zeros(
        (H, W),
        dtype=np.float32
    )

    kcal_per_kg = np.zeros(
        (H, W),
        dtype=np.float32
    )

    for i, crop in enumerate(crop_names):

        mask = best_index == i

        yield_kg_ha[mask] = (
            CROPS[crop]["yield"]
        )

        kcal_per_kg[mask] = (
            CROPS[crop]["kcal_kg"]
        )

    # Toteutuva sato
    actual_yield_kg_ha = (
        yield_kg_ha
        * agricultural_score
    )

    food_kg = (
        agricultural_area_ha
        * actual_yield_kg_ha
    )

    food_kcal_year = (
        food_kg
        * kcal_per_kg
        * food_fraction
    )

    # --------------------------------------------------------
    # CARRYING CAPACITY
    # --------------------------------------------------------

    kcal_person_year = (
        kcal_person_day
        * 365.25
    )

    carrying_capacity = (
        food_kcal_year
        / kcal_person_year
    )

    carrying_capacity = np.maximum(
        carrying_capacity,
        0
    )

    # --------------------------------------------------------
    # POPULATION DENSITY
    # --------------------------------------------------------

    population_density = (
        pop
        / np.maximum(
            area_km2,
            1e-12
        )
    )

    # --------------------------------------------------------
    # CAPACITY RATIO
    # --------------------------------------------------------

    capacity_ratio = (
        carrying_capacity
        / np.maximum(
            pop,
            1e-6
        )
    )

    # --------------------------------------------------------
    # PRESSURE
    # --------------------------------------------------------

    population_pressure = (
        pop
        / np.maximum(
            carrying_capacity,
            1e-6
        )
    )

    return {

        # -------------------------
        # BASIC
        # -------------------------

        "pixel_area_km2":
            area_km2,

        "population":
            pop,

        "population_density":
            population_density,

        # -------------------------
        # BIOCLIM
        # -------------------------

        "bio":
            bio,

        # -------------------------
        # CROPS
        # -------------------------

        "maize_suitability":
            crop_scores["maize"],

        "potato_suitability":
            crop_scores["potato"],

        "wheat_suitability":
            crop_scores["wheat"],

        "rice_suitability":
            crop_scores["rice"],

        "soy_suitability":
            crop_scores["soy"],

        "best_crop":
            best_crop,

        "best_crop_suitability":
            best_score,

        # -------------------------
        # WATER
        # -------------------------

        "river_distance_km":
            river_distance,

        "irrigation_access":
            irrigation_access,

        "water_score":
            water_with_irrigation,

        # -------------------------
        # AGRICULTURE
        # -------------------------

        "growing_months":
            growing_months,

        "agricultural_score":
            agricultural_score,

        "agricultural_area_ha":
            agricultural_area_ha,

        "yield_kg_ha":
            actual_yield_kg_ha,

        "food_kg":
            food_kg,

        "food_kcal_year":
            food_kcal_year,

        # -------------------------
        # PEOPLE
        # -------------------------

        "carrying_capacity":
            carrying_capacity,

        "capacity_ratio":
            capacity_ratio,

        "population_pressure":
            population_pressure,

        # Relief säilytetään tuloksissa,
        # mutta sitä ei käytetä vielä suoraan
        # rajoittavana tekijänä.

        "relief":
            relief,
    }







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

import math


# ============================================================
# VAKIOT
# ============================================================

SIGMA = 5.670374419e-8

# Maan luonnollisen kasvihuoneilmiön vaikutus
MAA_GHG_DELTA_T = 31.82

import math


# ============================================================
# VAKIOT
# ============================================================

SIGMA = 5.670374419e-8

# Maan luonnollisen kasvihuoneilmiön vaikutus °C
MAA_GHG_DELTA_T = 31.82

# ============================================================
# PYÖRIMISEN KALIBROINTI
# ============================================================

# Pyörimisen vaikutuksen voimakkuus °C.
#
# 1 vrk = 0 °C
# 2 vrk = -1.5 °C
# 4 vrk = -3.0 °C
# 0.5 vrk = +1.5 °C
#
# Arvoa voi säätää.
ROTATION_DELTA_T_PER_LOG2 = 1.5

# Kuinka voimakkaasti pyöriminen vaikuttaa päivä–yö-vaihteluun.
#
# 1.0 = täysi vaikutus
# 0.0 = ei vaikutusta
PAIVA_YO_ROTATION_KERROIN = 1.0


def laske_planeetan_lampotila(
    ecc,
    tilt,
    mvelp,
    S1,
    atmos_co2_ppm,

    # Planeetan ominaisuudet
    planet_atmosphere_pressure=1.0,
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

    # Lämmönkuljetus
    lammon_kuljetus_paivantaasaajalta_navoille=3.5,

    # Lämpökapasiteetti
    lampo_kapasiteetti=1.0,

    # Vuodenaikaisvaihtelu
    vuodenaika_kerroin=1.0,
):
    """
    Planeetan yksinkertaistettu lämpötilamalli.

    Kaikki lämpötilat palautetaan Celsiusasteina.

    Pyöriminen:
        rotation_period_days < 1:
            nopeampi pyöriminen

        rotation_period_days = 1:
            Maan kaltainen pyöriminen

        rotation_period_days > 1:
            hitaampi pyöriminen

    Tässä mallissa:
        nopeampi pyöriminen -> hieman lämpimämpi keskiarvo
        hitaampi pyöriminen -> hieman kylmempi keskiarvo

    Lisäksi pyöriminen vaikuttaa päivä–yö-vaihteluun.
    """

    # ============================================================
    # 1. TARKISTUKSET
    # ============================================================

    if not 0.0 <= ecc < 1.0:
        raise ValueError(
            "ecc pitää olla välillä 0...1."
        )

    if not 0.0 <= tilt <= 90.0:
        raise ValueError(
            "tilt pitää olla välillä 0...90 astetta."
        )

    if S1 <= 0:
        raise ValueError(
            "S1 pitää olla > 0."
        )

    if atmos_co2_ppm <= 0:
        raise ValueError(
            "CO2-pitoisuuden pitää olla > 0 ppm."
        )

    if co2_viite_ppm <= 0:
        raise ValueError(
            "CO2-vertailuarvon pitää olla > 0 ppm."
        )

    if planet_atmosphere_pressure <= 0:
        raise ValueError(
            "planet_atmosphere_pressure pitää olla > 0."
        )

    if planet_radius_re <= 0:
        raise ValueError(
            "planet_radius_re pitää olla > 0."
        )

    if rotation_period_days <= 0:
        raise ValueError(
            "rotation_period_days pitää olla > 0."
        )

    if lampo_kapasiteetti <= 0:
        raise ValueError(
            "lampo_kapasiteetti pitää olla > 0."
        )

    if vesihoyry_kerroin < 0:
        raise ValueError(
            "vesihoyry_kerroin ei voi olla negatiivinen."
        )

    if not 0 <= albedo_keski < 1:
        raise ValueError(
            "albedo pitää olla välillä 0...1."
        )

    # ============================================================
    # 2. GLOBAALI ABSORBOITUVA AURINKOENERGIA
    # ============================================================

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

    delta_T_vesihoyry = (
        MAA_GHG_DELTA_T
        * vesihoyry_kerroin
    )

    if kasvihuone_kerroin is None:
        kasvihuone_kerroin = 1.0

    delta_T_natural_ghg = (
        delta_T_vesihoyry
        * kasvihuone_kerroin
    )

    # ============================================================
    # 5. CO2:N SÄTEILYPAKOTE
    # ============================================================

    dF_co2 = (
        5.35
        * math.log(
            (planet_atmosphere_pressure * atmos_co2_ppm)
            / co2_viite_ppm
        )
    )

    delta_T_co2 = (
        ilmastosensitiivisyys
        * dF_co2
    )

    # ============================================================
    # 6. PYÖRIMISEN VAIKUTUS KESKILÄMPÖTILAAN
    # ============================================================

    # Maan pyörimisaika:
    #
    #     1 vrk -> 0 °C
    #
    # Hitaampi:
    #
    #     2 vrk -> negatiivinen
    #     4 vrk -> enemmän negatiivinen
    #
    # Nopeampi:
    #
    #     0.5 vrk -> positiivinen
    #     0.25 vrk -> vielä positiivisempi
    #
    # Logaritminen riippuvuus estää liian voimakkaan
    # kasvun erittäin pitkillä tai lyhyillä pyörimisajoilla.

    rotation_delta_T_C = (
        -ROTATION_DELTA_T_PER_LOG2
        * math.log2(rotation_period_days)
    )

    # ============================================================
    # 7. GLOBAALI KESKILÄMPÖTILA
    # ============================================================

    T_global_C = (
        T_effective_C
        + delta_T_natural_ghg
        + delta_T_co2
        + rotation_delta_T_C
    )

    # ============================================================
    # 8. ALUEELLINEN AURINKOENERGIA
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
    # 9. ALUEELLINEN EFEKTIIVINEN LÄMPÖTILA
    # ============================================================

    def vuo_to_effective_temp(vuo, geometria):

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
    # 10. ALUEELLISET POIKKEAMAT GLOBAALISTA
    # ============================================================

    T_eq_anomalia = (
        T_eq_effective
        - T_effective_C
    )

    T_pole_N_anomalia = (
        T_pole_N_effective
        - T_effective_C
    )

    T_pole_S_anomalia = (
        T_pole_S_effective
        - T_effective_C
    )

    # ============================================================
    # 11. ALUEELLISET LÄMPÖTILAT ENNEN KULJETUSTA
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
    # 12. LÄMMÖNKULJETUS
    # ============================================================

    # Hitaampi pyöriminen kasvattaa päivä–yö-jakson pituutta.
    #
    # Nopeampi pyöriminen:
    #     lyhyempi päivä/yö
    #
    # Hitaampi pyöriminen:
    #     pidempi päivä/yö
    #
    # Tässä lämmönkuljetus kasvaa hitaammalla pyörimisellä.
    # Ilmakehän paine kasvattaa kuljetusta.
    #
    # Planeetan suurempi koko kasvattaa kuljetusmatkaa.

    kuljetuskerroin = (
        lammon_kuljetus_paivantaasaajalta_navoille
        * math.sqrt(rotation_period_days)
        * math.sqrt(planet_radius_re)
        * planet_atmosphere_pressure
    )

    delta_T_eq_poles = (
        delta_T_eq_poles_raw
        / (1.0 + kuljetuskerroin)
    )

    # ============================================================
    # 13. ALUEELLISET LÄMPÖTILAT
    # ============================================================

    T_poles = (
        T_global_C
        - delta_T_eq_poles / 2.0
    )

    T_eq = (
        T_global_C
        + delta_T_eq_poles / 2.0
    )

    # Pohjoisen ja eteläisen navan ero.

    pole_N_difference = (
        T_pole_N_raw
        - T_poles_raw
    )

    pole_S_difference = (
        T_pole_S_raw
        - T_poles_raw
    )

    T_pole_N = (
        T_poles
        + pole_N_difference
        / (1.0 + kuljetuskerroin)
    )

    T_pole_S = (
        T_poles
        + pole_S_difference
        / (1.0 + kuljetuskerroin)
    )

    # ============================================================
    # 14. PÄIVÄ–YÖ-VAIHTELU
    # ============================================================

    # Perusvaihtelu suhteessa globaaliin lämpötilaan.
    #
    # Hitaampi pyöriminen -> suurempi päivä–yö-vaihtelu.
    #
    # Nopeampi pyöriminen -> pienempi vaihtelu.

    paiva_yo_kerroin = (
        math.sqrt(rotation_period_days)
        * PAIVA_YO_ROTATION_KERROIN
    )

    # Ilmakehä vaimentaa päivä–yö-vaihtelua.
    #
    # Suurempi ilmanpaine -> tehokkaampi lämmönkuljetus.

    paiva_yo_vaimennys = (
        1.0
        / math.sqrt(planet_atmosphere_pressure)
    )

    # Planeetan lämpökapasiteetti vaimentaa vaihtelua.

    paiva_yo_vaihtelu_C = (
        8.0
        * paiva_yo_kerroin
        * paiva_yo_vaimennys
        / math.sqrt(lampo_kapasiteetti)
    )

    # Erittäin pitkällä pyörimisajalla estetään
    # epärealistisen suuri arvo.

    paiva_yo_vaihtelu_C = min(
        paiva_yo_vaihtelu_C,
        60.0
    )

    # ============================================================
    # 15. VUODENAIKAINEN VAIHTELU
    # ============================================================

    delta_T_vuosi = (
        abs(delta_T_eq_poles)
        * vuodenaika_kerroin
        / lampo_kapasiteetti
    )

    delta_t_paivantaasaajalla_vuosi = (
        delta_T_vuosi / 2.0
    )

    delta_t_navoilla_vuosi = (
        -delta_T_vuosi / 2.0
    )

    # ============================================================
    # 16. VUODEN MAKSIMI / MINIMI
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
    # 17. PALAUTUS
    # ============================================================

    return {

        # -------------------------
        # Globaali lämpötila
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
        # Pyöriminen
        # -------------------------

        "pyorimisaika_paivaa":
            rotation_period_days,

        "pyorimisen_delta_T_C":
            rotation_delta_T_C,

        "paiva_yo_vaihtelu_C":
            paiva_yo_vaihtelu_C,

        # -------------------------
        # Alueelliset lämpötilat
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




def luokittele_koppen(temps, precs):
    """
    Luokittelee Köppen–Geiger-ilmastoluokan jokaiselle ruudulle.

    Parametrit
    ----------
    temps : numpy.ndarray
        Kuukausittaiset keskilämpötilat muodossa
        [12, height, width], °C.

    precs : numpy.ndarray
        Kuukausittaiset sademäärät muodossa
        [12, height, width], mm/kk.

    Palauttaa
    ----------
    koppen : numpy.ndarray
        2D-taulukko, jossa Köppen-luokat merkkijonoina.
    """

    # ------------------------------------------------------------
    # Tarkistukset
    # ------------------------------------------------------------

    if temps.ndim != 3 or precs.ndim != 3:
        raise ValueError(
            "temps ja precs pitää olla muodossa [12, height, width]."
        )

    if temps.shape != precs.shape:
        raise ValueError(
            "temps ja precs pitää olla samanmuotoiset."
        )

    kuukausia, height, width = temps.shape

    if kuukausia != 12:
        raise ValueError(
            "temps ja precs pitää sisältää 12 kuukautta."
        )

    # ------------------------------------------------------------
    # Tulostaulukko
    # ------------------------------------------------------------

    koppen = np.full(
        (height, width),
        "??",
        dtype="<U3"
    )

    # ------------------------------------------------------------
    # Perussuureet
    # ------------------------------------------------------------
    precs=np.copy(precs)/10 ## mm--> cm
    vuosikeskilampo = np.mean(temps, axis=0)

    kylmin = np.min(temps, axis=0)

    lampimin = np.max(temps, axis=0)

    vuosisade = np.sum(precs, axis=0)

    # Kuinka monta kuukautta lämpötila on vähintään 10 °C?
    lampimien_kuukausien_lkm = np.sum(
        temps >= 10.0,
        axis=0
    )

    # ------------------------------------------------------------
    # A: TROPIIKKI
    # ------------------------------------------------------------

    A = kylmin >= 18.0

    # ------------------------------------------------------------
    # E: POLAARISET
    # ------------------------------------------------------------

    EF = lampimin < 0.0

    ET = (
        (lampimin >= 0.0)
        &
        (lampimin < 10.0)
    )

    # ------------------------------------------------------------
    # B: KUIVAT ILMASTOT
    #
    # Köppenin kuivuusraja.
    #
    # Koska funktio ei tiedä sijaintia, käytetään lämpötilan
    # perusteella määritettyä lämpimän puoliskon sadantaa.
    # ------------------------------------------------------------

    # Selvitetään jokaiselle ruudulle kuuden kuukauden
    # lämpimin puolisko.
    #
    # Ensin lasketaan 6 kuukauden liukuva sademäärä.
    # Tämä käsittelee pohjoisen ja eteläisen pallonpuoliskon
    # vuodenaikojen vaihtelun paremmin kuin kiinteä 0:6 jako.

    kuuden_kk_sade = np.zeros(
        (12, height, width)
    )

    for i in range(12):
        kuukaudet = [
            (i + j) % 12
            for j in range(6)
        ]

        kuuden_kk_sade[i] = np.sum(
            precs[kuukaudet],
            axis=0
        )

    # Suurin kuuden kuukauden sadanta = sateisin puolisko.
    sateisin_puolisko = np.max(
        kuuden_kk_sade,
        axis=0
    )

    sateisin_puolisko_prosentti = (
        sateisin_puolisko
        / np.maximum(vuosisade, 1e-9)
    ) * 100.0

    sadanta_raja = np.where(
        sateisin_puolisko_prosentti >= 70.0,
        2.0 * (vuosikeskilampo + 14.0),
        2.0 * (vuosikeskilampo + 7.0)
    )

    B = vuosisade < sadanta_raja

    # ------------------------------------------------------------
    # B: AAVIKKO / ARO
    # ------------------------------------------------------------

    aavikko = (
        vuosisade
        < sadanta_raja * 0.5
    )

    aro = (
        B
        & ~aavikko
    )

    BWh = (
        aavikko
        & (vuosikeskilampo >= 18.0)
    )

    BWk = (
        aavikko
        & (vuosikeskilampo < 18.0)
    )

    BSh = (
        aro
        & (vuosikeskilampo >= 18.0)
    )

    BSk = (
        aro
        & (vuosikeskilampo < 18.0)
    )

    # ------------------------------------------------------------
    # A: Af / Am / Aw
    # ------------------------------------------------------------

    kuivin_A = np.min(
        precs,
        axis=0
    )

    Af = (
        A
        & (kuivin_A >= 60.0)
    )

    Am = (
        A
        & ~Af
        & (
            kuivin_A
            >= (100.0 - vuosisade / 25.0)
        )
    )

    Aw = (
        A
        & ~Af
        & ~Am
    )

    # ------------------------------------------------------------
    # C
    # ------------------------------------------------------------

    C = (
        ~A
        & ~B
        & ~EF
        & ~ET
        & (kylmin > 0.0)
        & (kylmin < 18.0)
        & (lampimin >= 10.0)
    )

    # ------------------------------------------------------------
    # D
    # ------------------------------------------------------------

    D = (
        ~A
        & ~B
        & ~EF
        & ~ET
        & (kylmin <= 0.0)
        & (lampimin >= 10.0)
    )

    # ------------------------------------------------------------
    # Kesä / talvi
    #
    # Tässä käytetään lämpötilaa eikä oleteta,
    # että kuukaudet 0-5 ovat kesä.
    #
    # Kullekin ruudulle:
    # - kuusi lämpimintä kuukautta = kesä
    # - kuusi kylmintä kuukautta = talvi
    # ------------------------------------------------------------

    jarjestys = np.argsort(
        temps,
        axis=0
    )

    kylmat_6 = np.take_along_axis(
        precs,
        jarjestys[:6],
        axis=0
    )

    lampimat_6 = np.take_along_axis(
        precs,
        jarjestys[6:],
        axis=0
    )

    kesan_sade = np.min(
        lampimat_6,
        axis=0
    )

    talven_sade = np.min(
        kylmat_6,
        axis=0
    )

    kesan_sade_summa = np.sum(
        lampimat_6,
        axis=0
    )

    talven_sade_summa = np.sum(
        kylmat_6,
        axis=0
    )

    # ------------------------------------------------------------
    # C / D: a, b, c, d
    # ------------------------------------------------------------

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
        kylmin <= -38.0
    )

    # ------------------------------------------------------------
    # C / D: s, w, f
    # ------------------------------------------------------------

    kuiva_kesa = (
        kesan_sade < 40.0
    )

    kuiva_talvi = (
        talven_sade < 40.0
    )

    s = (
        kuiva_kesa
        &
        (
            talven_sade_summa
            > kesan_sade_summa * 3.0
        )
    )

    w = (
        kuiva_talvi
        &
        (
            kesan_sade_summa
            > talven_sade_summa * 10.0
        )
    )

    f = ~s & ~w

    # ------------------------------------------------------------
    # C-luokat
    # ------------------------------------------------------------

    Cfa = C & f & kirjain2_a
    Cfb = C & f & kirjain2_b
    Cfc = C & f & kirjain2_c & ~kirjain2_b

    Csa = C & s & kirjain2_a
    Csb = C & s & kirjain2_b
    Csc = C & s & kirjain2_c & ~kirjain2_b

    Cwa = C & w & kirjain2_a
    Cwb = C & w & kirjain2_b
    Cwc = C & w & kirjain2_c & ~kirjain2_b

    # ------------------------------------------------------------
    # D-luokat
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

    # ------------------------------------------------------------
    # Kirjoitetaan luokat kartalle
    # ------------------------------------------------------------

    luokat = {
        "Af": Af,
        "Am": Am,
        "Aw": Aw,

        "BWh": BWh,
        "BWk": BWk,
        "BSh": BSh,
        "BSk": BSk,

        "Cfa": Cfa,
        "Cfb": Cfb,
        "Cfc": Cfc,
        "Csa": Csa,
        "Csb": Csb,
        "Csc": Csc,
        "Cwa": Cwa,
        "Cwb": Cwb,
        "Cwc": Cwc,

        "Dfa": Dfa,
        "Dfb": Dfb,
        "Dfc": Dfc,
        "Dfd": Dfd,
        "Dsa": Dsa,
        "Dsb": Dsb,
        "Dsc": Dsc,
        "Dsd": Dsd,
        "Dwa": Dwa,
        "Dwb": Dwb,
        "Dwc": Dwc,
        "Dwd": Dwd,

        "ET": ET,
        "EF": EF,
    }

    for luokka, maski in luokat.items():
        koppen[maski] = luokka

    return koppen




def piirra_koppen(koppen, landmask):
    """
    Piirtää Köppen–Geiger-ilmastoluokituksen karttana.

    Parametrit
    ----------
    koppen : 2D numpy-taulukko
        luokittele_koppen()-funktion palauttama taulukko.
    """
    landmask2=np.copy(landmask)
    landmask2=np.where(landmask2==0,np.nan,landmask2)

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
        numero*landmask2,
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

    # Jos lat on laskeva (np.diff(lat)[0] < 0), käännetään se ja siihen liittyvä data
    if lat[0] > lat[-1]:
        lat2 = lat[::-1]      # Käännetään leveysasteet nouseviksi
        U2 = U[::-1, :]       # Käännetään U-matriisin rivit (Y-akseli)
        V2 = V[::-1, :]       # Käännetään V-matriisin rivit (Y-akseli)
        X2 = X[::-1, :]       # Käännetään U-matriisin rivit (Y-akseli)
        Y2 = Y[::-1, :]       # Käännetään V-matriisin rivit (Y-akseli)
        if isinstance(color, np.ndarray) and color.ndim == 2:
            color2 = color[::-1, :] # Käännetään myös värimatriisi, jos käytössä on 2D


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
    #print(lon)
    #print(lat)
    q2 = ax.streamplot(
        X,
        Y2,
        U2,
        V2,
        color=color,
        linewidth=2,
        arrowsize=2
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


def load_raster(input_path, new_width, new_height, menetelma=Resampling.bilinear):
    """
    Lataa rasterin, muuttaa sen koon haluttuun leveyteen ja korkeuteen,
    päivittää georeferoinnin ja tallentaa uuden rasterin.
    """
    with rasterio.open(input_path) as src:
        # Lasketaan uusi transformaatio, jotta karttakoordinaatit säilyvät oikeina
        transform = src.transform * src.transform.scale(
            (src.width / new_width),
            (src.height / new_height)
        )
        
        # Kopioidaan ja päivitetään metadataprofiili
        profiili = src.profile.copy()
        profiili.update({
            'height': new_height,
            'width': new_width,
            'transform': transform
        })
        
        # Kirjoitetaan uusi rasteritiedosto
        for i in range(1, src.count + 1):
            data = src.read(
                    i,
                    out_shape=(new_height, new_width),
                    resampling=menetelma
               )
                
    print(f"Koko muutettu: {src.width}x{src.height} -> {new_width}x{new_height}")
    return(data)

def load_many_rasters_numpy_taulukkoon(maara,tiedostopolut, new_width, new_height, menetelma=Resampling.bilinear):
    """
    Lataa annetut rasteritiedostot, muuttaa niiden koon lennosta ja
    yhdistää ne yhdeksi NumPy-taulukoksi, jonka muoto (shape) on [N, height, width].
    """
    # Varmistetaan, että tiedostoja on oikea määrä (esim. 12 kuukautta)
    if len(tiedostopolut) != maara:
        print(f"Huomautus: Tiedostolistan pituus on {len(tiedostopolut)}, ei maara.")
        
    lista_taulukoista = []
    
    for polku in tiedostopolut:
        with rasterio.open(polku) as src:
            # Luetaan rasterin ensimmäinen kanava (band 1) ja skaalataan se lennosta.
            # out_shape ottaa muodon (korkeus, leveys).
            data = src.read(
                1, 
                out_shape=(new_height, new_width), 
                resampling=menetelma
            )
            lista_taulukoista.append(data)
            
    # Pinotaan listan taulukot uudeksi akseliksi (N, height, width) -> [12, height, width]
    valmis_taulukko = np.stack(lista_taulukoista, axis=0)
    
    return valmis_taulukko

 

def parse_args():
    ## command line args
    #generate_dem=1 ## generate ifractal dem type 1, if will		
    ##will_load_image=False ## input gray scale image for dem
    #will_load_rasters=False ## input netcdf or geotiff dem, metere height and depth 
    #input_dem_path="../data/etopo/etopo7200.tif"
    #input_imagename='./indata1/helliconia2.png'
    global generate_dem, will_load_image, will_load_rasters,input_imagename,input_dem_path 
    global seed1, width, height, land_fraction, delta_height
    global add_water_in_oceans 

    parser = argparse.ArgumentParser(description="Params")
    parser.add_argument(
    "--seed", type=int, default=seed1, help="Seed from rangom generator"
    )
    parser.add_argument(
    "--width", type=int, default=width, help="Map width px"
    )
    parser.add_argument(
    "--height", type=int, default=height, help="Map height px"
    )
    parser.add_argument(
    "--delta_height", type=float, default=delta_height, help="peak-abyss: Highest point -owest point under sea "
    )
    parser.add_argument(
    "--land_fraction", type=float, default=land_fraction, help="Lowest depth meters"
    )
    parser.add_argument(
    "--water_in_oceans", type=float, default=add_water_in_oceans, help="Lowest depth meters"
    )
    parser.add_argument(
    "--image", default="input.png", help="Input heighfield image" 
    )
    parser.add_argument(
    "--dem",  default="input.tif", help="Input dem im meters" 
    )
    
    args = parser.parse_args()
    

    seed1=args.seed
    width=args.width
    height=args.height
    delta_height=args.delta_height
    land_fraction=args.land_fraction
    add_water_in_oceans=args.water_in_oceans
    #if args.dem is not None:
    #    print(f"Input dem raster is : {args.dem}")
    #    generate_dem=0
    #    will_load_image=False
    #    will_load_rasters=True
    #    #input_imagename=args.image
    #    input_dem_path=args.dem
    # Koodissa tarkistetaan, onko muuttujassa jotain muuta kuin None
    #if args.image is not None:
    #    print(f"Input heightfield image is : {args.image}")
    #    generate_dem=0
    #    will_load_image=True
    #    will_load_rasters=False
    #     input_imagename=args.image
    #     #input_dem_path=

              
    print("Params from command line")
    print(f"Seed is: {seed1}")
    print(f"Width is: {width}")
    print(f"Height is: {height}")
    print(f"Landrfac is: {land_fraction}")
    print(f"delta-height is: {delta_height}")    
    return(0)



def etsi_suurin_asuinkelpoinen_manner(manner_jarjestys, habitability, raja=0.5):
    """
    Etsii mantereen, jolla on eniten asuinkelpoista pinta-alaa.

    Parametrit
    ----------
    manner_jarjestys : 2D numpy array
        Rasteri, jossa jokaisella maasolulla on mantereen ID.
        Esim. 1, 2, 3, ...

    habitability : 2D numpy array
        Asuinkelpoisuusindeksi.

    raja : float
        Alin habitability-arvo, joka lasketaan asuinkelpoiseksi.

    Palauttaa
    ----------
    suurin_manner : int
        Eniten asuinkelpoista pinta-alaa sisältävän mantereen ID.

    suurin_manner_mask : 2D numpy array
        Maski, jossa valitun mantereen solut ovat 1 ja muut 0.

    asuinkelpoinen_ala : int
        Asuinkelpoisten solujen määrä.
    """

    suurin_manner = None
    suurin_ala = 0

    for manner_id in np.unique(manner_jarjestys):

        # 0 = meri / ei manner
        if manner_id <= 0:
            continue

        manner_mask = manner_jarjestys == manner_id

        asuinkelpoinen_mask = (
            manner_mask &
            (habitability >= raja)
        )

        ala = np.sum(asuinkelpoinen_mask)

        if ala > suurin_ala:
            suurin_ala = ala
            suurin_manner = manner_id

    suurin_manner_mask = np.where(
        manner_jarjestys == suurin_manner,
        1,
        0
    )

    return suurin_manner, suurin_manner_mask, suurin_ala


    
def laske_planeetan_imasto_parametrit(dem, relief, landmask):
    print("")
    print("Star")
    print("mass_msun luminosity teff")
    print(star_mass, star_luminosity, star_teff)
    print("Planet")
    print("dist_au orb_period_yr rotation_period_h")
    print(distance_au, orbital_period_years,rotation_period_days*24) 
    print("S1  ecc tilt  mvelp")
    print(S1, ecc, tilt, mvelp)
    print("Mass_me radius_re")
    print(planet_mass_me, planet_radius_re)   
    print("atmosphere_pressure co2")
    print(planet_atmosphere_pressure, planet_atmos_co2_ppm) 
    
    np.random.seed(seed=seed1)
 
    tulokset = laske_planeetan_lampotila(
    ecc=ecc,
    tilt=tilt,
    mvelp=mvelp,
    S1=S1,

    atmos_co2_ppm=planet_atmos_co2_ppm,

    planet_atmosphere_pressure=planet_atmosphere_pressure,

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
    #temp_diff=70*math.sqrt(rotation_period_days)*math.sqrt(planet_radius_re)/planet_atmosphere_pressure
    ## orbital_period_years?
    result_landsea =calculate_land_sea_percentage(landmask)
    #print(result_landsea)   
    mannerosuus=result_landsea['maa_prosentti']/100
    #print("mannerosuus")
    #print(mannerosuus)
    manner_efekti=1.0    
    manner_efekti=laske_lammonkuljetus(mannerosuus)

    temp_diff=110*manner_efekti*math.sqrt(rotation_period_days)*math.sqrt(planet_radius_re)/planet_atmosphere_pressure
    temp_diff_daily=10*manner_efekti*math.sqrt(rotation_period_days)*math.sqrt(planet_radius_re)/planet_atmosphere_pressure
    ## sqrt(orbital_period_years)
    temp_diff_annual=10*(1+math.pow(ecc, 0.5))*math.sqrt(orbital_period_years)*math.sqrt(rotation_period_days)*manner_efekti*math.sqrt(planet_radius_re)/planet_atmosphere_pressure
    temp_dev=temp_diff/2
    polar_temp=mean_temp-temp_dev
    temp_diff=mean_temp+temp_dev
    print("mean_temp, temp_diff")
    print(mean_temp, temp_diff)
    #quit(-1)
 

    painom_matriisi = np.repeat(painot_1d[:, np.newaxis], width, axis=1)
    	

    #plt.imshow(distmountains)
    #plt.show()
    #quit(-1)

    tulokset = calculate_land_sea_percentage(landmask)
    print(f"Land : {tulokset['maa_prosentti']}%")
    print(f"Sea  : {tulokset['meri_prosentti']}%")
    stats = calculate_dem_statistics(dem)    
    print(f"Deepest sea (min)  : {stats['min_sea_depth_m']} m")
    print(f"Highest peak (maxi): {stats['max_land_height_m']} m")
    print(f"Ocean mean depth   : {stats['mean_sea_depth_m']} m")
    print(f"Laind mean height  : {stats['mean_land_height_m']} m")

    #temps12, winds12, precips12 = laske_perus_ilmasto(relief, tilt=tilt, ecc=ecc, P=1.0)
    kiertosolut=int(3/rotation_period_days)
    temps12, winds12_x, winds12_y, winds12_z, precips12 = (
    laske_perus_ilmasto(
        relief, t_mean=mean_temp, delta_t=temp_dev,
        tilt=tilt,
        ecc=ecc,
        mvelp=mvelp,
        P=orbital_period_years,
        kiertosolut=kiertosolut
    )
    )


    temp_annual=np.mean(temps12, axis=0)
    precip_annual=np.sum(precips12, axis=0)
    maski = np.isnan(temp_annual)
    alueen_keski_lampotila = np.average(temp_annual, weights=painom_matriisi)
    moisture_coeff=tulokset['meri_prosentti']/100.0
    #moisture_temperature=
    delta_T = (alueen_keski_lampotila - 15.0) ## earth now 15 c
    sademäärä_kerroin = 0.02
    uusi_sadek = (1 + sademäärä_kerroin * delta_T)
    uusi_sadek = max(0.0, uusi_sadek)
    kosteus_kerroin_kaatosade = 0.07
   
    uusi_kosteus_kapasiteetti_kaatosade = 1* math.exp(kosteus_kerroin_kaatosade * delta_T)
    #print(moisture_coeff, uusi_sadek)
    moisture_coeff=moisture_coeff*uusi_sadek*3.5
    #print(moisture_coeff)
    ## earth now 15 c, 990 mm
    #precip_annual, tuulet = laske_sademaara_soluilla(relief, distsea, sealevel=0.5, moisture_coeff=moisture_coeff)
    
    
    
    precips12=precips12*moisture_coeff
    precip_annual=precip_annual*moisture_coeff
    
    
    bios = calculate_bioclim(temps12, precips12)
    
    alueen_keski_sademaara  = np.average(precip_annual, weights=painom_matriisi)
    alueen_keski_lampotila  = np.average(temp_annual, weights=painom_matriisi)
    #print(f"Planeetan painotettu keskilämpötila : {alueen_keski_lampotila:.2f} °C")
    #print(f"Planeetan painotettu keskisademäärä : {alueen_keski_sademaara:.2f} mm")

    
    return (temps12, precips12, winds12_x, winds12_y, winds12_z)






def max_likelihood_raster(
    rasteri_lista,
    hae_lista,
    sigma_lista,
    weight_lista,
    bounds=(-180.0, -90.0, 180.0, 90.0),
    top_n=100,
):
    """
    Etsii rastereista pikselit, joiden arvot ovat todennäköisimpiä
    suhteessa annettuun tavoitevektoriin.

    Jokaiselle muuttujalle voidaan antaa paino välillä 0...1.
    Paino vaikuttaa kyseisen muuttujan log-likelihood-kontribuutioon.

    Parameters
    ----------
    rasteri_lista : list[np.ndarray]
        Esim. [lampotila, sademaara].
        Kaikkien rasterien pitää olla saman kokoisia.

    hae_lista : list[float]
        Haettavat arvot, esim. [20, 1000].

    sigma_lista : list[float]
        Keskihajonnat vastaaville muuttujille, esim. [5, 250].

    weight_lista : list[float]
        Muuttujien painot välillä 0...1.
        1.0 = täysi vaikutus
        0.5 = puolet vaikutuksesta
        0.0 = muuttuja ei vaikuta likelihoodiin

    bounds : tuple
        (xmin, ymin, xmax, ymax), oletuksena koko maapallo.

    top_n : int
        Kuinka monta parasta pikseliä palautetaan.

    Returns
    -------
    probability_surface : np.ndarray
        Normalisoitu likelihood-pinta väliltä 0...1.

    max_likelihood_pixels : np.ndarray
        Indeksit (row, col) parhaisiin pikseleihin.

    max_likelihood_coords : np.ndarray
        Koordinaatit muodossa [[lon, lat], ...].
    """

    # ---------------------------------------------------------
    # Validoinnit
    # ---------------------------------------------------------

    if not (
        len(rasteri_lista)
        == len(hae_lista)
        == len(sigma_lista)
        == len(weight_lista)
    ):
        raise ValueError(
            "rasteri_lista, hae_lista, sigma_lista ja "
            "weight_lista pitää olla saman pituisia."
        )

    if len(rasteri_lista) == 0:
        raise ValueError("rasteri_lista ei voi olla tyhjä.")

    if any(s <= 0 for s in sigma_lista):
        raise ValueError("Keskihajontojen pitää olla > 0.")

    if any(w < 0 or w > 1 for w in weight_lista):
        raise ValueError("Painojen pitää olla välillä 0...1.")

    shape = rasteri_lista[0].shape

    if any(r.shape != shape for r in rasteri_lista):
        raise ValueError("Kaikkien rasterien pitää olla saman kokoisia.")

    if top_n < 1:
        raise ValueError("top_n pitää olla vähintään 1.")

    # ---------------------------------------------------------
    # Log-likelihood jokaiselle pikselille
    # ---------------------------------------------------------

    log_likelihood = np.zeros(shape, dtype=np.float64)

    for raster, target, sigma, weight in zip(
        rasteri_lista,
        hae_lista,
        sigma_lista,
        weight_lista,
    ):
        raster = np.asarray(raster, dtype=np.float64)

        valid = np.isfinite(raster)

        # Jos rasterin arvo puuttuu, koko pikseli invalid
        log_likelihood[~valid] = np.nan

        if weight == 0:
            # Muuttuja ei vaikuta likelihoodiin,
            # mutta sen puuttuva arvo tekee pikselistä edelleen invalidin.
            continue

        z = (raster - target) / sigma

        contribution = (
            -0.5 * z[valid] ** 2
            - np.log(sigma * np.sqrt(2.0 * np.pi))
        )

        # Painotettu log-likelihood
        log_likelihood[valid] += weight * contribution
        # Painotettu log-likelihood kertoma
        #log_likelihood[valid] *= weight * contribution
    # ---------------------------------------------------------
    # Muutetaan likelihoodiksi ja normalisoidaan 0...1
    # ---------------------------------------------------------

    valid = np.isfinite(log_likelihood)

    probability_surface = np.full(shape, np.nan)

    if not np.any(valid):
        raise ValueError("Rastereissa ei ole yhtään validia pikseliä.")

    max_log = np.max(log_likelihood[valid])

    # Numerisesti stabiili:
    # paras pikseli = 1.0
    probability_surface[valid] = np.exp(
        log_likelihood[valid] - max_log
    )

    # ---------------------------------------------------------
    # Parhaat pikselit
    # ---------------------------------------------------------

    valid_indices = np.flatnonzero(valid)

    n = min(top_n, len(valid_indices))

    # np.argpartition on paljon nopeampi kuin täysi sorttaus
    best_flat = valid_indices[
        np.argpartition(
            log_likelihood.flat[valid_indices],
            -n
        )[-n:]
    ]

    # Järjestetään parhaasta huonoimpaan
    best_flat = best_flat[
        np.argsort(
            log_likelihood.flat[best_flat]
        )[::-1]
    ]

    max_likelihood_pixels = np.column_stack(
        np.unravel_index(best_flat, shape)
    )

    # ---------------------------------------------------------
    # Pixel -> lon/lat
    # ---------------------------------------------------------

    xmin, ymin, xmax, ymax = bounds

    height, width = shape

    rows = max_likelihood_pixels[:, 0]
    cols = max_likelihood_pixels[:, 1]

    # Pikselin keskipisteen koordinaatit
    lons = xmin + (cols + 0.5) * (xmax - xmin) / width
    lats = ymax - (rows + 0.5) * (ymax - ymin) / height

    max_likelihood_coords = np.column_stack(
        (lons, lats)
    )

    return (
        probability_surface,
        max_likelihood_pixels,
        max_likelihood_coords,
    )


def calculate_culture_static_00():
    #manner_osuudet, manner_jarjestys=laske_mantereet(relief, planet_radius=planet_radius)
    #suurin_manner=np.copy(manner_jarjestys)
    #suurin_manner=np.where(suurin_manner==1,1,0)
    manner_osuudet, manner_jarjestys = laske_mantereet(
    relief,
    planet_radius=planet_radius
    )
    habitability = laske_human_habitability_index_annual(
    temp_annual,
    precip_annual
    )
    suurin_manner, suurin_manner_mask, ala = (
    etsi_suurin_asuinkelpoinen_manner(
        manner_jarjestys,
        habitability,
        raja=0.5
    )
    )

    print("Suurin asuinkelpoinen manner #:", suurin_manner)
    print("Asuinkelpoista pinta-alaa :", ala)

    #plt.imshow(suurin_manner_mask)
    #plt.show()


    #quit(-1)
    suitability_rain_agriculture = lbk_sadeviljely_soveltuvuus(relief,precip_annual, temp_annual)*landmask
    agriculture, pastoralism, nomadism = subsistence_suitability_threepars(relief,precip_annual, temp_annual )
 
    results1=subsistence_suitability(
    elevation=relief,
    precipitation=precip_annual,
    temperature=temp_annual,
    npp=npp,
    twi=twi,
    distance_to_water=distwater,
    tpi=tpi, biome=biomi_kartta
    )

    agriculture = results1["agriculture"]
    pastoralism = results1["pastoralism"]
    nomadism = results1["nomadism"]
    hunter_gatherer = results1["hunter_gatherer"]


def calculate_culture_static():
    #manner_osuudet, manner_jarjestys=laske_mantereet(relief, planet_radius=planet_radius)
    #suurin_manner=np.copy(manner_jarjestys)
    #suurin_manner=np.where(suurin_manner==1,1,0)
    #pix_areas

    curvature, profile, plan = dem_curvature(
    relief,
    cellsize_x=2.0,
    cellsize_y=2.0
    )
    
    manner_osuudet, manner_jarjestys = laske_mantereet(
    relief,
    planet_radius=planet_radius
    )
    habitability = normalize(laske_human_habitability_index_annual(
    temp_annual,
    precip_annual
    ))
    
    suurin_manner, suurin_manner_mask, ala = (
    etsi_suurin_asuinkelpoinen_manner(
        manner_jarjestys,
        habitability,
        raja=0.5
    )
    )

    suurin_manner_mask=np.where(suurin_manner_mask==0, 0, 1)
    #ice age, sealevel minus 120
    relief_minus120=np.copy(dem)
    relief_minus120=relief_minus120+120
    relief_minus120=np.where(relief_minus120>0, relief_minus120,0)   
    #3 area whete human can spread without canoes etc due to ice ages
    jmanner_osuudet, jmanner_jarjestys = laske_mantereet(
    relief_minus120,
    planet_radius=planet_radius
    )

    jsuurin_manner, jsuurin_manner_mask, jala = (
    etsi_suurin_asuinkelpoinen_manner(
        jmanner_jarjestys,
        habitability,
        raja=0.5
    )
    )

    jsuurin_manner_mask=np.where(jsuurin_manner_mask==0, 0, 1)
    jsuurin_manner_mask=jsuurin_manner_mask*landmask ## lift sea level after ice age



    print("Suurin asuinkelpoinen manner #:", suurin_manner)
    print("Asuinkelpoista pinta-alaa :", ala)
    distance_to_mountains=laske_etaisyys_korkeuteen_km(relief, 1500, planet_radius, tolerance=10)
    mountain_eff_uruk=np.exp(-distance_to_mountains/200)
    #plt.imshow(distance_to_mountains*landmask) 
    #plt.show()  
	
    nearwater=landmask*normalize(np.exp(-normalize(np.copy(distwater))))
    #nearwater=np.where(nearwater==0, np.nan, nearwater)
    #plt.imshow(nearwater)

    #plt.imshow(suurin_manner_mask*habitability)
    #plt.imshow(suurin_manner_mask*normalize(twi))
    #plt.imshow(suurin_manner_mask)
    #plt.show()

    #bio1=bios['BIO1']   ## mean temp annof year deg c  
    #bio12=bios['BIO12'] ## precip sum of year months
    #bio4=bios['BIO4'] ## temperature seasonality stdev*100   
    #bio5=bios['BIO5'] ## max warmest temp    
    #bio6=bios['BIO6'] ## min coldest temp
    #bio7=bios['BIO7'] ## bio5-bio6 temperature annual range
    #bio13=bios['BIO13'] ## wettest monthprecip
    #bio14=bios['BIO14'] ## diriest mont precip      
    #bio15=bios['BIO15'] ## precip seasonality   
    #bio18=bios['BIO18'] ## warmest quarter precip    
    #bio19=bios['BIO19'] ## coldest quarter precip

    intelligence_list=[landmask*suurin_manner_mask, bio5, bio12, relief]
    intelligence_surface, intelligence_pixels, intelligence_coords=max_likelihood_raster(intelligence_list,
    [1,27,550,1450],
    [0.001,5,20,100],
    [1,1,1,1],
    #bounds=(-180.0, -90.0, 180.0, 90.0),
    top_n=10)
    #plt.imshow(intelligence_surface)
    #plt.show()
       
    agriculture_list=[landmask*suurin_manner_mask, bio1*suurin_manner_mask, bio12*suurin_manner_mask, relief*suurin_manner_mask]
    agriculture_surface, agriculture_pixels, agriculture_coords=max_likelihood_raster(agriculture_list,
    [1,18.5,450,750],
    [0.001,5,50,250],
    [1,1,0.8,0.2],
    bounds=(-180.0, -90.0, 180.0, 90.0),
    top_n=10)
    print(" Spread agriculture")
    aalku_x=agriculture_pixels[0,1]
    aalku_y=agriculture_pixels[0,0]
    aalku_lon=agriculture_coords[0,1]
    aalku_lat=agriculture_coords[0,0]
    print(" First agriculture location")
    print(aalku_x, aalku_y)
    print(aalku_lon, aalku_lat)
    agriculture_index,pastoral_index,suitability,kulttuuri,leviamisaika,vaesto,carrying_capacity=levia(aalku_x, aalku_y,relief,bio1,bio12,ajo_vuodet=5000)
    print(leviamisaika)
    #plt.imshow(agriculture_index)
    #plt.imshow(leviamisaika)
    #plt.imshow(vaesto)
    #plt.show()    
        
    ag_lon=agriculture_coords[0,1]
    ag_lat=agriculture_coords[0,0]
    coords_area = [-180, 180, -90, 90]
    distance_to_agriculture_origin = distance_to_point(ag_lon, ag_lat, height, width, coords_area, planet_radius)
    agriculture_birthplace_effect=np.exp(-distance_to_agriculture_origin/200)

    #plt.imshow(distance_to_agriculture_origin )
    #plt.show()    

    uruk_list=[nearwater, mountain_eff_uruk,agriculture_birthplace_effect, bio1*suurin_manner_mask, bio12*suurin_manner_mask]
    uruk_surface, uruk_pixels, uruk_coords=max_likelihood_raster(uruk_list,
    [1,1, 2, 24,125],
    [0.05,0.5,2, 1.5,25],
    [1,1,1,1,1.0],
    bounds=(-180.0, -90.0, 180.0, 90.0),
    top_n=10) 

    salku_x=uruk_pixels[0,1]
    salku_y=uruk_pixels[0,0]
    salku_lon=uruk_coords[0,1]
    salku_lat=uruk_coords[0,0]
    distance_to_civilization_origin = distance_to_point(salku_lon, salku_lat, height, width, coords_area, planet_radius)
    civilization_birthplace_effect=np.exp(-distance_to_civilization_origin/200)


    print(" First civilization location")
    print(salku_x, salku_y)
    print(salku_lon, salku_lat)
    
    roma_list=[relief,
    bio6*suurin_manner_mask, bio12*suurin_manner_mask]
    roma_surface, roma_pixels, roma_coords=max_likelihood_raster(roma_list,
    [25,        7.5,  775],
    [4,     1.5, 21],
    [1, 1,1],
    bounds=(-180.0, -90.0, 180.0, 90.0),
    top_n=10) 

    ralku_x=roma_pixels[0,1]
    ralku_y=roma_pixels[0,0]
    ralku_lon=roma_coords[0,1]
    ralku_lat=roma_coords[0,0]


    print(" First civilization location")
    print(salku_x, salku_y)
    print(salku_lon, salku_lat)    

    agriculture_caps = calculate_carrying_capacity(
     temps12,
     precips12,
     population_in_pixels, ## hunter gatherers
     arable_fraction=None,
    )


    agri_suitability=agriculture_caps["climate_suitability"]
    agri_density=agriculture_caps["population_density"]
    agri_capacity=agriculture_caps["carrying_capacity"]
    agri_capacity_ratio=agriculture_caps["capacity_ratio"]

    capacity_world=np.sum(agri_capacity*landmask)
    print(" WORLD agriculture K millions", int(capacity_world/1e6))
 
 
    agri2_results = agricultural_potential(
    temps12,
    precips12,
    relief=relief,
    rivers1=rivers1,
    population= population_in_pixels,
    )



    agri3_results = agricultural_carrying_capacity(
    temps12=temps12,
    precips12=precips12,
    relief=relief,
    rivers1=rivers1,
    population=population_in_pixels,

    # Alustava oletus:
    arable_fraction=0.5,

    # 30 % potentiaalisesta jokiveden
    # hyödyntämisestä.
    irrigation_technology=0.30,

    food_fraction=0.70,

    kcal_person_day=2500,
    )



    carrying_capacity = agri3_results[
    "carrying_capacity"
    ]

    population_density = agri3_results[
    "population_density"
    ]

    capacity_ratio = agri3_results[
    "capacity_ratio"
    ]

    best_crop = agri3_results[
    "best_crop"
    ]

    agricultural_score = agri3_results[
    "agricultural_score"
    ]
    # Esimerkiksi:
    #
    # agri2_results["agricultural_score"]
    best_crop=agri2_results["best_crop"]
    # agri2_results["potato_suitability"]
    maize_suitability=agri2_results["maize_suitability"]
    wheat_suitability=agri2_results["wheat_suitability"]
    # agri2_results["growing_months"]
    #plt.imshow(population_density*landmask) 
    #plt.imshow(wheat_suitability*landmask) 
    #plt.imshow(agri_capacity*landmask)
    #plt.plot()
    #plt.show()
    #quit(-1)
    
       
    im = plt.imshow(biomi_kartta, cmap=cmap_biomit, origin='upper', vmin=0, vmax=7, extent=[-180,180,-90,90])
    shad = plt.imshow(hillshade1*landmask, cmap="gray", origin='upper', extent=[-180,180,-90,90], alpha=0.3)
    cmap_rivers1 = ListedColormap(['#7f7fff', 'lightblue'])
    cmap_lakes1 = ListedColormap(['#5f5fff', 'lightblue'])
    plt.imshow(rivers1*noice, cmap=cmap_rivers1,origin='upper', extent=[-180,180,-90,90]) 
    #plt.imshow(lakes1*noice, cmap=cmap_lakes1,origin='upper', extent=[-180,180,-90,90])     
    #print(intelligence_pixels)
    #print(intelligence_coords)
    #print(intelligence_pixels[:,0])
    #plt.imshow(biomi_kartta)
    #plt.imshow(intelligence_surface)
    #plt.contour(landmask, levels=[0,0.5,1], color="white", alpha=0.5)

    #plt.contour(jsuurin_manner_mask, levels=[0,0.5,1], colors=["gray"], origin='upper', extent=[-180,180,-90,90],)

    plt.title("Planet biomes and culture sites", fontsize=18)
    plt.scatter(x=intelligence_coords[0,0],y=intelligence_coords[0,1], color="black", s=50)
    plt.text(intelligence_coords[0,0],intelligence_coords[0,1],"Origin", color="black", fontsize=16 ) 
    
    plt.scatter(x=intelligence_coords[0:,0],y=intelligence_coords[0:,1], color="black", s=5)
    plt.scatter(x=agriculture_coords[:,0],y=agriculture_coords[:,1], color="yellow", s=5) 
    plt.scatter(x=agriculture_coords[0,0],y=agriculture_coords[0,1], color="yellow", s=50) 
    plt.text(agriculture_coords[0,0],agriculture_coords[0,1],"Agriculture", color="yellow", fontsize=16 ) 


    plt.scatter(x=uruk_coords[0,0],y=uruk_coords[0,1], marker="s", color="red", s=50) 
    plt.scatter(x=uruk_coords[:,0],y=uruk_coords[:,1], marker="s", color="red", s=5) 
    plt.text(uruk_coords[0,0],uruk_coords[0,1],"Cities", color="red", fontsize=16 ) 
    
    plt.scatter(x=roma_coords[0,0],y=roma_coords[0,1], marker="s", edgecolors="k", linewidths=1,  color="green", s=60) 
    plt.scatter(x=roma_coords[:,0],y=roma_coords[:,1], marker="s",edgecolors="k", linewidths=1,  color="green", s=10) 
    plt.text(roma_coords[0,0],roma_coords[0,1],"Roma", color="#003f00", fontsize=16 ) 
    plt.show()
    
    #max_likelihood_pixels : np.ndarray
    #    Indeksit (row, col) parhaisiin pikseleihin.

    #max_likelihood_coords : np.ndarray
    #    Koordinaatit muodossa [[lon, lat], ...
        
    return(0)


#############################################
## Python main program, cilmate and map of planet
# --- Esimerkki käytöstä ---



if __name__ == "__main__":
	
    parse_args()

    if(will_load_rasters==True):
        print("Load dem raster")
        dem=load_raster(input_dem_path, width, height, menetelma=Resampling.bilinear)
        dem_min=np.min(dem)
        dem_max=np.max(dem)
        landmask=np.copy(dem)
        landmask=np.where(landmask<0,0,1) 
        relief=(np.copy(dem))*landmask
        seamask=np.copy(relief)
        seamask=np.where(seamask>0,0,1)
        relief2=np.copy(relief)
        relief2=np.where(relief2<=0,np.nan,relief2 )
        #plt.imshow(dem)
        #plt.show()
        #quit(-1)
        #indeksit = range(1, 13)
        #rasteri_lista = [f"data_kuukausi_{i:02d}.tif" for i in indeksit]
        #temps12 = lataa_rasterit_numpy_taulukkoon(rasteri_lista, leveys, korkeus)
        
    if(will_load_image==True):
        print("Load grayscale image")
        imagee, width, height=load_gray_image_and_normalize(input_imagename)
        imagee=normalize(imagee)
        dem, dem_min, dem_max, delta_dem, sealevel_from_min = \
        transform_dem(
        imagee,
        delta_height=delta_height,
        land_fraction=land_fraction, debug=True
        )
        landmask=np.copy(dem)
        landmask=np.where(landmask<1,0,1) 
        relief=(np.copy(dem))*landmask
        seamask=np.copy(relief)
        seamask=np.where(seamask>0,0,1)
        #relief = simulate_thermal_erosion(relief, iterations=1, c_repose=2.0, talus_rate=0.2)
        #relief = simulate_erosion(relief, num_droplets=10000)  #30000
        relief2=np.copy(relief)
        relief2=np.where(relief2<=0,np.nan,relief2 )
        #plt.imshow(landmask)
        #plt.imshow(relief)
        #plt.show()
        #quit(-1)
        
    if(generate_dem==1):
        print("Generate dem")
        imagee = generate_spherical_noise(width, height, scale=base_noisescale, octaves=16, seed=seed1)
        imagee2 = generate_spherical_noise(width, height, scale=base_noisescale, octaves=16, seed=seed1+3)
        imagee=normalize(imagee)
        imagee2=normalize(imagee)
        imagee2=(np.sin(imagee2* 1 * 2 * np.pi) + 1) / 2
        #imagee=sigmoid_dem(imagee, saatokerroin=0.5, keskiarvo=0.5) 
        #imagee=sigmoid_meri_manner_jakauma(imagee, säätökerroin=-2)
        #imagee=muotoile_maan_jakauma(imagee, sealevel=0.5)
        #imagee=normalize(imagee)
        imagee=spherical_noise_offset(imagee, move_amount_max=distort_coeff*height, scale=distort_noisescale, seed=seed1)
        imagee=normalize(imagee)
        imagee2=spherical_noise_offset(imagee2, move_amount_max=distort_coeff*height, scale=distort_noisescale, seed=seed1+1)
        imagee2=normalize(imagee2)
        #imagee=np.exp(imagee)/np.exp(1)
        #p=2.5
        #a=3.0
        #lam=0.3
        #power = ax ** imagee
        #exponential = (math.exp(a * ax) - 1.0) / (math.exp(a) - 1.0)
        #imagee = (1.0 - lam) * power + lam * exponential
        ##imagee=np.power(imagee, 2.5)
        #imagee=np.exp(imagee)          
        imagee=np.pow(imagee, 2.5)
        imagee2=np.exp(imagee2)
        imagee=normalize(imagee)
        imagee2=normalize(imagee2)
        imagee=imagee*0.5+0.5*imagee2       
        #dem, landmask=create_dem_from_array(imagee, sealevel, dem_min, dem_max)
        #plt.imshow(imagee)
        #plt.imshow(relief)
        #plt.show()        
        
        dem, dem_min, dem_max, delta_dem, sealevel_from_min = \
        transform_dem(
        imagee,
        delta_height=delta_height,
        land_fraction=land_fraction, debug=True
        )
        landmask=np.copy(dem)
        landmask=np.where(landmask<1,0,1) 
        relief=(np.copy(dem))*landmask
        seamask=np.copy(relief)
        seamask=np.where(seamask>0,0,1)
        #relief = simulate_thermal_erosion(relief, iterations=1, c_repose=2.0, talus_rate=0.2)
        #relief = simulate_erosion(relief, num_droplets=10000)  #30000
        relief2=np.copy(relief)
        relief2=np.where(relief2<=0,np.nan,relief2 )
        #plt.imshow(landmask)
        #plt.imshow(relief)
        #plt.show()
        #quit(-1)

    if(generate_dem==2):
        print("Generate dem")
        imagee = generate_spherical_noise(width, height, scale=base_noisescale, octaves=16, seed=seed1)
        imagee=normalize(imagee)
        
        imagee=spherical_noise_offset(imagee, move_amount_max=distort_coeff*height, scale=distort_noisescale, seed=seed1)
        imagee=normalize(imagee)
                
        dem0=imagee*delta_height
        
        #add_water_in_oceans=1
        #delta_height=10000
        dem, dem_min, dem_max, delta_dem, sealevel, \
        added_water_volume, target_water_volume = flood_planet_with_ocean(
        dem0,
        delta_height=delta_height,
        add_water_in_earth_ocean_units=add_water_in_oceans,
        planet_radius_km=planet_radius,)
        #print(np.shape(dem))
        #quit(-1)
        landmask=np.copy(dem)
        landmask=np.where(landmask<1,0,1) 
        relief=(np.copy(dem))*landmask
        seamask=np.copy(relief)
        seamask=np.where(seamask>0,0,1)
        #relief = simulate_thermal_erosion(relief, iterations=1, c_repose=2.0, talus_rate=0.2)
        #relief = simulate_erosion(relief, num_droplets=10000)  #30000
        relief2=np.copy(relief)
        relief2=np.where(relief2<=0,np.nan,relief2 )
        #plt.imshow(landmask)
        #plt.imshow(relief)
        #plt.imshow(dem)
        #plt.show()
        #quit(-1)

    if(generate_dem==3):
        print("Generate ocean/land dem")
        imagee = generate_spherical_noise(width, height, scale=base_noisescale, octaves=16, seed=seed1)
        imagee=normalize(imagee)
        
        imagee=spherical_noise_offset(imagee, move_amount_max=distort_coeff*height, scale=distort_noisescale, seed=seed1)
        imagee=normalize(imagee)
        dem=np.copy(imagee)*delta_height
        dem, dem_min, dem_max, delta_dem, sealevel_from_min = \
        transform_dem(
        imagee,
        delta_height=delta_height,
        land_fraction=land_fraction, debug=True
        )
        landmask=np.copy(dem)
        landmask=np.where(landmask<1,0,1)        
        plt.imshow(dem)
        plt.show()        
        
        #dem=hypsometric_terrain_00(
        #dem,
        #landmask,
        #peak_height=3000,
        #sea_level=0,
        #shelf_height=-200,
        #deep_height=-4500,
        # Kuinka monta korkeusaluetta maalle tehdään
        #land_levels=5,
        # Kuinka paljon alkuperäisestä reliefistä säilytetään
        #relief=1.0)
        #dem = make_hypsometric_dem_02(
        #dem,
        #landmask,
        #peak_height=5000,
        #continental_scale=200,
        #regional_scale=60,
        #local_scale=10,
        #continental_weight=0.60,
        #regional_weight=0.30,
        #local_weight=0.10,
        #relief_strength=0.25,
        #elevation_exponent=1.4,
        #shelf_height=-200,
        #deep_height=-4500,
        #)
        
        #new_dem = make_hypsometric_dem(
        #dem,
        #landmask,

        #peak_height=5000,

        #continental_scale=200,
        #regional_scale=60,
        #local_scale=10,

        #continental_weight=0.60,
        #regional_weight=0.30,
        #local_weight=0.10,

        #relief_strength=0.25,
        #elevation_exponent=1.4,

        #shelf_height=-200,
        #deep_height=-4500,
        #)
        dem_min=-4000
        dem_max=4000
        delta_height=8000
        dem = make_hypsometric_dem(
        dem,
        landmask,
        resolution=1000,
        peak_height=dem_max,
        shelf_depth=-200,
        deep_depth=dem_min,
        shelf_width=100,
        slope_width=200,
        continental_scale=150,
        regional_scale=50,
        local_scale=10,
        relief_strength=0.25,
        elevation_exponent=1.4,
        )

        landmask=np.copy(dem)
        landmask=np.where(landmask<1,0,1) 
        relief=(np.copy(dem))*landmask
        seamask=np.copy(relief)
        seamask=np.where(seamask>0,0,1)
        relief2=np.copy(relief)
        relief2=np.where(relief2<=0,np.nan,relief2 )
        #plt.imshow(dem)
        #plt.show()
        #quit(-1)		

    lats = np.linspace(90, -90, height)
    lons= np.linspace(-180, -180, width)
    painot_1d = np.cos(np.radians(lats))  
    painom_matriisi = np.repeat(painot_1d[:, np.newaxis], width, axis=1)
    pix_areas = laske_globaalit_pikselialat(height, width, planet_radius)
    sphere_area=4 * np.pi * planet_radius**2
    
    light_direction = [-0.0, 1, 0.0] 
    # 3. Lasketaan varjot
    #hillshade2 = laske_ray_trace_shadows(relief, light_direction)
    # Lasketaan emboss/hillshade (valo luoteesta, 45 asteen kulmassa)
    hillshade1 = laske_hillshade(relief, azimuth=315, angle_altitude=45)
    print(np.shape(dem))
    distsea=distance_to_sea(relief, planet_radius)
    distmountains=distance_to_someheight(relief, planet_radius, 1500)
    twi = calculate_twi(dem, cell_size=100.0)    
    tpi=calculate_tpi(dem, window_size=5)
    manner_osuudet, manner_koko_jarjestys = laske_mantereet(relief, planet_radius=planet_radius)
    ## base climate
    temps12, precips12, winds12_x, winds12_y, winds12_z= laske_planeetan_imasto_parametrit(dem, relief, landmask)
    #print (np.shape(temps12))
    #quit(-1)
    
    ## ocean currents
    (current_x,current_y,current_z,upwelling,deep_x,deep_y,deep_z,sea_temperature,cold_current_mask,warm_current_mask) = calculate_ocean_currents_12(
    dem=dem,
    wind_x=winds12_x,
    wind_y=winds12_y,
    wind_z=winds12_z,
    temperature=temps12,
    precipitation=precips12,
    planet_radius_km=planet_radius,
    rotation_period_hours=rotation_period_hours,
    rotation_direction=rotation_direction)
    
    #fig, ax = plot_ocean_currents(dem,current_x[kk], current_y[kk], current_z[kk],step=18,scale =0.01, color="white",title="Pintamerivirrat")

    winds12_speed=np.sqrt( winds12_x*winds12_x+ winds12_y*winds12_y+ winds12_z*winds12_z)
    #plt.imshow(winds12_speed[6])
    #plt.show()
    bios = calculate_bioclim(temps12, precips12)    
    #print(bios)
    kk=6
    #fig, ax = plot_ocean_currents(dem,winds12_x[kk], winds12_y[kk], winds12_z[kk],step=18,scale =0.1, color="white",title="Tuuli")
    bio18=bios['BIO18'] ## summer drought
    #bio18 = bios[f"BIO{18}"]
    #plt.imshow(bio18*landmask)
    #plt.show()
    #quit(-1)
    bio1=bios['BIO1']   ## mean temp annof year deg c  
    bio12=bios['BIO12'] ## precip sum of year months
    bio4=bios['BIO4'] ## temperature seasonality stdev*100   
    bio5=bios['BIO5'] ## max warmest temp    
    bio6=bios['BIO6'] ## min coldest temp
    bio7=bios['BIO7'] ## bio5-bio6 temperature annual range
    bio13=bios['BIO13'] ## wettest monthprecip
    bio14=bios['BIO14'] ## diriest mont precip      
    bio15=bios['BIO15'] ## precip seasonality   
    bio18=bios['BIO18'] ## warmest quarter precip    
    bio19=bios['BIO19'] ## coldest quarter precip
        
    temp_annual=bio1
    precip_annual=bio12   
   
    koppen = luokittele_koppen(temps12,precips12)   
    biomi_kartta = luo_biomikartta(relief, temp_annual, precip_annual, sealevel=0.5)
   # 3. Kutsutaan aluohjelmaa laskentaa varten
    holdridge_tulos = laske_holdridge_luokat(temp_annual, precip_annual, nodata_arvo=np.nan)



    pet = calculate_pet(
    temps12,
    relief,
    landmask,

    tilt=tilt,
    rotation_period_h=rotation_period_hours,

    orbital_period_days=orbital_period_days,
    eccentricity=ecc,
    mvelp=mvelp,

    stellar_flux=S1,

    days_in_month=np.full(12, orbital_period_days/ 12.0),

    axis=0
    )

    gdd_above5, gdd_below5 = calculate_gdd_above_and_below(
    temps12,
    t_base=5.0, orbital_period_in_days=orbital_period_days,
    rotation_period_in_hours=rotation_period_hours)
    npp = laske_npp_miami(temp_annual, precip_annual, nodata_arvo=np.nan)*landmask 
    fire_risk, components = calculate_fire_risk(
        temp=temps12,
        prec=precips12,
        wind=winds12_speed,
        npp=npp,
    )
    
 
     
    ice=np.copy(temp_annual)
    ice=np.where(temp_annual<-10,1,np.nan)
    noice=np.where(temp_annual<-10,np.nan,1)
    maan_albedo=laske_albedoluokat(landmask, temp_annual, precip_annual)
    merijaa=arvioi_merijaa(landmask, temp_annual, precip_annual)
    meren_albedo=np.copy(merijaa)
    meren_albedo=np.where(meren_albedo>0,0.6,0.06)*seamask
    albedo=np.copy(meren_albedo)
    albedo=np.where(albedo==0,maan_albedo, meren_albedo)


    
  

    kk=6
    #plt.imshow(sea_temperature[kk])
    #plt.show()   

    
    #plt.imshow(cold_current_mask[kk])
    #plt.imshow(upwelling[kk])
    #plt.imshow(deep_z[kk])
    #plt.show()
    
    

    #rivers0, accumulation0, flow_to0=calculate_rivers(relief,precip_annual, pet, 10,100000)
    #rivers0, accumulation0, flow_to0=calculate_rivers(relief,precip_annual, pet, 100,10000)
    #lakes1, lake_depths1, lake_ids1 = calculate_lakes(relief,precip_annual,pet,flow_to0,accumulation0,cell_size=1000,min_inflow=1000000)

    #result = calculate_hydrology_00(
    #relief,
    #precip_annual,
    #pet,
    #cell_size=1000,
    #river_threshold=1_000_000,
    #min_lake_inflow=500_000,
    #min_lake_depth=1.5,
    #runoff_coefficient=0.65,
    #lake_strength=1.0,
    #)
    hydroresult = calculate_hydrology(
    relief=relief,
    precip_annual=precip_annual,
    pet=pet,
    cell_size=1000,

    #river_threshold=1_000_000,

    #min_lake_inflow=500_000,
    river_threshold=10000000,

    min_lake_inflow=5000000,
    min_lake_depth=1, ##1.5

    runoff_coefficient=0.65,

    lake_strength=1.0, 
    )

    #rivers = hydro["rivers"]
    #lakes = hydro["lakes"]
    #lake_depth = hydro["lake_depth"]
    #lake_id = hydro["lake_id"]
    #accumulation = hydro["accumulation"]
    #flow_to = hydro["flow_to"]

    rivers0 = hydroresult["rivers"]
    lakes1 = hydroresult["lakes"]
    lake_depth = hydroresult["lake_depth"]
    lake_id = hydroresult["lake_id"]
    accumulation = hydroresult["accumulation"]
    flow_to = hydroresult["flow_to"]
    
    #plt.imshow(lakes1)
    #plt.show()
    #quit(-1)

    rivers1=np.copy(rivers0)
    rivers1=np.where(rivers1==0,np.nan, rivers1)
    distrivers=np.copy(rivers0)
    #distrivers=np.where(distrivers==np.nan,-1,1)
    distrivers=distance_to_someheight(distrivers, planet_radius, 1)
    
    distwater=np.copy(distsea)
    distwater=np.where(distwater>distrivers, distrivers, distwater)

    npp_naytettava = np.where(npp == -1, np.nan, npp)*landmask
    # 3. Kutsutaan aluohjelmaa keskiarvon laskentaan
    manner_keski_npp = laske_mannerten_keski_npp(npp, landmask, height, width)
    
    maski=np.isnan(precip_annual)
    painom_matriisi[maski] = 0

    planeetan_keski_lampotila  = np.average(temp_annual, weights=painom_matriisi)
    planeetan_keski_sademaara  = np.average(precip_annual, weights=painom_matriisi)
    print(f"Planeetan keskilämpötila: {planeetan_keski_lampotila:.2f} °C")
    print(f"Planeetan keskisademäärä: {planeetan_keski_sademaara:.2f} mm")
    # 4. NÄYTTÖ (Tulostetaan laskettu arvo erikseen mainissa)
    print("\n--- ANALYYSIN TULOKSET ---")
    print(f"Rasterin resoluutio: {width} x {height} pikseliä")
    print(f"Mantereiden pinta-alapainotettu keski-NPP: {manner_keski_npp:.2f} g/m²/vuosi")
    print("--------------------------")   
  
    #population_density_0=calculate_population_density(npp)*landmask
    population_density_0=10**(0.00096 * npp - 1.53)
    population_in_pixels=pix_areas*population_density_0*1/30
    # pop dens 0.05-1 people km2 max, hunter gather arctiic 0.001 ... 0.01

    habitability_monthly = calc_monthly_human_habitability(temps12, precips12)
    annual_habitability=annual = np.mean(habitability_monthly , axis=0)*landmask
    population_in_pixels=pix_areas*population_density_0*annual_habitability*0.0015

    planet_mean_habitability = np.average(annual_habitability, weights=painom_matriisi) 
    print("Mean human habitability",planet_mean_habitability  )
    planet_sum_population = np.sum( population_in_pixels) 
    print("Planet sum population time 0",planet_sum_population   )

    #centers0, nation_map0, nation_populations0 = muodosta_kansat(
    #population_in_pixels,
    #population_per_nation=50000,
    #min_distance_km=500, planet_radius=planet_radius
    #)
    #centers0, nation_map0, nation_populations0 = muodosta_kansat(
    #population_in_pixels,
    #relief,
    #planet_radius=6371.0,
    #population_per_nation=100_000,
    #min_distance_km=500,
    #)
    #centers0, nation_map0, nation_populations0 = muodosta_kansat(
    #population_in_pixels,
    #relief,
    #planet_radius,

    #population_per_nation=100_000,

    #min_distance_km=500,

    #slope_scale=15,
    #slope_power=2,

    #diagonal=True,
    #)
    #centers0, nation_map0, nation_populations0 = muodosta_kansat(
    #population_in_pixels,
    #relief,
    #planet_radius,

    #population_per_nation=100_000,

    #min_distance_km=500,

    # Maaston vaikutus
    #slope_scale=15,
    #slope_power=2,

    # Kulttuurisen etäisyyden vaikutus
    #culture_distance_scale_km=1500,
    #culture_power=2,

    #diagonal=True,
    #)

    centers0,nation_map0,nation_populations0,terrain_factor0,river_strength0,physical_distance0,total_cost0= muodosta_kansat(
    population_in_pixels,
    relief,
    planet_radius,
    population_per_nation=10000,
    min_distance_km=500,
    slope_scale=15,
    slope_power=2,
    culture_distance_scale_km=1500,
    culture_power=2,
    population_influence=0.35,
    population_scale=1000,
    river_influence=0.30,
    river_valley_scale=500,
    diagonal=True)

    
    #print("Kansoja:", len(centers0))
    #for i, (center, population) in enumerate(
    #    zip(centers0, nation_populations0),
    #    start=1
    #):
    #   print(
    #    i,
    #    center,
    #    int(population)
    #   )


    #plt.imshow(nation_map0*landmask)
    #plt.imshow(annual_habitability)
    #plt.imshow(population_density_0*landmask)
    #plt.imshow(npp*landmask)
    #plt.show()
    #quit(-1)    
  
  
  
    if(planet_mean_habitability>0.001):	
        if(planeetan_keski_lampotila<45):	
            calculate_culture_static()
        else:
            print("Too hot world")
    else:
        print("Not human habitable")
		
    print(".")
