
#######################################
#
## Create fractal world climate and biomes
## ## estimate primary civilization areas
#
## simple mean t, deltat approach
##
## 30.08.2026 0000.0005
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

from PIL import Image

from noise import pnoise3

import rasterio


##seed1=42
##seed1=777
seed1=1246
#seed1=321


noisescale=0.6

will_load_image=False

#imagename='orogen1.png'
imagename='terra.png'

sealevel=0.5
dem_min=-7000
dem_max=5000

width=720
height=360

## NOTE theres affects only some climate props!!!

#ecc=0.0167
#tilt=23.44
#mvelp=102.7

## basic parematers of planet, here terra-like composition

S1=1361*(1/1) ## S1 solar constant W m-2
ecc=0.0167 ## eccentricity
tilt=23.44 ## axis tilt or obliquity
mvelp=102.7 ## position of axis against orbit

air_pressure_atm=1
atmos_co2_ppm= 280
planet_mass_me=1
rotation_period_days=1

planet_radius_re=math.pow(planet_mass_me, 0.27) ## terra-like internal composition 
planet_radius=6371*planet_radius_re #3 km

#mean_temp=15
mean_temp=15
temp_diff=70**math.sqrt(rotation_period_days)*math.sqrt(planet_radius_re)/air_pressure_atm

temp_dev=temp_diff/2
polar_temp=mean_temp-temp_dev
temp_diff=mean_temp+temp_dev


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


def spherical_noise_native(width, height, scale, octaves=4, persistence=0.5, lacunarity=2.0, seed_offset=0.0, seed_value=12):
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


def laske_tuuli_origo(dem, kiertosolut=3):
    # 1. LEVEYSPIIRIT
    y_lin = np.linspace(-1, 1, height)
    _, Y_leveyspiirit = np.meshgrid(np.arange(width), y_lin)

    tuuli_x = np.zeros_like(Y_leveyspiirit)
    tuuli_y = np.zeros_like(Y_leveyspiirit)
    tuuli_z = np.zeros_like(Y_leveyspiirit)

    # ---------------------------------------------------------
    # 2. ILMAKEHÄN KIERTOSOLUT
    # ---------------------------------------------------------
    #
    # Esim. kiertosolut=3:
    #
    # Etelä:
    #   polaarisolu  -> länsi
    #   ferrel       -> itä
    #   hadley       -> länsi
    #
    # Pohjoinen:
    #   hadley       -> länsi
    #   ferrel       -> itä
    #   polaarisolu  -> länsi
    #
    # Yksinkertaistettu malli:
    # jokainen solu vaihtaa tuulen suunnan.

    solun_leveys = 2.0 / (kiertosolut * 2)

    for i in range(kiertosolut * 2):
        y_min = -1.0 + i * solun_leveys
        y_max = y_min + solun_leveys

        maski = (
            (Y_leveyspiirit >= y_min) &
            (Y_leveyspiirit < y_max)
        )

        # Vaihdetaan suunta soluittain
        if i % 2 == 0:
            tuuli_x[maski] = -1.0
        else:
            tuuli_x[maski] = 1.0

    return tuuli_x, tuuli_y, tuuli_z
## ... origo2 code



def laske_tuuli_2(dem, kiertosolut=3):
    # LEVEYSPIIRIT
    y_lin = np.linspace(-1, 1, height)
    _, Y = np.meshgrid(np.arange(width), y_lin)

    tuuli_x = np.zeros_like(Y, dtype=float)
    tuuli_y = np.zeros_like(Y, dtype=float)
    tuuli_z = np.zeros_like(Y, dtype=float)
    # ---------------------------------------------------------
    # 1. GLOBAALI ILMAKEHÄN KIERTOLIIKE
    # ---------------------------------------------------------
    # kiertosolut = solujen määrä per pallonpuolisko
    # 3 = Maan kaltainen Hadley/Ferrel/Polaarinen-rakenne

    solu_leveys = 1.0 / kiertosolut

    for solu in range(kiertosolut):
        # Eteläisen ja pohjoisen pallonpuoliskon solut
        for merkki in (-1, 1):

            y_min = (solu / kiertosolut)
            y_max = ((solu + 1) / kiertosolut)

            maski = (
                (np.abs(Y) >= y_min) &
                (np.abs(Y) < y_max)
            )

            # Vaihtuva tuulensuunta:
            # 0 = itätuuli
            # 1 = länsituuli
            if solu % 2 == 0:
                tuuli_x[maski] = -1.0
            else:
                tuuli_x[maski] = 1.0

    # ---------------------------------------------------------
    # 2. MAASTON VAIKUTUS
    # ---------------------------------------------------------
    # Ilma pyrkii kulkemaan korkeuserojen mukaan.
    # Tämä antaa tuulelle myös Y-komponentin.

    dem_y, dem_x = np.gradient(dem)

    # Pieni vaikutus, ettei paikallinen maasto kumoa
    # kokonaan globaalia tuulijärjestelmää.
    tuuli_x += -dem_x * 0.15
    tuuli_y += -dem_y * 0.15

    # ---------------------------------------------------------
    # 3. MANTEREIDEN SISÄOSIEN KORKEAPAINE
    # ---------------------------------------------------------
    # Hyvin yksinkertainen approksimaatio:
    # kaukana korkeuserojen reunoista oleva maa saa
    # laskevan ilman komponentin.

    # Maaston kaltevuuden voimakkuus
    kaltevuus = np.hypot(dem_x, dem_y)

    # Tasaiset alueet tulkitaan helpommin "sisämaaksi".
    sisamaa = kaltevuus < np.percentile(kaltevuus, 60)

    # Pieni pystysuuntainen laskeva liike
    tuuli_z[sisamaa] -= 0.1

    # ---------------------------------------------------------
    # 4. NORMALISOINTI
    # ---------------------------------------------------------
    nopeus = np.sqrt(
        tuuli_x**2 +
        tuuli_y**2 +
        tuuli_z**2
    )

    nopeus = np.maximum(nopeus, 1e-6)

    tuuli_x /= nopeus
    tuuli_y /= nopeus
    tuuli_z /= nopeus

    return tuuli_x, tuuli_y, tuuli_z

def laske_tuuli3(dem, kiertosolut=3):
    """
    Laskee planeetan yksinkertaistetun globaalin tuulikentän.

    Parametrit
    ----------
    dem : np.ndarray
        Maaston korkeusdata. Meri = 0, maa > 0.
    kiertosolut : int
        Ilmakehän kiertosolujen määrä per pallonpuolisko.
        Oletus 3 vastaa Maan kaltaista kolmisoluista rakennetta.

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

    # kiertosolut = solujen määrä per pallonpuolisko
    # 3 -> Hadley + Ferrel + Polaari
    #
    # Solut vaihtavat tuulen pääsuuntaa.
    # Solujen rajat jaetaan automaattisesti.

    solu_leveys = 1.0 / kiertosolut

    for solu in range(kiertosolut):

        y_min = solu * solu_leveys
        y_max = (solu + 1) * solu_leveys

        # Pohjoinen ja eteläinen pallonpuolisko
        maski = (
            (np.abs(Y) >= y_min) &
            (np.abs(Y) < y_max)
        )

        # Vuorotteleva pääsuunta.
        #
        # solu 0 = itätuuli
        # solu 1 = länsituuli
        # solu 2 = itätuuli
        # jne.

        if solu % 2 == 0:
            tuuli_x[maski] = -1.0
        else:
            tuuli_x[maski] = 1.0

    # ---------------------------------------------------------
    # 4. MAASTON PAIKALLINEN VAIKUTUS
    # ---------------------------------------------------------

    dem_y, dem_x = np.gradient(dem)

    # Tuuli pyrkii seuraamaan paine- ja maastovaikutuksia.
    # Vaikutus pidetään pienenä, jotta globaali kiertoliike
    # säilyy hallitsevana.

    tuuli_x += -dem_x * 0.10
    tuuli_y += -dem_y * 0.10

    # ---------------------------------------------------------
    # 5. VUORISTOJEN NOUSU-/LASKUVIRTA
    # ---------------------------------------------------------

    kaltevuus = np.hypot(dem_x, dem_y)

    # Vain maan päällä.
    #
    # Nousua tapahtuu erityisesti voimakkailla rinteillä.
    # Tämä ei vielä ole varsinainen sade-/pilvimalli.

    vuoristo = landmask & (kaltevuus > np.percentile(kaltevuus, 75))

    tuuli_z[vuoristo] += kaltevuus[vuoristo] * 0.10

    # Tasainen sisämaa saa hyvin pienen laskevan komponentin.
    # Tämä toimii tässä vaiheessa yksinkertaisena
    # manneralueen korkeapainevaikutuksen approksimaationa.

    tasainen_maa = landmask & (kaltevuus < np.percentile(kaltevuus, 50))

    tuuli_z[tasainen_maa] -= 0.02

    # ---------------------------------------------------------
    # 6. NORMALISOINTI
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


def laske_tuuli(dem, kiertosolut=3):
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

import numpy as np


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



import numpy as np


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
        normalize(kastelupotentiaali) * 0.30 *
        normalize(pisteet_lampo)      * 0.25 
        #normalize(pisteet_joki)       * 0.20 *
        #normalize(kauppapotentiaali)  * 0.15 *
        #normalize(sade_kuivuus)       * 0.60
    )

    kokonaispisteet=normalize(kokonaispisteet)
   
    #plt.imshow(kokonaispisteet, cmap="rainbow")
    #plt.show()    
    # ---------------------------------------------------------
    # 9. TURVALLINEN RAJAUS
    # ---------------------------------------------------------

    return np.clip(kokonaispisteet, 0.0, 1.0)






#def laske_co2_delta_t(co2_ppm, co2_viite=280.0, ilmastosensitiivisyys=0.8):
def laske_co2_delta_t(co2_ppm, co2_viite=0.15, ilmastosensitiivisyys=0.8):
    """
    Laskee CO2-pitoisuuden muutoksesta johtuvan lämpötilan muutoksen (Delta T).
    
    Parametrit:
    - co2_ppm: Nykyinen tai haluttu CO2-pitoisuus (ppm)
    - co2_viite: Vertailupisteen pitoisuus (esim. esiteollinen aika 280 ppm)
    - ilmastosensitiivisyys: Lämpötilan muutos per W/m^2 (keskiarvo n. 0.8 K/(W/m^2))
    """
    if co2_ppm <= 0:
        raise ValueError("CO2-pitoisuuden täytyy olla suurempi kuin 0.")
        
    # 1. Lasketaan säteilypakote (Radiative Forcing, dF) vakiintuneella kaavalla
    # dF = 5.35 * ln(C / C0)
    dF = 5.35 * np.log(co2_ppm / co2_viite)
    
    # 2. Lasketaan lämpötilan muutos (Delta T)
    delta_t = ilmastosensitiivisyys * dF
    
    return dF, delta_t
# Vakio: Stefanin-Boltzmannin vakio (W / (m^2 * K^4))
SIGMA = 5.670374419e-8

def laske_greenhouse_factor(T_surface, OLR=None, albedo=0.30, S0=1361):
    """
    Laskee kasvihuoneilmiön voimakkuuskertoimen G.
    
    Parametrit:
    - T_surface: Maanpinnan keskilämpötila kelvineinä (esim. 15 °C = 288.15 K)
    - OLR: Avaruuteen karkaava pitkäaaltoinen säteily (W/m²). 
           Jos None, oletetaan energiatasapaino: OLR = (S0 / 4) * (1 - albedo)
    - albedo: Maapallon kokonaisalbedo (heijastuskyky, n. 0.30)
    - S0: Aurinkovakio (n. 1361 W/m²)
    """
    # 1. Lasketaan maanpinnan lähettämä mustan kappaleen säteily (E_surface)
    E_surface = SIGMA * (T_surface ** 4)
    
    # 2. Määritetään avaruuteen karkaava säteily (OLR)
    if OLR is None:
        OLR = (S0 / 4) * (1 - albedo)
        
    # 3. Lasketaan kerroin G
    # G = (E_surface - OLR) / E_surface
    G = (E_surface - OLR) / E_surface
    
    return E_surface, OLR, G




def arvioi_planeetan_lampotilat(ecc, tilt, mvelp, S1, albedo_keski=0.3,G=0.39):
    """Laskee planeetan keskilämpötilan sekä päiväntasaajan ja napojen lämpötilaerot.

    Parametrit:
    -----------
    ecc : float          - Radan soikeus (0.0 - 0.9)
    tilt : float         - Akselin kaltevuus asteina (0 - 90)
    mvelp : float        - Perihelin pituus kevätpäiväntasauksesta asteina (0 - 360)
    S1 : float           - Tähtivakio / Aurinkovakio (W/m^2) keski etäisyydellä 1361
    albedo_keski : float - Planeetan keskimääräinen heijastuskyky (0.0 - 1.0)
    """
    # Stefanin-Boltzmannin vakio
    sigma = 5.67e-8
    # Ilmakehän kasvihuonekerroin (Maan nykyinen tehokas emissiivisyys on n. 0.61 -> G = 0.39)
    ##G = 0.39

    # 1. GLOBAALI KESKILÄMPÖTILA
    # ecc nostaa kokonaisenergiaa tekijällä 1 / sqrt(1 - ecc^2)
    vuo_globaali = (S1 / 4) * (1 - albedo_keski) / np.sqrt(1 - ecc**2)
    T_globaali_K = (vuo_globaali / (sigma * (1 - G))) ** 0.25
    T_globaali_C = T_globaali_K - 273.15

    # 2. ENERGIAN JAKAUTUMINEN LEVEYSASTEILLE (Vuotuinen keskiarvo approksimaationa)
    # Mitä suurempi tilt, sitä enemmän energiaa siirtyy päiväntasaajalta navoille.
    tilt_rad = np.radians(tilt)
    mvelp_rad = np.radians(mvelp)

    # Vuotuinen keskimääräinen geometrinen insolaatiotekijä (pohjautuu sarjakehitelmiin)
    # Päiväntasaajalla (lat = 0)
    vuo_korjaus_eq = (2 / np.pi) * np.cos(tilt_rad) + (tilt_rad / np.pi) * np.sin(
        tilt_rad
    )  # suhteellinen osuus
    # Korjataan pyöreyden ja soikeuden suhde
    vuo_eq = (S1 / np.pi) * vuo_korjaus_eq / np.sqrt(1 - ecc**2)

    # Navoilla (lat = 90) vuotuinen insolaatio riippuu suoraan sin(tilt)
    # mvelp määrittää pienen eron pohjoisen (N) ja etelän (S) välille radan soikeuden vuoksi
    vuo_pole_base = (S1 / np.pi) * np.sin(tilt_rad) / np.sqrt(1 - ecc**2)

    # mvelp siirtää painopistettä: jos mvelp=270, pohjoisnavan kesä on perihelissä (kuumempi)
    pohjoinen_epasymmetria = 1 + ecc * np.sin(mvelp_rad)
    etelainen_epasymmetria = 1 - ecc * np.sin(mvelp_rad)

    vuo_pole_N = vuo_pole_base * pohjoinen_epasymmetria
    vuo_pole_S = vuo_pole_base * etelainen_epasymmetria

    # Muunnetaan vuot lämpötiloiksi (Celsius) ottaen huomioon albedo ja kasvihuoneilmiö
    def vuo_to_celsius(vuo_pinta):
        K = (vuo_pinta * (1 - albedo_keski) / (sigma * (1 - G))) ** 0.25
        return K - 273.15

    T_eq = vuo_to_celsius(vuo_eq * 0.75)  # Geometria- ja diffuusiokorjaus tasolle
    T_pole_N = vuo_to_celsius(vuo_pole_N * 0.85)
    T_pole_S = vuo_to_celsius(vuo_pole_S * 0.85)

    # Keskimääräinen napalämpötila eron laskemista varten
    T_pole_keski = (T_pole_N + T_pole_S) / 2
    delta_T = T_eq - T_pole_keski

    return {
        "Globaali keskilämpötila (°C)": round(T_globaali_C, 1),
        "Päiväntasaajan lämpötila (°C)": round(T_eq, 1),
        "Pohjoisnapa (°C)": round(T_pole_N, 1),
        "Etelänapa (°C)": round(T_pole_S, 1),
        "Päiväntasaaja-Napa ero (ΔT °C)": round(delta_T, 1),
    }


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
        imagee = generate_spherical_noise(width, height, scale=noisescale, octaves=16, seed=seed1)
        #imagee=np.pow(imagee,2)
        #imagee=np.exp(np.exp(np.exp(imagee)))
        imagee=np.pow(imagee,3)
        imagee=normalize(imagee)
        #imagee=sigmoid_dem(imagee, saatokerroin=0.5, keskiarvo=0.5) 
        #imagee=sigmoid_meri_manner_jakauma(imagee, säätökerroin=-2)
        imagee=muotoile_maan_jakauma(imagee, sealevel=0.5)
        imagee=normalize(imagee)
        imagee=spherical_noise_offset(imagee, move_amount_max=int(360/4), scale=0.75, seed=seed1)
        imagee=normalize(imagee)

    
    print("S1, ecc, tilt, mvelp")
    print(S1, ecc, tilt, mvelp)
    print("Rotation        P / h       :" , rotation_period_days*24)
    print("mass, radius in earth units :", planet_mass_me, planet_radius_re)
    
	# Maapallon nykyarvot: T_surface = 15 °C (288.15 K), OLR = n. 239 W/m²
    #T_maa = 288.15
    #OLR_maa = 239.0
    #E_surf, olr, G_arvo = laske_greenhouse_factor(T_surface=T_maa, OLR=OLR_maa)
    #print(f"Maanpinnan säteily (E_surface): {E_surf:.1f} W/m²")
    #print(f"Avaruuteen karkaava säteily (OLR): {olr:.1f} W/m²")
    #print(f"Kasvihuonekerroin G: {G_arvo:.3f} (eli {G_arvo*100:.1f} %)")
    # --- AJETAAN ESIMERKKI (Maan kaltaiset arvot) ---
    #planet_base_temps = arvioi_planeetan_lampotilat(ecc=0.0167, tilt=23.44, mvelp=102.7)
    planet_base_temps = arvioi_planeetan_lampotilat(ecc=0.0167, tilt=23.44, mvelp=102.7, S1=S1)
    #for avain, arvo in planet_base_temps.items():
    #   print(f"{avain}: {arvo}")

    #return {
    #    "Globaali keskilämpötila (°C)": round(T_globaali_C, 1),
    #    "Päiväntasaajan lämpötila (°C)": round(T_eq, 1),
    #    "Pohjoisnapa (°C)": round(T_pole_N, 1),
    #    "Etelänapa (°C)": round(T_pole_S, 1),
    #"    "Päiväntasaaja-Napa ero (ΔT °C)": round(delta_T, 1),
    #"}
    global_temp= list(planet_base_temps.values())[0]
    global_temp_no_greenhouse= list(planet_base_temps.values())[1]
    global_deltat_poles_to_equator= list(planet_base_temps.values())[4]
    #print(global_temp)
    #print(global_temp_no_greenhouse)
    #print(global_deltat_poles_to_equator)
    #quit(-1)    
    # Esimerkki: Lasketaan nykyiselle pitoisuudelle (esim. 420 ppm)

    pakote, deltatemp_gh = laske_co2_delta_t(atmos_co2_ppm*air_pressure_atm)
    global_calculated_temp=global_temp_no_greenhouse+deltatemp_gh
    #print(f"CO2-pitoisuus: {atmos_co2_ppm} ppm (Vertailutaso: 280 ppm, 1 atm)")
    #print(f"Säteilypakote (dF): {pakote:.2f} W/m²")
    #print(f"Lämpötilan muutos (Delta T): {deltatemp_gh:.2f} °C")
    #print(f"Total global temp : {global_calculated_temp:.2f} °C")    

    #air_pressure_atm=1

    #mean_temp=15
    temp_diff=3.5*global_deltat_poles_to_equator*math.sqrt(rotation_period_days)*math.sqrt(planet_radius_re)/air_pressure_atm
    mean_temp=global_calculated_temp
    temp_dev=temp_diff/2
    polar_temp=mean_temp-temp_dev
    temp_diff=mean_temp+temp_dev
    print(" Global temperature                   C : ", round(mean_temp,2))
    print(" Temperature difference equator-poles C :",temp_diff)
    #quit(-1)
    
    np.random.seed(seed=seed1)
    lats = np.linspace(90, -90, height)
    lons= np.linspace(-180, -180, width)
    painot_1d = np.cos(np.radians(lats))
    painom_matriisi = np.repeat(painot_1d[:, np.newaxis], width, axis=1)
    	
    dem, landmask=create_dem_from_array(imagee, sealevel, dem_min, dem_max)
    
    relief=(np.copy(dem))*landmask
    seamask=np.copy(relief)
    seamask=np.where(seamask>0,0,1)
    
    #plt.imshow(seamask)
    #plt.show()
    #quit(-1)


    distsea=distance_to_sea(relief, planet_radius)
    distmountains=distance_to_someheight(relief, planet_radius, 1500)
    twi = calculate_twi(dem, cell_size=100.0)    
    tpi=calculate_tpi(dem, window_size=5)
  
    #plt.imshow(distrivers)        
    #plt.imshow(distmountains)
    #plt.imshow(rivers1)
    #plt.show()
    #quit(-1)
    
    relief2=np.copy(relief)
    relief2=np.where(relief2<=0,np.nan,relief2 )
    light_direction = [-0.0, 1, 0.0] 
    # 3. Lasketaan varjot
    hillshade2 = laske_ray_trace_shadows(relief, light_direction)
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

    # Lasketaan lämpötilat
    lampotilat = laske_vuosittainen_lampotila(relief, polar_temp, temp_diff, sealevel=0.5, max_korkeus_m=5000)
    pet = calculate_pet(lampotilat,relief,landmask)
    ice=np.copy(lampotilat)
    ice=np.where(lampotilat<-10,1,np.nan)
    noice=np.where(lampotilat<-10,np.nan,1)
    maski = np.isnan(lampotilat)
    alueen_keski_lampotila = np.average(lampotilat, weights=painom_matriisi)
    moisture_coeff=tulokset['meri_prosentti']/100.0
    #moisture_temperature=
    delta_T = (alueen_keski_lampotila - 15.0) ## earh now 15 c
    sademäärä_kerroin = 0.02
    uusi_sadek = (1 + sademäärä_kerroin * delta_T)
    uusi_sadek = max(0.0, uusi_sadek)
    kosteus_kerroin_kaatosade = 0.07
   
    uusi_kosteus_kapasiteetti_kaatosade = 1* math.exp(kosteus_kerroin_kaatosade * delta_T)
    print(moisture_coeff, uusi_sadek)
    moisture_coeff=moisture_coeff*uusi_sadek*2.5
    print(moisture_coeff)
    ## earth now 15 c, 990 mm
    sateet, tuulet = laske_sademaara_soluilla(relief, distsea, sealevel=0.5, moisture_coeff=moisture_coeff)
    rivers0, accumulation0, flow_to0=calculate_rivers(relief,sateet, pet, 10,100000)
    lakes1, lake_depths1, lake_ids1 = calculate_lakes(relief,sateet,pet,flow_to0,accumulation0,cell_size=1000,min_inflow=1000000)
    plt.imshow(lakes1)
    plt.show()    
    #rivers, accumulation, flow_to
    rivers1=np.where(rivers0==0,np.nan, rivers0)
    distrivers=np.copy(rivers0)
    #distrivers=np.where(distrivers==np.nan,-1,1)
    distrivers=distance_to_someheight(distrivers, planet_radius, 1)
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
    kartta = luo_biomikartta(relief, lampotilat, sateet, sealevel=0.5)
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

    habitable_temp=np.copy(lampotilat)
    habitable_temp=np.where(habitable_temp>-10,habitable_temp,1)
    habitable_temp=np.where(habitable_temp<35,1,habitable_temp)
    aavikko=np.copy(kartta)
    aavikko=np.where(aavikko==1,1,0)
    vuoriin=np.copy(distmountains)
    mereen=np.copy(distsea)
    vuoriin=np.where(vuoriin<500,1,0)
    mereen=np.where(mereen<500,1,0)
    sivilisaatio=habitable_temp*aavikko*mereen*vuoriin
    sivilisaatio=np.where(sivilisaatio==1,1,np.nan )
    
    civ_raster=laske_sivilisaatiopisteet(sateet, lampotilat, distrivers, distsea, distmountains, relief)# --- ESIMERKKIKÄYTTÖ ---
    civ_sites = find_max_points(civ_raster,planet_radius_km=6371.0,min_distance_km=500.0,n_points=5)
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
    im = plt.imshow(kartta, cmap=cmap_biomit, origin='upper', vmin=0, vmax=7, extent=[-180,180,-90,90])
    shad = plt.imshow(hillshade1*landmask, cmap="gray", origin='upper', extent=[-180,180,-90,90], alpha=0.3)   
    plt.imshow(rivers1*noice, cmap="viridis",origin='upper', extent=[-180,180,-90,90]) 
    
    civim=plt.imshow(sivilisaatio, cmap="coolwarm_r",origin='upper', alpha=0.4, extent=[-180,180,-90,90])
    plt.scatter(
        civ_lons,
        civ_lats,
        s=80,
        c="red",
        edgecolors="white",
        linewidths=1.5,
        zorder=10
    )
    # Tehdään selkeä selite (legend) biomeille
    plt.contour(relief, lw=2, levels=[1500], alpha=0.5, color="#100000", origin="upper",extent=[-180,180,-90,90])
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
