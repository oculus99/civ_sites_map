


import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import (
    gaussian_filter,
    distance_transform_edt, map_coordinates
)


from PIL import Image
from noise import pnoise3


seed1=42

noisescale=0.6

will_load_image=False

imagename='orogen1.png'

sealevel=0.6
dem_min=-8000
dem_max=4000

width=720
height=360


polar_temp=-20
temp_diff=50


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

import numpy as np

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


def spherical_noise_native(width, height, scale, octaves=4, persistence=0.5, lacunarity=2.0, seed_offset=0.0):
    """Aiemmin luotu funktio pienenä variaationa (seed_offset lisätty erottamaan X/Y kohinat)."""
    lat = np.linspace(-np.pi / 2, np.pi / 2, height)
    lon = np.linspace(-np.pi, np.pi, width)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    
    nx = scale * np.cos(lat_grid) * np.cos(lon_grid)
    ny = scale * np.cos(lat_grid) * np.sin(lon_grid)
    nz = scale * np.sin(lat_grid) + seed_offset # Siirretään 3D-avaruudessa eri kohtaan
    
    v_pnoise3 = np.vectorize(lambda x, y, z: pnoise3(x, y, z, octaves=octaves, 
                                                     persistence=persistence, 
                                                     lacunarity=lacunarity))
    return normalize(v_pnoise3(nx, ny, nz))

def spherical_noise_offset(input_image, move_amount_max=10.0, scale=1.5):
    """
    Siirtää input_image-kuvan pikseleitä pallon pinnalla 3D-kohinan mukaan.
    Säilyttää saumattomuuden reunoilla ja napojen geometrian.
    """
    height, width = input_image.shape[:2]
    
    # 1. Luodaan kaksi erillistä kohinakarttaa (yksi pituusasteelle, yksi leveysasteelle)
    # Käytetään siirrosta (seed_offset) eri arvoja, jotta liikkeet eivät ole identtiset
    noise_lon = spherical_noise_native(width, height, scale, seed_offset=0.0)
    noise_lat = spherical_noise_native(width, height, scale, seed_offset=100.0)
    
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
                lacunarity=lacunarity
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


import numpy as np
import matplotlib.pyplot as plt

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

import numpy as np
import matplotlib.pyplot as plt

def laske_sademaara_soluilla(dem, sealevel=0.5):
    """
    Laskee sademäärän (mm/vuosi) käyttäen Hadleyn, Ferrelin ja polaarisolujen
    mukaisia globaaleja tuulivektoreita maaston sadenkatveefektille.
    """
    height, width = dem.shape
    
    # 1. LEVEYSPIIRIT (-1 = Etelänapa, 0 = Päiväntasaaja, 1 = Pohjoisnapa)
    y_lin = np.linspace(-1, 1, height)
    _, Y_leveyspiirit = np.meshgrid(np.arange(width), y_lin)
    
    # 2. GLOBAALI POHJASADEMÄÄRÄ (Ilmanpaineen vyöhykkeet)
    # Päiväntasaajan sateet, kuivat hevosleveysasteet ja lauhkeat rintamasateet
    leveyspiiri_sade = 2200 * np.exp(-15 * Y_leveyspiirit**2) + \
                       900 * np.exp(-25 * (abs(Y_leveyspiirit) - 0.6)**2) + 100

    # 3. GLOBAALIT TUULIVEKTORIT (Hadley, Ferrel, Polaarinen)
    # Luodaan X-suuntainen tuulikartta. 
    # - Positiivinen = Länsituuli (vasemmalta oikealle)
    # - Negatiivinen = Itätuuli/Pasaati (oikealta vasemmalle)
    tuuli_x = np.zeros_like(Y_leveyspiirit)
    
    # Pasaatituulet (Hadley-solu: 0° - 30° eli Y välillä -0.33 ... 0.33) -> Puhaltaa itälounaaseen (Negatiivinen X)
    hadley_maski = abs(Y_leveyspiirit) <= 0.33
    tuuli_x[hadley_maski] = -1.0
    
    # Länsituulet (Ferrel-solu: 30° - 60° eli Y välillä 0.33 ... 0.66) -> Puhaltaa itään (Positiivinen X)
    ferrel_maski = (abs(Y_leveyspiirit) > 0.33) & (abs(Y_leveyspiirit) <= 0.66)
    tuuli_x[ferrel_maski] = 1.0
    
    # Polaarituulet (Polaarisolu: 60° - 90° eli Y > 0.66) -> Puhaltaa länteen (Negatiivinen X)
    polaari_maski = abs(Y_leveyspiirit) > 0.66
    tuuli_x[polaari_maski] = -0.8

    # 4. MAASTON VAIKUTUS (Tuulivektori kohtaa maaston gradientin)
    # Lasketaan maaston kaltevuus x-suunnassa
    _, grad_x = np.gradient(dem)
    
    # Orografinen sade syntyy, kun tuuli puhaltaa YLÄMÄKEEN maaston suuntaan.
    # Pistetulo: tuuli_x * grad_x. Jos molemmat positiivisia tai molemmat negatiivisia,
    # tuuli osuu rinteeseen -> Ilma nousee -> Sataa.
    orografinen_efekti = tuuli_x * grad_x * 5000
    
    # Vain mantereilla on väliä vuoristosateella
    manner_maski = dem > sealevel
    
    # 5. LOPULLINEN YHDISTÄMINEN
    sademaara = leveyspiiri_sade.copy()
    sademaara[manner_maski] += orografinen_efekti[manner_maski]
    
    # Rajoitetaan loogisiin sademääriin
    sademaara = np.clip(sademaara, 40, 5000)
    
    return sademaara, tuuli_x


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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
    'Meri', 'Aavikko', 'Savanni / Ruohikko', 'Sademetsä', 
    'Lauhkea metsä', 'Havumetsä', 'Tundra', 'Ikijää'
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
        imagee=spherical_noise_offset(imagee, move_amount_max=int(360/4), scale=0.75)
        imagee=normalize(imagee)
    	
    dem, landmask=create_dem_from_array(imagee, sealevel, dem_min, dem_max)
    relief=np.copy(dem)*landmask
    relief2=np.copy(relief)
    relief2=np.where(relief2==0,np.nan,relief2 )
    light_direction = [-0.0, 1, 0.0] 
    # 3. Lasketaan varjot
    varjokartta = laske_ray_trace_shadows(relief, light_direction)
    # Lasketaan emboss/hillshade (valo luoteesta, 45 asteen kulmassa)
    varjostus = laske_hillshade(relief, azimuth=315, angle_altitude=45)
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



    #plt.imshow(imagee)
    plt.imshow(relief2, cmap="terrain",origin='lower')
    plt.imshow(varjostus, cmap='gray', alpha=0.2, origin='lower')
    #plt.imshow(dem)
    #plt.imshow(landmask)
    plt.show()
    # --- PIIRETÄÄN KARTTA ---
    plt.figure(figsize=(12, 5))
    # Käytetään 'coolwarm'-värikarttaa (sininen = kylmä, punainen = kuuma)
    plt.imshow(lampotilat, cmap='coolwarm', origin='lower')
    plt.colorbar(label="Vuoden keskilämpötila (°C)")
    plt.title("Maailman lämpötilajakautuma (Leveyspiiri + Korkeusasema)")
    plt.axis('off')
    plt.show()
    sateet, tuulet = laske_sademaara_soluilla(relief, sealevel=0.5)
    # --- VISUALISOINTI ---
    fig, ax = plt.subplots(figsize=(12, 6))
    # Piirretään sademääräkartta
    im = ax.imshow(sateet, cmap='YlGnBu', origin='lower')
    fig.colorbar(im, label="Sademäärä (mm / vuosi)")
    # Lisätään nuolia kuvaamaan tuulen suuntaa valituilla leveyspiireillä (näytetään joka 20. rivi)
    for row in range(10, height, 20):
        suunta = "--> LÄNSITUULI" if tuulet[row, 0] > 0 else "<-- PASAATI / ITÄTUULI"
        ax.text(10, row, suunta, color='red', fontsize=9, fontweight='bold', va='center')
    ax.set_title("Sademäärä globaaleilla ilmastosoluilla (Hadley & Ferrel)")
    ax.axis('off')
    plt.show()
    # 2. Generoidaan biomikartta
    kartta = luo_biomikartta(relief, lampotilat, sateet, sealevel=0.5)
    # 3. Piirretään lopputulos
    plt.figure(figsize=(14, 7))
    im = plt.imshow(kartta, cmap=cmap_biomit, origin='lower', vmin=0, vmax=7)
    # Tehdään selkeä selite (legend) biomeille
    cbar = plt.colorbar(im, ticks=np.arange(8), fraction=0.03, pad=0.04)
    cbar.ax.set_yticklabels(biomi_nimet)
    plt.title("Proseduraalisen maailman perusbiomit (Whittakerin luokittelu)")
    plt.axis('off')
    plt.show()
quit(-1)


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
