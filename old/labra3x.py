import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.transform import resize
from scipy import ndimage
import noise  # Perlin-kohinalle, asenna: pip install noise

def add_mountain_ranges(dem, latitude_weight=True):
    """Lisää vuoristojaksoja uskottavasti"""
    height, width = dem.shape
    
    # 1. Perlin-kohinaa vuoristoille
    scale = 0.5
    octaves = 6
    persistence = 0.5
    lacunarity = 2.0
    
    mountains = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            nx = x/width - 0.5
            ny = y/height - 0.5
            # Käytä 3D-kohinaa saadaksesi parempia muodostumia
            elevation = noise.pnoise3(nx * scale, 
                                    ny * scale, 
                                    0.5, 
                                    octaves=octaves,
                                    persistence=persistence, 
                                    lacunarity=lacunarity,
                                    repeatx=1024,
                                    repeaty=1024,
                                    base=42)
            mountains[y,x] = elevation
    
    # 2. Normalisoi ja tee terävämpiä huippuja
    mountains = (mountains - mountains.min()) / (mountains.max() - mountains.min())
    mountains = np.power(mountains, 1.5)  # Tee huipuista terävämpiä
    
    # 3. Levitä vuoristoja napojen alueille
    if latitude_weight:
        # Luo leveysaste-painotus (enemmän vuoria napoihin päin)
        lat_weight = np.abs(np.linspace(-1, 1, height))
        lat_weight = np.power(lat_weight, 2)  # Korosta napoja
        lat_weight = lat_weight.reshape(-1, 1) * np.ones(width)
        
        mountains *= lat_weight
    
    return mountains

def add_plate_tectonics(dem):
    """Simuloi laattatektoniikkaa vuoristojen muodostumiseen"""
    height, width = dem.shape
    
    # Laattojen reunat (simuloi törmäysvyöhykkeitä)
    plate_boundaries = np.zeros((height, width))
    
    # Lisää muutamia kaarevia laattareunoja
    boundaries = [
        (0.3, 0.7, 0.1),  # (x, y, voimakkuus)
        (0.7, 0.3, 0.15),
        (0.5, 0.8, 0.12),
        (0.2, 0.4, 0.08),
    ]
    
    for bx, by, strength in boundaries:
        for y in range(height):
            for x in range(width):
                dx = (x/width - bx)
                dy = (y/height - by)
                dist = np.sqrt(dx*dx + dy*dy)
                # Kaareva reuna
                boundary_effect = np.exp(-dist * 20) * strength
                plate_boundaries[y,x] += boundary_effect
    
    return plate_boundaries

def enhance_glacial_topography(dem, original_biomes):
    """Parantaa jäätikköalueiden topografiaa"""
    height, width = dem.shape
    
    # Etsi jäätikköalueet (alkuperäinen arvo 3)
    glacial_mask = (original_biomes == 3)
    
    # 1. Jäätiköiden alla voi olla vuoria
    glacial_mountains = np.zeros((height, width))
    
    # Käytä kohinaa luodaksesi vuoristoa jäätiköiden alle
    for y in range(height):
        for x in range(width):
            if glacial_mask[y,x]:
                nx = x/width
                ny = y/height
                # Eri taajuus jäätikkövuorille
                mountain_val = noise.pnoise2(nx * 0.2, ny * 0.2, octaves=4, base=123)
                glacial_mountains[y,x] = max(0, mountain_val * 0.3)
    
    # 2. Lisää jäätikköjen reunoille moreenimaisemaa
    glacial_edges = ndimage.gaussian_filter(glacial_mask.astype(float), sigma=1.0)
    glacial_edges = glacial_edges * (1 - glacial_edges)  # Etsi reunat
    
    return glacial_mountains, glacial_edges

def create_improved_dem(original_biomes):
    """Luo parannellun DEM:n systeemisen lähestymistavan avulla"""
    print("Luodaan paranneltua topografiaa...")
    
    height, width = original_biomes.shape
    
    # Perus DEM alkuperäisistä maastotyypeistä
    base_dem = np.zeros((height, width))
    base_dem = np.where(original_biomes == 0, 0.1, base_dem)    # vesi -> matala
    base_dem = np.where(original_biomes == 1, 0.3, base_dem)    # metsä -> mäkiä
    base_dem = np.where(original_biomes == 2, 0.7, base_dem)    # kivi -> korkeaa
    base_dem = np.where(original_biomes == 3, 0.2, base_dem)    # jää -> keskikorkea
    base_dem = np.where(original_biomes == 4, 0.15, base_dem)   # hiekka -> matala
    
    # 1. Lisää vuoristojaksot
    mountains = add_mountain_ranges(original_biomes)
    
    # 2. Lisää laattatektoniikka-efekti
    plate_effects = add_plate_tectonics(original_biomes)
    
    # 3. Paranna jäätikköalueiden topografiaa
    glacial_mountains, glacial_edges = enhance_glacial_topography(base_dem, original_biomes)
    
    # 4. Yhdistä kaikki komponentit
    improved_dem = base_dem.copy()
    
    # Lisää vuoristot (painotetaan olemassa olevien kivialueiden päälle)
    mountain_weight = np.where(original_biomes == 2, 1.5, 0.8)  # Enemmän vaikutusta olemassa oleville vuorille
    improved_dem += mountains * 0.4 * mountain_weight
    
    # Lisää laattatektoniikka-efekti
    improved_dem += plate_effects * 0.3
    
    # Lisää jäätikkövuoret
    improved_dem += glacial_mountains * 0.5
    
    # Lisää moreenireunat
    improved_dem += glacial_edges * 0.2
    
    # Varmista, että arvot pysyvät [0,1] välillä
    improved_dem = np.clip(improved_dem, 0, 1)
    
    # 5. Viimeistely: Gaussian-suodatus sileämmän näköisille vuorille
    improved_dem = ndimage.gaussian_filter(improved_dem, sigma=0.5)
    
    return improved_dem, mountains, plate_effects, glacial_mountains

# PÄÄOHJELMA
def main():
    # Lue data
    data1 = np.loadtxt("material_map_helliconia.cfg", dtype=float)
    print("Alkuperäisen datan koko:", data1.shape)
    
    # Luo paranneltu DEM
    improved_dem, mountains, plates, glacial_mts = create_improved_dem(data1)
    
    # Visualisoi eri vaiheet
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Alkuperäinen biokartta
    colors = ['blue', 'green', 'gray', 'white', 'yellow']
    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(colors)
    
    axes[0,0].imshow(data1, cmap=custom_cmap, vmin=0, vmax=4)
    axes[0,0].set_title('Alkuperäiset maastotyypit')
    
    # Perus DEM
    axes[0,1].imshow(improved_dem, cmap='terrain')
    axes[0,1].set_title('Paranneltu DEM')
    
    # Vuoristokohina
    axes[0,2].imshow(mountains, cmap='plasma')
    axes[0,2].set_title('Vuoristojaksot')
    
    # Laattatektoniikka
    axes[1,0].imshow(plates, cmap='hot')
    axes[1,0].set_title('Laattatektoniikka-efekti')
    
    # Jäätikkövuoret
    axes[1,1].imshow(glacial_mts, cmap='viridis')
    axes[1,1].set_title('Jäätikkövuoret')
    
    # Lopullinen DEM värikartalla
    im = axes[1,2].imshow(improved_dem, cmap='gist_earth')
    axes[1,2].set_title('Lopullinen DEM (gist_earth)')
    plt.colorbar(im, ax=axes[1,2])
    
    plt.tight_layout()
    plt.show()
    
    # SUURENNUS bicubic-interpoloinnilla
    print("Suurennetaan bicubic-interpoloinnilla...")
    enlarged_dem = resize(improved_dem, 
                         (data1.shape[0] * 4, data1.shape[1] * 4), 
                         order=3,  # bicubic
                         mode='reflect',
                         anti_aliasing=True)
    
    # Tallenna suurennettu DEM
    dem_normalized = (enlarged_dem - enlarged_dem.min()) / (enlarged_dem.max() - enlarged_dem.min())
    dem_16bit = (dem_normalized * 65535).astype(np.uint16)
    
    result = Image.fromarray(dem_16bit)
    result.save('helliconia_improved_dem.png')
    print("Paranneltu DEM tallennettu: helliconia_improved_dem.png")
    
    # Näytä suurennettu tulos
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(improved_dem, cmap='terrain')
    plt.title('Alkuperäinen paranneltu DEM')
    
    plt.subplot(1, 2, 2)
    plt.imshow(enlarged_dem, cmap='terrain')
    plt.title('Suurennettu DEM (bicubic)')
    plt.tight_layout()
    plt.show()
    
    print(f"DEM-tilastot: Min={improved_dem.min():.3f}, Max={improved_dem.max():.3f}")

if __name__ == "__main__":
    main()
