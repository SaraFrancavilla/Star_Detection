import utils
import os

# Usa il percorso della directory dello script
script_dir = os.path.dirname(os.path.abspath(__file__))
coord_file = os.path.join(script_dir, 'output', 'output', 'coordinates.csv')
lost_file = os.path.join(script_dir, 'output', 'output', 'LOSTcoordinates.csv')

print(f"Looking for files in: {os.path.dirname(coord_file)}")

coord = utils.load_csv_coordinates(coord_file)
print("Loaded coordinates:", len(coord), "labels")

LOSTcoord = utils.load_csv_coordinates(lost_file)
print("Loaded LOST coordinates:", len(LOSTcoord), "labels")

# Inizializza accumulatori
stats = {
    '0': {'lost': {'n_matched': 0, 'ratio': 0}, 'ours': {'n_matched': 0, 'ratio': 0}},
    '1': {'lost': {'n_matched': 0, 'ratio': 0}, 'ours': {'n_matched': 0, 'ratio': 0}},
    '5': {'lost': {'n_matched': 0, 'ratio': 0}, 'ours': {'n_matched': 0, 'ratio': 0}},
    '6': {'lost': {'n_matched': 0, 'ratio': 0}, 'ours': {'n_matched': 0, 'ratio': 0}},
    'noise': {'lost': {'n_matched': 0, 'ratio': 0}, 'ours': {'n_matched': 0, 'ratio': 0}},
}
counts = {cat: 0 for cat in stats}

groundtruth = None

# Naviga tra le keys di LOST
for key in LOSTcoord.keys():
    category = key[-1]  # ultimo carattere della label
    
    # Determina la categoria
    if category == "0":
        cat_key = '0'
        groundtruth = LOSTcoord[key]  # salva ground truth
    elif category == "1":
        cat_key = '1'
    elif category == "5":
        cat_key = '5'
    elif category == "6":
        cat_key = '6'
    else:  # 2, 3, 4
        cat_key = 'noise'
    
    if groundtruth is None:
        continue
    
    # Confronta LOST con ground truth
    _, lost_n_matched, _ = utils.match_centroids(groundtruth, LOSTcoord[key], 3)
    
    # Confronta i nostri centroidi con ground truth
    _, our_n_matched, _ = utils.match_centroids(groundtruth, coord[key], 3)
    
    # Accumula risultati
    stats[cat_key]['lost']['n_matched'] += lost_n_matched
    stats[cat_key]['ours']['n_matched'] += our_n_matched
    counts[cat_key] += 1

# Calcola medie
for cat in stats:
    if counts[cat] > 0:
        stats[cat]['lost']['n_matched'] /= counts[cat]
        stats[cat]['ours']['n_matched'] /= counts[cat]

print("\n=== STATISTICS ===")
print(f"{'CATEGORY':<15} {'LOST':<20} {'OUR FUNCTION':<20}")
print("-" * 55)
print(f"{'IMAGE 0':<15} {'Perfect':<20} {stats['0']['ours']['n_matched']:.1f}")
print(f"{'IMAGE 1 (ATM)':<15} {stats['1']['lost']['n_matched']:.1f}{'':>8} {stats['1']['ours']['n_matched']:.1f}")
print(f"{'IMAGE 2,3,4 (NOISE)':<15} {stats['noise']['lost']['n_matched']:.1f}{'':>8} {stats['noise']['ours']['n_matched']:.1f}")
print(f"{'IMAGE 5 (BLUR)':<15} {stats['5']['lost']['n_matched']:.1f}{'':>8} {stats['5']['ours']['n_matched']:.1f}")
print(f"{'IMAGE 6 (BLUR+NOISE)':<15} {stats['6']['lost']['n_matched']:.1f}{'':>8} {stats['6']['ours']['n_matched']:.1f}")
    




