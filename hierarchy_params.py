import pandas as pd
from data_classes import CaseInsensitiveDict
import yaml


def get_hierarchy_info(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    dataset = config.get('dataset')

    if dataset == 'RIO':
        hierarchy = {
            'FISH': None,
            'Rock': None,
            'Unk': None,
            'SHAD': None,
            # cyanobacterias
            'Cyanobacteria': None,
            'CIAN4': 'Cyanobacteria',
            'CIAN2': 'Cyanobacteria',
            # substrates
            'Substrates': None,
            'Sand': 'Substrates',
            'ARC': 'Substrates',
            # other invertebrates
            'Other invertebrates': None,
            'Sponges': 'Other invertebrates',
            'OSPO5': 'Sponges',
            'PLACO': 'Sponges',
            'INC3': 'Sponges',
            'OASC': 'Other invertebrates',
            'ANM': 'Other invertebrates',
            # algaes
            'Algae': None,
            # non calcified
            'Non Calcified': 'Algae',
            'PTR': 'Non Calcified',
            'COD': 'Non Calcified',
            'WRAN': 'Non Calcified',
            'CLAD': 'Non Calcified',
            'OMFL1': 'Non Calcified',
            'HYP': 'Non Calcified',
            'OMCO2': 'Non Calcified',
            'SARG': 'Non Calcified',
            'VALV': 'Non Calcified',
            'LAU': 'Non Calcified',
            'Gelidiaceae': 'Non Calcified',
            'GLD': 'Gelidiaceae',
            'Gespp.': 'Gelidiaceae',
            'Caulerpa': 'Non Calcified',
            'CAU': 'Caulerpa',
            'CAU2': 'Caulerpa',
            'CAUC': 'Caulerpa',
            'Dictyopteris': 'Non Calcified',
            'DICMER': 'Dictyopteris',
            'DICP': 'Dictyopteris',
            'DICPP': 'Dictyopteris',
            'DICPD': 'Dictyopteris',
            'DICPJM': 'Dictyopteris',
            'Dictyota': 'Non Calcified',
            'DICTC': 'Dictyota',
            'DICTM': 'Dictyota',
            'CCV': 'Dictyota',
            'DICTP': 'Dictyota',
            'DICT4': 'Dictyota',
            'Turf': 'Non Calcified',
            'TFL': 'Turf',
            'Turf_sand': 'Turf',
            'TCC': 'Turf',
            # calcified
            'Calcified': 'Algae',
            'CNA/CC': 'Calcified',
            'OMCA': 'Calcified',
            'A/J': 'Calcified',
            # corals
            'Corals': None,
            'C': 'Corals',
            'Soft': 'Corals',
            'Zoanthidae': 'Soft',
            'Palythoa': 'Zoanthidae',
            'PAC': 'Palythoa',
            'PAL': 'Palythoa',
            'PRPV': 'Palythoa',
            'Zoanthus': 'Zoanthidae',
            'ZO': 'Zoanthus',
            'ZOS': 'Zoanthus',
            'Octocoral': 'Soft',
            'Plexauridae': 'Octocoral',
            'PLE': 'Plexauridae',
            'PLG': 'Plexauridae',

            'Hydro-corals': 'Corals',
            'Millepora': 'Hydro-corals',
            'MIL': 'Millepora',
            'MIA': 'Millepora',

            'Hard': 'Corals',
            'Favia': 'Hard',
            'FAV': 'Favia',
            'FAL': 'Favia',
            'Porites': 'Hard',
            'POA': 'Porites',
            'Siderastrea': 'Hard',
            'SID': 'Siderastrea',
            'SIDS': 'Siderastrea',
            'Tubastrea': 'Hard',
            'TUB': 'Tubastrea',
            'Mussimila': 'Hard',
            'MSP': 'Mussimila',
            'Agaricia': 'Hard',
            'AGF': 'Agaricia',
            'AGH': 'Agaricia',

            'Bleached': 'Corals',
            'B_HC': 'Bleached',
            'SINV_BLC': 'Bleached',
            'BL': 'Bleached',
            'D_coral': 'Bleached',
            'RD_HC': 'Bleached'
        }


        labelset_file = config.get('labelset_file')
        labelset_df = pd.read_csv(labelset_file)
        label_to_id = {label: label_id for label_id, label in zip(labelset_df["Label ID"], labelset_df["Short Code"])}
        label_to_line_number = {label: index for index, label in enumerate(labelset_df["Short Code"])}

        short_codes_unique = labelset_df["Short Code"].unique()
        index_list = list(range(len(short_codes_unique)))

        code_label_full_name = {
            'PAL': 'Palythoa spp.',
            'Sand': 'Sand',
            'Turf_sand': 'Turf and sand',
            'TFL': 'Turf Filamenteous',
            'CNA/CC': 'Calcifying calcareous crustose algae: DHC',
            'DICT4': 'Dictyota spp.',
            'ZOS': 'Zoanthus Sociatus',
            'TCC': 'Calcareous Turf',
            'SIDS': 'Siderastrea Stellata',
            'DICMER': 'Dictyota Mertensii',
            'SHAD': 'Shadow',
            'DICTC': 'Dictyota Ciliolata',
            'CIAN4': 'Cyanobacteria films',
            'Unk': 'Unknown',
            'SINV_BLC': 'Soft Coral Bleached',
            'POA': 'Porites Astreoides',
            'BL': 'Bleached Coral Point',
            'Rock': 'Rock',
            'DICPD': 'Dictyopteris Delicatula',
            'DICP': 'Dictyopteris spp.',
            'LAU': 'Laurencia spp.',
            'WRAN': 'Wrangelia',
            'ARC': 'Substrate: Unconsolidated (soft)',
            'MIL': 'Millepora spp.',
            'FAV': 'Favia Gravida',
            'CIAN2': 'Cyanobacteria',
            'ZOA': 'Zoanthus spp.',
            'Gespp.': 'Gelidiella spp.',
            'MIA': 'Millepora Alcicornis',
            'DICPP': 'Dictyopteris Plagiogramma',
            'RD_HC': 'Recent Dead Coral',
            'SARG': 'Sargassum',
            'PRPV': 'Protopalythoa Variabilis',
            'SID': 'Siderastrea spp.',
            'HYP': 'Hypnea Musciformis',
            'OMCO2': 'Leathery Macrophytes: Other',
            'A/J': 'Amphiroa spp.',
            'PLG': 'Plexaurella Grandiflora',
            'PAC': 'Palythoa Caribaeorum',
            'OSPO5': 'Sponge',
            'VALV': 'Valonia Ventricosa',
            'ANM': 'Anemone',
            'DICTM': 'Dictyota Menstrualis',
            'TUB': 'Tubastrea',
            'C': 'Coral',
            'GLD': 'Gelidium spp.',
            'FISH': 'Fish',
            'D_coral': 'Dead Coral',
            'INC3': 'Encrusting sponge',
            'DICPJM': 'Dictyopteris Jamaicensis',
            'CAU': 'Caulerpa',
            'AGH': 'Agaricia Humilis',
            'OMCA': 'Macroalgae: Articulated calcareous',
            'PLACO': ' Placospongia'
        }
    elif dataset == 'MLC':
        hierarchy = {
            'Corals': None,
            'Soft': 'Corals',
            'Hard': 'Corals',
            'Sand': None,
            'Algae': None,
            'Macro': 'Algae',
            'Turf': 'Algae',
            'Cca': 'Algae',
            'Acrop': 'Hard',
            'Acan': 'Hard',
            'Astreo': 'Hard',
            'Cypha': 'Hard',
            'Favia': 'Hard',
            'Fung': 'Hard',
            'Gardin': 'Hard',
            'Herpo': 'Hard',
            'Lepta': 'Hard',
            'Lepto': 'Hard',
            'Lobo': 'Hard',
            'Monta': 'Hard',
            'Monti': 'Hard',
            'Pachy': 'Hard',
            'Pavon': 'Hard',
            'Pocill': 'Hard',
            'Porite': 'Hard',
            'Porit': 'Porite',
            'P. Rus': 'Porite',
            'P. Irr': 'Porite',
            'P mass': 'Porite',
            'Psam': 'Hard',
            'Sando': 'Hard',
            'Stylo': 'Hard',
            'Tuba': 'Hard',
            'Mille': 'Soft',
            'Off': None
        }
        label_to_line_number = CaseInsensitiveDict({
            'CCA': 0,
            'Porit': 1,
            'Macro': 2,
            'Sand': 3,
            'Off': 4,
            'Turf': 5,
            'Monti': 6,
            'Acrop': 7,
            'Pavon': 8,
            'Pocill': 9,
            'Monta': 10,
            'Lepto': 11,
            'Lobo': 12,
            'Pachy': 13,
            'Acan': 14,
            'Astreo': 15,
            'Cypha': 16,
            'Favia': 17,
            'Fung': 18,
            'Gardin': 19,
            'Herpo': 20,
            'Psam': 21,
            'Sando': 22,
            'Stylo': 23,
            'Tuba': 24,
            'Soft': 25,
            'Mille': 26,
            'Lepta': 27,
            'P. Rus': 28,
            'P mass': 29,
            'P. Irr': 30,
            'Null': 31
        })
        code_label_full_name = {}
    elif dataset == 'TasCPC':
        hierarchy = {
            'Biota': None,
            'Sponges': 'Biota',
            'Echinoderms': 'Biota',
            'Cnidaria': 'Biota',
            'Bryozoa': 'Biota',
            'Algae': 'Biota',

            'Canopy': 'Algae',
            'ErectBranching': 'Algae',
            'Crustose': 'Algae',

            'ECK': 'Canopy',
            'BOCF': 'Canopy',
            'PHY': 'Canopy',

            'SAR': 'ErectBranching',
            'THAM': 'ErectBranching',

            'ECOR': 'Crustose',
            'SOND': 'Crustose',

            'BUST': 'Algae',
            'COD': 'Algae',
            'CAL': 'Algae',
            'UNA': 'Algae',
            'DRIFT': 'Algae',
            'RFOL': 'Algae',
            'GO': 'Algae',
            'TURF': 'Algae',

            # Bryozoa
            'BRY1': 'Bryozoa',
            'BRY2': 'Bryozoa',
            'BRY3': 'Bryozoa',
            'BRY4': 'Bryozoa',
            'BRY5': 'Bryozoa',
            'BRY6': 'Bryozoa',
            'BRY7': 'Bryozoa',
            'BRY8': 'Bryozoa',

            # Cnidaria
            'C2S': 'Cnidaria',
            'C3BR': 'Cnidaria',
            'C4BR': 'Cnidaria',
            'C5OS': 'Cnidaria',
            'C6SB': 'Cnidaria',
            'C7SY': 'Cnidaria',
            'G1P': 'Cnidaria',
            'G2R': 'Cnidaria',
            'HYD1': 'Cnidaria',
            'SPEN': 'Cnidaria',
            'SW1': 'Cnidaria',
            'PARA1': 'Cnidaria',
            'ANEM1': 'Cnidaria',

            # Echinoderms
            'CENOL': 'Echinoderms',
            'HOL': 'Echinoderms',
            'SS': 'Echinoderms',
            'BS': 'Echinoderms',
            'URCH': 'Echinoderms',

            # Molluscs
            'Mollusc': 'Biota',
            'ABAL': 'Mollusc',
            'SCAL': 'Mollusc',
            'MOL': 'Mollusc',

            # Ascidians
            'Ascidian': 'Biota',
            'A1Cl': 'Ascidian',
            'A2Cl': 'Ascidian',
            'AS3O': 'Ascidian',
            'A4Sy': 'Ascidian',
            'A5Sol': 'Ascidian',
            'A6R': 'Ascidian',
            'A7Sol': 'Ascidian',
            'A8O': 'Ascidian',

            # Arborescent Sponges
            'Arborescent': 'Sponges',
            'A1WF': 'Arborescent',  # Arborescent 1 white flat
            'A2GR': 'Arborescent',  # Arborescent 2 grey
            'A3PT': 'Arborescent',  # Arborescent 3 purple thin
            'A4OF': 'Arborescent',  # Arborescent 4 orange flat
            'A5W': 'Arborescent',  # Arborescent 5 white
            'A6Y': 'Arborescent',  # Arborescent 6 yellow
            'A7PU': 'Arborescent',  # Arborescent 7 purple
            'A8T': 'Arborescent',  # Arborescent 8 tan
            'A9OT': 'Arborescent',  # Arborescent 9 orange thin
            'A10F': 'Arborescent',  # Arborescent 10 orange/brown fingers
            'A11OF': 'Arborescent',  # Arborescent 11 orange fan
            'A12BT': 'Arborescent',  # Arborescent 12 brown thorny
            'A13O': 'Arborescent',  # Arborescent 13 orange
            'A14B': 'Arborescent',  # Arborescent 14 black
            'A15WS': 'Arborescent',  # Arborescent 15 white short
            'A16BT': 'Arborescent',  # Arborescent 16 brown thick

            # Cup Sponges
            'Cup': 'Sponges',
            'C1W': 'Cup',  # Cup 1 white
            'C2WF': 'Cup',  # Cup 2 white frilly
            'C3B': 'Cup',  # Cup 3 blue
            'C4BT': 'Cup',  # Cup 4 blue thick
            'C5R': 'Cup',  # Cup 5 red
            'C6PT': 'Cup',  # Cup 6 pink thick
            'C7LPFT': 'Cup',  # Cup 7 light pink flat thick
            'C8Y': 'Cup',  # Cup 8 yellow

            # Encrusting Sponges
            'Encrusting': 'Sponges',
            'E1OR': 'Encrusting',  # Encrusting 1 orange
            'E2OR': 'Encrusting',  # Encrusting 2 light orange
            'E3Y': 'Encrusting',  # Encrusting 3 yellow
            'E4BL': 'Encrusting',  # Encrusting 4 blue
            'E5BR': 'Encrusting',  # Encrusting 5 brown
            'E6WH': 'Encrusting',  # Encrusting 6 white
            'E7G': 'Encrusting',  # Encrusting 7 green

            # Fan Sponges
            'Fan': 'Sponges',
            'F1OR': 'Fan',  # Fan 1 orange
            'F2BR': 'Fan',  # Fan 2 brown
            'F3OF': 'Fan',  # Fan 3 orange flat
            'F4PI': 'Fan',  # Fan 4 pink
            'F5PE': 'Fan',  # Fan 5 peach
            'F6Y': 'Fan',  # Fan 6 yellow
            'F7ORT': 'Fan',  # Fan 7 orange thin blade
            'F8BT': 'Fan',  # Fan 8 blue thick
            'F9ORT': 'Fan',  # Fan 9 orange thick
            'F10': 'Fan',  # Fan 10 thick large oscules
            'F11PT': 'Fan',  # Fan 11 thick pink
            'F12BT': 'Fan',  # Fan 12 browm thin
            'F13OF': 'Fan',  # Fan 13 orange frilly
            'F14WT': 'Fan',  # Fan 14 white thin
            'F15OT': 'Fan',  # Fan 15 orange thorny

            # Globular Sponges
            'Globular': 'Sponges',
            'G1OR': 'Globular',  # Globular 1 orange Tethya like
            'G2WH': 'Globular',  # Globular 2 white Tethya like
            'G3BL': 'Globular',  # Globular 3 blue
            'G4O': 'Globular',  # Globular 4 orange

            # Lumpy Sponges
            'Lumpy': 'Sponges',
            'L1PS': 'Lumpy',  # Lumpy 1 purple stumps
            'L2O': 'Lumpy',  # Lumpy 2 orange
            'L3W': 'Lumpy',  # Lumpy 3 white
            'L4P': 'Lumpy',  # Lumpy 4 pink
            'L5Y': 'Lumpy',  # Lumpy 5 yellow

            # Massive Sponges
            'Massive': 'Sponges',
            'M1': 'Massive',  # Massive 1
            'M2': 'Massive',  # Massive 2
            'M3OR': 'Massive',  # Massive 3 orange
            'M4DON': 'Massive',  # Massive 4 donut
            'M5': 'Massive',  # Massive 5 fungi
            'M6VEL': 'Massive',  # Massive 6 velvet
            'M7BL': 'Massive',  # Massive 7 blue
            'M8': 'Massive',  # Massive 8
            'M9WH': 'Massive',  # Massive 9 white
            'M10BR': 'Massive',  # Massive 10 brown
            'M11WH': 'Massive',  # Massive 11 white holey
            'M12YP': 'Massive',  # Massive 12 yellow papillate
            'M13WP': 'Massive',  # Massive 13 white papillate
            'M14OR': 'Massive',  # Massive 14 orange shapeless
            'M15PS': 'Massive',  # Massive 15 shapeless
            'M16P': 'Massive',  # Massive 16 purple
            'M17WL': 'Massive',  # Massive 17 white lumpy
            'M18OH': 'Massive',  # Massive 18 orange holey
            'M19': 'Massive',  # Massive 19 yellow shapeless
            'M20P': 'Massive',  # Massive 20 pink

            # Papillate Sponges
            'Papillate': 'Sponges',
            'P1SU': 'Papillate',  # Papillate 1 Suberites like
            'P2Y': 'Papillate',  # Papillate 2 yellow
            'P3B': 'Papillate',  # Papillate 3 black
            'P4LO': 'Papillate',  # Papillate 4 light orange

            # Repent Sponges
            'Repent': 'Sponges',
            'R1B': 'Repent',  # Repent 1 brown

            # Tubular Sponges
            'Tubular': 'Sponges',
            'T1PI': 'Tubular',  # Tubular 1 pink
            'T2': 'Tubular',  # Tubular 2 apricot
            'T3WC': 'Tubular',  # Tubular 3 white colony
            'T4T': 'Tubular',  # Tubular 4 tan
            'T5': 'Tubular',  # Tubular 5 tan singular
            'T6WT': 'Tubular',  # Tubular 6 white thorny
            'T7PT': 'Tubular',  # Tubular 7 pink thorny
            'T8OR': 'Tubular',  # Tubular 8 orange
            'T9PP': 'Tubular',  # Tubular 9 pink small oscules
            'T10OT': 'Tubular',  # Tubular 10 orange thorny
            'T11B': 'Tubular',  # Tubular 11 blue
            'T12PO': 'Tubular',  # Tubular 12 pale orange
            'T13': 'Tubular',  # Tubular 13 Sycon
            'T14S': 'Tubular',  # Tubular 14 solitary

            'FISH': 'Biota',
            'TW1': 'Biota',
            'BIOT': 'Biota',

            'Physical': None,
            'Hard': 'Physical',
            'Soft': 'Physical',

            'ROCK': 'Hard',
            'SAND': 'Soft',

            'MATR': 'Biota',
            'UNK': 'Biota',
            'BRUB': 'Biota',
            'UNID': 'Biota',
            'NZSS': 'Biota',

            'Off': None,
            'Blank': None,
            'UNS': None

        }
        label_to_line_number = assign_leaf_numbers(hierarchy)
        code_label_full_name = {}
    line_number_to_label = {v: k for k, v in label_to_line_number.items()}

    return hierarchy, label_to_line_number, code_label_full_name, line_number_to_label
def assign_leaf_numbers(hierarchy):
    # Trouver les leaf nodes (nœuds sans enfants)
    children = set(hierarchy.keys())
    parents = set(hierarchy.values()) - {None}
    leaf_nodes = children - parents

    # Assigner un numéro à chaque leaf node
    leaf_dict = {}
    for i, node in enumerate(sorted(leaf_nodes), 1):
        leaf_dict[node] = i

    return leaf_dict
