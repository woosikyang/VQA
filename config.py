import os
data_path = 'data'
data_glove_path = os.path.join(data_path,'glove')
saved_data_path = '/media/woosik/7878282F7827EA98/Users/yang/Desktop/논문/연습/data'

COCO_CATEGORIES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]



vg_sgg_dicts_json = {"object_count": {"wire": 4017, "pant": 13147, "boot": 2714, "laptop": 3077, "paper": 4714, "shoe": 12419, "railing": 3049, "chair": 11936, "windshield": 3177, "kite": 3706, "tire": 6353, "cup": 4305, "bench": 5589, "window": 42466, "bike": 4884, "board": 4649, "orange": 2536, "hat": 8366, "finger": 3059, "plate": 11632, "woman": 26910, "handle": 7996, "animal": 3611, "food": 6057, "bear": 3487, "wave": 5938, "vegetable": 2264, "giraffe": 4953, "desk": 2969, "lady": 3335, "towel": 3231, "shelf": 5659, "bag": 7391, "fork": 2470, "nose": 6035, "rock": 7922, "sidewalk": 9478, "motorcycle": 4537, "sneaker": 2212, "fence": 12027, "people": 14415, "house": 5006, "sign": 23499, "hair": 17422, "street": 10996, "zebra": 4449, "racket": 2610, "logo": 4606, "girl": 7543, "arm": 9951, "flower": 8228, "leaf": 13984, "clock": 4877, "hill": 4457, "bird": 4339, "umbrella": 6389, "bed": 3296, "leg": 22335, "screen": 2498, "men": 2037, "sink": 3456, "trunk": 5191, "post": 6163, "tower": 2957, "box": 5467, "boy": 8125, "cow": 4531, "skateboard": 3898, "fruit": 2366, "stand": 2934, "pillow": 5499, "ski": 3529, "sock": 3159, "pot": 2726, "number": 3294, "pole": 21205, "table": 19064, "boat": 6164, "sheep": 4117, "horse": 5315, "eye": 5619, "seat": 4360, "tail": 8821, "vehicle": 3943, "curtain": 3241, "kid": 2264, "banana": 4566, "engine": 2673, "head": 21376, "door": 13354, "bus": 5118, "cabinet": 4476, "glass": 10951, "flag": 3238, "train": 6411, "child": 3799, "ear": 12069, "surfboard": 4117, "room": 2552, "drawer": 2612, "car": 17352, "cap": 4130, "tree": 49902, "roof": 6453, "cat": 3720, "coat": 4285, "skier": 2567, "toilet": 2702, "player": 5567, "guy": 2823, "airplane": 2633, "glove": 4169, "mountain": 4894, "shirt": 33920, "paw": 2595, "bowl": 5152, "snow": 9437, "lamp": 4054, "book": 4297, "branch": 6161, "elephant": 4870, "tile": 7619, "tie": 2654, "beach": 3693, "pizza": 3678, "wheel": 9236, "plant": 6478, "helmet": 6306, "track": 7054, "hand": 17497, "plane": 5058, "mouth": 3630, "letter": 6630, "vase": 2853, "man": 54659, "building": 31805, "short": 7807, "neck": 4545, "phone": 2645, "light": 14182, "counter": 4632, "dog": 4651, "face": 8419, "jacket": 10463, "person": 41278, "truck": 4624, "bottle": 6246, "basket": 2666, "jean": 5535, "wing": 4455}, "idx_to_label": {"1": "airplane", "2": "animal", "3": "arm", "4": "bag", "5": "banana", "6": "basket", "7": "beach", "8": "bear", "9": "bed", "10": "bench", "11": "bike", "12": "bird", "13": "board", "14": "boat", "15": "book", "16": "boot", "17": "bottle", "18": "bowl", "19": "box", "20": "boy", "21": "branch", "22": "building", "23": "bus", "24": "cabinet", "25": "cap", "26": "car", "27": "cat", "28": "chair", "29": "child", "30": "clock", "31": "coat", "32": "counter", "33": "cow", "34": "cup", "35": "curtain", "36": "desk", "37": "dog", "38": "door", "39": "drawer", "40": "ear", "41": "elephant", "42": "engine", "43": "eye", "44": "face", "45": "fence", "46": "finger", "47": "flag", "48": "flower", "49": "food", "50": "fork", "51": "fruit", "52": "giraffe", "53": "girl", "54": "glass", "55": "glove", "56": "guy", "57": "hair", "58": "hand", "59": "handle", "60": "hat", "61": "head", "62": "helmet", "63": "hill", "64": "horse", "65": "house", "66": "jacket", "67": "jean", "68": "kid", "69": "kite", "70": "lady", "71": "lamp", "72": "laptop", "73": "leaf", "74": "leg", "75": "letter", "76": "light", "77": "logo", "78": "man", "79": "men", "80": "motorcycle", "81": "mountain", "82": "mouth", "83": "neck", "84": "nose", "85": "number", "86": "orange", "87": "pant", "88": "paper", "89": "paw", "90": "people", "91": "person", "92": "phone", "93": "pillow", "94": "pizza", "95": "plane", "96": "plant", "97": "plate", "98": "player", "99": "pole", "100": "post", "101": "pot", "102": "racket", "103": "railing", "104": "rock", "105": "roof", "106": "room", "107": "screen", "108": "seat", "109": "sheep", "110": "shelf", "111": "shirt", "112": "shoe", "113": "short", "114": "sidewalk", "115": "sign", "116": "sink", "117": "skateboard", "118": "ski", "119": "skier", "120": "sneaker", "121": "snow", "122": "sock", "123": "stand", "124": "street", "125": "surfboard", "126": "table", "127": "tail", "128": "tie", "129": "tile", "130": "tire", "131": "toilet", "132": "towel", "133": "tower", "134": "track", "135": "train", "136": "tree", "137": "truck", "138": "trunk", "139": "umbrella", "140": "vase", "141": "vegetable", "142": "vehicle", "143": "wave", "144": "wheel", "145": "window", "146": "windshield", "147": "wing", "148": "wire", "149": "woman", "150": "zebra"}, "predicate_to_idx": {"and": 5, "says": 39, "belonging to": 9, "over": 33, "parked on": 35, "growing on": 18, "standing on": 41, "made of": 27, "attached to": 7, "at": 6, "in": 22, "hanging from": 19, "wears": 49, "in front of": 23, "from": 17, "for": 16, "watching": 47, "lying on": 26, "to": 42, "behind": 8, "flying in": 15, "looking at": 25, "on back of": 32, "holding": 21, "between": 10, "laying on": 24, "riding": 38, "has": 20, "across": 2, "wearing": 48, "walking on": 46, "eating": 14, "above": 1, "part of": 36, "walking in": 45, "sitting on": 40, "under": 43, "covered in": 12, "carrying": 11, "using": 44, "along": 4, "with": 50, "on": 31, "covering": 13, "of": 30, "against": 3, "playing": 37, "near": 29, "painted on": 34, "mounted on": 28}, "predicate_count": {"and": 3477, "says": 2241, "belonging to": 3288, "over": 9317, "parked on": 2721, "growing on": 1853, "standing on": 14185, "made of": 2380, "part of": 2065, "attached to": 10190, "at": 9903, "in": 251756, "hanging from": 9894, "wears": 15457, "in front of": 13715, "from": 2945, "for": 9145, "lying on": 1869, "to": 2517, "behind": 41356, "flying in": 1973, "looking at": 3083, "on back of": 1914, "holding": 42722, "under": 22596, "laying on": 3739, "riding": 8856, "has": 277936, "across": 1996, "wearing": 136099, "walking on": 4613, "eating": 4688, "above": 47341, "watching": 3490, "walking in": 1740, "sitting on": 18643, "between": 3411, "covered in": 2312, "carrying": 5213, "using": 1925, "along": 3624, "with": 66425, "on": 712409, "covering": 3806, "of": 146339, "against": 3092, "mounted on": 2253, "near": 96589, "painted on": 3095, "playing": 3810}, "idx_to_predicate": {"1": "above", "2": "across", "3": "against", "4": "along", "5": "and", "6": "at", "7": "attached to", "8": "behind", "9": "belonging to", "10": "between", "11": "carrying", "12": "covered in", "13": "covering", "14": "eating", "15": "flying in", "16": "for", "17": "from", "18": "growing on", "19": "hanging from", "20": "has", "21": "holding", "22": "in", "23": "in front of", "24": "laying on", "25": "looking at", "26": "lying on", "27": "made of", "28": "mounted on", "29": "near", "30": "of", "31": "on", "32": "on back of", "33": "over", "34": "painted on", "35": "parked on", "36": "part of", "37": "playing", "38": "riding", "39": "says", "40": "sitting on", "41": "standing on", "42": "to", "43": "under", "44": "using", "45": "walking in", "46": "walking on", "47": "watching", "48": "wearing", "49": "wears", "50": "with"}, "label_to_idx": {"kite": 69, "pant": 87, "bowl": 18, "laptop": 72, "paper": 88, "motorcycle": 80, "railing": 103, "chair": 28, "windshield": 146, "tire": 130, "cup": 34, "bench": 10, "tail": 127, "bike": 11, "board": 13, "orange": 86, "hat": 60, "finger": 46, "plate": 97, "woman": 149, "handle": 59, "branch": 21, "food": 49, "bear": 8, "vase": 140, "vegetable": 141, "giraffe": 52, "desk": 36, "lady": 70, "towel": 132, "glove": 55, "bag": 4, "nose": 84, "rock": 104, "guy": 56, "shoe": 112, "sneaker": 120, "fence": 45, "people": 90, "house": 65, "seat": 108, "hair": 57, "street": 124, "roof": 105, "racket": 102, "logo": 77, "girl": 53, "arm": 3, "flower": 48, "leaf": 73, "clock": 30, "hill": 63, "bird": 12, "umbrella": 139, "leg": 74, "screen": 107, "men": 79, "sink": 116, "trunk": 138, "post": 100, "sidewalk": 114, "box": 19, "boy": 20, "cow": 33, "skateboard": 117, "plane": 95, "stand": 123, "pillow": 93, "ski": 118, "wire": 148, "toilet": 131, "pot": 101, "sign": 115, "number": 85, "pole": 99, "table": 126, "boat": 14, "sheep": 109, "horse": 64, "eye": 43, "sock": 122, "window": 145, "vehicle": 142, "curtain": 35, "kid": 68, "banana": 5, "engine": 42, "head": 61, "door": 38, "bus": 23, "cabinet": 24, "glass": 54, "flag": 47, "train": 135, "child": 29, "ear": 40, "surfboard": 125, "room": 106, "player": 98, "car": 26, "cap": 25, "tree": 136, "bed": 9, "cat": 27, "coat": 31, "skier": 119, "zebra": 150, "fork": 50, "drawer": 39, "airplane": 1, "helmet": 62, "shirt": 111, "paw": 89, "boot": 16, "snow": 121, "lamp": 71, "book": 15, "animal": 2, "elephant": 41, "tile": 129, "tie": 128, "beach": 7, "pizza": 94, "wheel": 144, "plant": 96, "tower": 133, "mountain": 81, "track": 134, "hand": 58, "fruit": 51, "mouth": 82, "letter": 75, "shelf": 110, "wave": 143, "man": 78, "building": 22, "short": 113, "neck": 83, "phone": 92, "light": 76, "counter": 32, "dog": 37, "face": 44, "jacket": 66, "person": 91, "truck": 137, "bottle": 17, "basket": 6, "jean": 67, "wing": 147}}