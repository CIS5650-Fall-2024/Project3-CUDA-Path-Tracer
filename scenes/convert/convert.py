import json
import sys

if __name__ == '__main__':
    filename = sys.argv[1]
    with open("../" + filename) as file:
        contents = file.read()

    in_json = json.loads(contents)["frames"][0]["scene"]
    out_json = {}

    in_camera = in_json["camera"]
    out_camera = {
        "RES": [in_camera["width"], in_camera["height"]],
        "FOVY": in_camera["fov"],
        "ITERATIONS": 5000,
        "DEPTH": 8,
        "FILE": filename.split(".")[0],
        "EYE": in_camera["eye"],
        "LOOKAT": in_camera["target"],
        "UP": in_camera["worldUp"]
    }
    out_json["Camera"] = out_camera

    out_objects = []
    def shape_convert(in_shape):
        if in_shape == "SquarePlane":
            return "square"
        elif in_shape == "Sphere":
            return "sphere"

    for primitive in in_json["primitives"]:
        in_transform = primitive["transform"]
        out_objects.append({
            "TYPE": shape_convert(primitive["shape"]),
            "MATERIAL": primitive["material"],
            "TRANS": in_transform["translate"],
            "ROTAT": in_transform["rotate"],
            "SCALE": in_transform["scale"]
        })

    light_materials = []
    for light in in_json["lights"]:
        in_transform = light["transform"]
        
        material = {
            "TYPE": "Emitting",
            "RGB": light["lightColor"],
            "EMITTANCE": light["intensity"]
        }

        if material in light_materials:
            material_index = light_materials.index(material)
        else:
            material_index = len(light_materials)
            light_materials.append(material)

        out_objects.append({
            "TYPE": shape_convert(primitive["shape"]),
            "MATERIAL": "light_" + str(material_index),
            "TRANS": in_transform["translate"],
            "ROTAT": in_transform["rotate"],
            "SCALE": in_transform["scale"]
        })
    out_json["Objects"] = out_objects

    out_materials = {}
    for (index, material) in enumerate(light_materials) :
        out_materials["light_" + str(index)] = material

    for material in in_json["materials"]:
        out_materials[material["name"]] = material

    out_json["Materials"] = out_materials
    out_json = json.dumps(out_json)

    print(out_json)