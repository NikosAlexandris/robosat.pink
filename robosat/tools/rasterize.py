import argparse
import collections
import json
import os
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm

import mercantile
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from rasterio.warp import transform
from supermercado import burntiles
from shapely.geometry import shape, mapping

from robosat.config import load_config
from robosat.colors import make_palette
from robosat.tiles import tiles_from_csv
from robosat.utils import leaflet


def add_parser(subparser):
    parser = subparser.add_parser(
        "rasterize", help="rasterize features to label masks", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("features", type=str, help="path to GeoJSON features file")
    parser.add_argument("tiles", type=str, help="path to .csv tiles file")
    parser.add_argument("out", type=str, help="directory to write converted images")
    parser.add_argument("--dataset", type=str, required=True, help="path to dataset configuration file")
    parser.add_argument("--zoom", type=int, required=True, help="zoom level of tiles")
    parser.add_argument("--size", type=int, default=512, help="size of rasterized image tiles in pixels")
    parser.add_argument("--leaflet", type=str, help="leaflet client base url")

    parser.set_defaults(func=main)


def feature_to_mercator(feature):
    """Convert polygon feature coords to 3857.

    Args:
      feature: geojson feature to convert to mercator geometry.
    """
    # Ref: https://gist.github.com/dnomadb/5cbc116aacc352c7126e779c29ab7abe

    if feature["geometry"]["type"] == "Polygon":
        xys = (zip(*ring) for ring in feature["geometry"]["coordinates"])
        xys = (list(zip(*transform(CRS.from_epsg(4326), CRS.from_epsg(3857), *xy))) for xy in xys)

        yield {"coordinates": list(xys), "type": "Polygon"}


def burn(tile, features, size, burn_value=1):
    """Burn tile with features.

    Args:
      tile: the mercantile tile to burn.
      features: the geojson features to burn.
      size: the size of burned image.
      burn_value: the value you want in the output raster where a shape exists

    Returns:
      image: rasterized file of size with features burned.
    """

    shapes = ((geometry, burn_value) for feature in features for geometry in feature_to_mercator(feature))

    bounds = mercantile.xy_bounds(tile)
    transform = from_bounds(*bounds, size, size)

    result = rasterize(shapes, out_shape=(size, size), transform=transform)
    return Image.fromarray(result, mode="P")


def main(args):
    dataset = load_config(args.dataset)

    classes = dataset["common"]["classes"]
    colors = dataset["common"]["colors"]
    assert len(classes) == len(colors), "classes and colors coincide"

    assert len(colors) == 2, "only binary models supported right now"
    bg = colors[0]
    fg = colors[1]

    os.makedirs(args.out, exist_ok=True)

    # We can only rasterize all tiles at a single zoom.
    assert all(tile.z == args.zoom for tile in tiles_from_csv(args.tiles))

    with open(args.features) as f:
        fc = json.load(f)

    # Find all tiles the features cover and make a map object for quick lookup.
    feature_map = collections.defaultdict(list)
    for i, feature in enumerate(tqdm(fc["features"], ascii=True, unit="feature")):

        if feature["geometry"]["type"] == "Polygon":
            feature["geometry"]["coordinates"] = [feature["geometry"]["coordinates"]]
            feature["geometry"]["type"] = "MultiPolygon"

        for polygon in shape(feature["geometry"]):
            simple_feature = {"type": "feature", "geometry": mapping(polygon)}
            try:
                for tile in burntiles.burn([simple_feature], zoom=args.zoom):
                    feature_map[mercantile.Tile(*tile)].append(simple_feature)
            except ValueError as e:
                print("Warning: invalid feature {}, skipping".format(i), file=sys.stderr)
                continue

    # Burn features to tiles and write to a slippy map directory.
    for tile in tqdm(list(tiles_from_csv(args.tiles)), ascii=True, unit="tile"):
        if tile in feature_map:
            out = burn(tile, feature_map[tile], args.size)
        else:
            out = Image.fromarray(np.zeros(shape=(args.size, args.size)).astype(int), mode="P")

        palette = make_palette(bg, fg)
        out.putpalette(palette)

        out_path = os.path.join(args.out, str(tile.z), str(tile.x))
        os.makedirs(out_path, exist_ok=True)

        out.save(os.path.join(out_path, "{}.png".format(tile.y)), optimize=True)

    if args.leaflet:
        leaflet(args.out, args.leaflet, args.tiles, ".png")
