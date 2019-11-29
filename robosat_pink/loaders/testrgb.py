"""PyTorch-compatible datasets. Cf: https://pytorch.org/docs/stable/data.html """

import os
import numpy as np
import pathlib
import torch.utils.data

from robosat_pink.tiles import (
    tiles_from_dir,
    tile_image_from_file,
    tile_label_from_file,
    tile_translate_from_file,
    images_from_dir,
    index_labels,
    enumerate_image_paths,
    enumerate_image_labels
)
from robosat_pink.da.core import to_normalized_tensor




class TestRGB(torch.utils.data.Dataset):
    def __init__(self, config, ts, root, cover, mode):
        super().__init__()

        self.root = os.path.expanduser(root)
        self.config = config
        self.mode = mode
        self.cover = cover

        assert mode in ["train", "predict", "predict_translate"]
        xyz_translate = True if mode == "predict_translate" else False

        num_channels = 0
        self.tiles = {}
        for channel in config["channels"]:
            path = os.path.join(self.root, channel["name"])
            # Fill tiles dict without any GIS stuff
            self.tiles[channel["name"]] = [
              # (tile, path) for tile, path in enumerate_image_paths(path, cover=cover, xyz_path=True, xyz_translate=xyz_translate)
              (tile, path) for tile, path in enumerate_image_paths(path, 'jpg')
              ]
            self.tiles[channel["name"]].sort(key=lambda tile: tile[0])
            num_channels += len(channel["bands"])

        self.shape_in = (num_channels,) + ts  # C,W,H
        self.shape_out = (len(config["classes"]),) + ts  # C,W,H

        if self.mode == "train":
            path = os.path.join(self.root, channel["name"])
            #MODIFIED Change list of label masks to single class name.
            self.tiles["labels"] = [(number, label) for number, label in enumerate_image_labels(path, 'jpg')]
            self.tiles["labels"].sort(key=lambda tile: tile[0])

        assert len(self.tiles), "Empty Dataset"


    def __len__(self):
        return len(self.tiles[self.config["channels"][0]["name"]])

    def __getitem__(self, i):

        tile = None
        mask = None
        image = None

        for channel in self.config["channels"]:

            image_channel = None
            bands = None if not channel["bands"] else channel["bands"]

            if tile is None:
                tile, path = self.tiles[channel["name"]][i]
            else:
                assert tile == self.tiles[channel["name"]][i][0], "Dataset channel inconsistency"
                tile, path = self.tiles[channel["name"]][i]

            image_channel = tile_image_from_file(path, bands)
            assert image_channel is not None, "Dataset channel {} not retrieved: {}".format(channel["name"], path)

            image = np.concatenate((image, image_channel), axis=2) if image is not None else image_channel

        if self.mode == "train":
            label = np.full((image.shape[0],image.shape[1]),self.tiles["labels"][i][1])
            #label = np.full()
            #label = self.tiles["labels"][i][1]
            assert label is not None, "Dataset label not retrieved"
            
            image, mask = to_normalized_tensor(self.config, self.shape_in[1:3], "train", image, mask=label)
            return image, mask, tile

        if self.mode in ["predict", "predict_translate"]:
            image = to_normalized_tensor(self.config, self.shape_in[1:3], "predict", image)
            return image,path.name,tile