#!/usr/bin/env python
"""tilestache-server.py will serve your cache.

This script is intended to be run directly from the command line.

It is intended for direct use only during development or for debugging TileStache.

For the proper way to configure TileStach for serving tiles see the docs at:

http://tilestache.org/doc/#serving-tiles

To use this built-in server, install werkzeug and then run tilestache-server.py:

    tilestache-server.py

By default the script looks for a config file named tilestache.cfg in the current directory and then serves tiles on http://127.0.0.1:8080/.

You can then open your browser and view a url like:

    http://localhost:8080/osm/0/0/0.png

The above layer of 'osm' (defined in the tilestache.cfg) will display an OpenStreetMap
tile proxied from http://tile.osm.org/0/0/0.png

Check tilestache-server.py --help to change these defaults.
"""


import TileStache

app = TileStache.WSGITileServer(config="tilestache.cfg", autoreload=False)


