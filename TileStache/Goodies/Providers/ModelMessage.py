""" Provider that listens to model messages and renders the array of the last message


Sample configuration:
"waterlevel": {
    "provider": {
        "class": "TileStache.Goodies.Providers.ModelMessage:Provider",
        "kwargs": {
            "resample": "nearest", "port": 5556,
            "epsg":28992, "geotransform": [185564.5, 5, 0, 559069.5, 0, -5]
        }
    }
}

Valid values for resample are "cubic", "cubicspline", "linear", and "nearest".
The port is the zmq sockt to listen to for model messages.
The variable to render is inferred from the layer name

# Based on the GDAL example
"""
import logging
import datetime

import zmq
from zmq.eventloop import ioloop, zmqstream
ioloop.install()

from tornado.ioloop import IOLoop
import threading
import dateutil.parser
import numpy as np

try:
    from PIL import Image
except ImportError:
    import Image

try:
    import osgeo.gdal
    import osgeo.osr
except ImportError:
    # well it won't work but we can still make the documentation.
    pass

resamplings = {'cubic': osgeo.gdal.GRA_Cubic, 'cubicspline': osgeo.gdal.GRA_CubicSpline,
               'linear': osgeo.gdal.GRA_Bilinear, 'nearest': osgeo.gdal.GRA_NearestNeighbour}


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



def recv_array(socket, flags=0, copy=False, track=False):
    """receive a numpy array"""
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    buf = buffer(msg)
    A = np.frombuffer(buf, dtype=md['dtype'])
    A.reshape(md['shape'])
    return A, md

def make_listenener(ctx, port, data):
    """make a socket that replies to message with the grid"""
    subsock = ctx.socket(zmq.SUB)
    subsock.connect("tcp://localhost:{port}".format(port=port))
    subsock.setsockopt(zmq.SUBSCRIBE,'')
    def model_listener(socket, data):
        while True:
            arr, metadata = recv_array(socket)
            logger.info("got msg {}".format(metadata))
            data[metadata['name']] = arr
    thread = threading.Thread(target=model_listener, args=[subsock, data])
    thread.start()

class Provider(object):
    """
    Render the last model message
    """
    def __init__(self, layer, epsg, geotransform, resample='nearest', port=5556):
        self.layer = layer

        if resample not in resamplings:
            raise Exception('Resample must be "cubic", "linear", or "nearest", not: '+resample)
        self.resample = resamplings[resample]
        self.epsg = epsg
        self.geotransform = geotransform


        req_port = 5556
        sub_port = 5558
        logger.info("Connecting to grid at port {}".format(req_port))
        ctx = zmq.Context()
        req = ctx.socket(zmq.REQ)
        # Blocks until connection is found
        req.connect("tcp://localhost:{port}".format(port=req_port))
        req.send("give me the grid")
        grid = req.recv_pyobj()
        logger.info("Grid  received")
        self.data = {}
        self.grid = grid
        # start listening to data in a background thread
        make_listenener(ctx, sub_port, self.data)

        # Start the model listener

    def renderArea(self, width, height, srs, xmin, ymin, xmax, ymax, zoom):
        """
        Render the requested variable
        """

        logger.info("width: {width}, height: {height}, srs: {srs}, xmin: {xmin}, ymin: {ymin}, xmax: {xmax}, ymax: {ymax}, zoom: {zoom}".format(**locals()) )
        logger.info("I have variables {}".format(self.data.keys()))
        memdriver = osgeo.gdal.GetDriverByName('MEM')
        tiffdriver = osgeo.gdal.GetDriverByName('GTiff')

        # name does not do anything
        grid =  self.grid
        quad_grid = grid['quad_grid']
        quad_transform= (grid['x0p'],  # xmin
                         grid['dxp'], # xmax
                         0,            # for rotation
                         grid['y0p'],
                         0,
                         grid['dyp'])


        src_ds = memdriver.Create(self.layer.name(),
                                  quad_grid.shape[1],
                                  quad_grid.shape[0], 1,
                                  eType=osgeo.gdal.GDT_Int32)

        if src_ds.GetGCPs():
            src_ds.SetProjection(src_ds.GetGCPProjection())

        src_ds.SetGeoTransform(quad_transform)

        src_srs = osgeo.osr.SpatialReference()
        epsg = 22234
        src_srs.ImportFromEPSG(epsg)
        src_ds.SetProjection(src_srs.ExportToWkt())

        band = src_ds.GetRasterBand(1)
        band.WriteArray(quad_grid)

        # Prepare output gdal datasource -----------------------------------

        area_ds = tiffdriver.Create('/vsimem/output.tiff', width, height, 1, eType = osgeo.gdal.GDT_Int32)

        if area_ds is None:
            raise Exception('uh oh.')


        merc = osgeo.osr.SpatialReference()
        merc.ImportFromProj4(srs)

        area_ds.SetProjection(merc.ExportToWkt())

        w = xmax - xmin
        h = ymax - ymin
        gtx = [xmin, w/width, 0, ymin, 0, h/height]
        area_ds.SetGeoTransform(gtx)
        band = area_ds.GetRasterBand(1)
        band.SetNoDataValue(-9999)
        band.WriteArray(np.zeros((band.YSize, band.XSize))-9999)


        # Adjust resampling method -----------------------------------------

        resample = osgeo.gdal.GRA_NearestNeighbour

        # Create rendered area ---------------------------------------------

        osgeo.gdal.ReprojectImage(src_ds, area_ds, src_ds.GetProjection(), area_ds.GetProjection(), resample, 1000000, 0.1 )
        data = area_ds.GetRasterBand(1).ReadAsArray()
        data = np.ma.masked_equal(data, -9999)
        tiffdriver.Delete('/vsimem/output.tiff')

        print data.dtype, data.shape
        area = Image.fromstring('RGBA', (width, height), data)

        return area
