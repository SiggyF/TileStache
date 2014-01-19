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

def make_quad_grid(grid):
    """create a quad grid based on the grid information"""
    # Create lookup index
    nodslice = slice(np.where(grid['nod_type'] == 2)[0].min(),
                     np.where(grid['nod_type'] == 2)[0].max())
    n = grid['nodn'][nodslice] # column lookup
    m = grid['nodm'][nodslice] # row lookup
    k = grid['nodk'][nodslice] # level lookup

    # pixel boundary lookup
    Y = grid['ip'][:,m-1,k-1] # rows
    X = grid['jp'][:,n-1,k-1] # columns

    minpx = 5 # is this fixed?

    # slices
    Xmin = (X[0,:] - 1)/minpx
    Xmax = (X[3,:] + 1)/minpx
    Ymin = (Y[0,:] - 1)/minpx
    Ymax = (Y[3,:] + 1)/minpx

    # total image size
    mmax = Xmax.max()
    nmax = Ymax.max()

    # quad lookup
    quad_grid = np.ma.empty((mmax,nmax), dtype='int32')
    quad_grid.mask = True

    slices = np.c_[Xmin, Xmax, Ymin, Ymax]
    indices = np.arange(k.shape[0])

    # fill single pixel quads separately
    quad_grid[slices[k==1,0], slices[k==1,2]] = indices[k==1]
    # fill the bigger quads
    for i, (xmin, xmax, ymin, ymax) in zip(indices[k>1], slices[k>1]):
        quad_grid[xmin:xmax, ymin:ymax] = i
    return quad_grid


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

        logger.info("Connecting to grid")
        zmqctx = zmq.Context()
        reqsock = zmqctx.socket(zmq.REQ)
        # Blocks until connection is found
        reqsock.connect("tcp://localhost:{port}".format(port=port+1))
        reqsock.send("give me the grid")
        grid = reqsock.recv_pyobj()
        logger.info("Grid  received")
        self.quad_grid = make_quad_grid(grid)
        self.data = {}

        # start listening to data in a background thread
        make_listenener(zmqctx, port, self.data)

        # Start the model listener

    def renderArea(self, width, height, srs, xmin, ymin, xmax, ymax, zoom):
        """
        Render the requested variable
        """

        logger.info("I have variables {}".format(self.data.keys()))
        memdriver = osgeo.gdal.GetDriverByName('MEM')
        tiffdriver = osgeo.gdal.GetDriverByName('GTiff')

        # name does not do anything
        quad_grid = self.quad_grid

        src_ds = memdriver.Create(self.layer.name(),
                                  quad_grid.shape[1],
                                  quad_grid.shape[0], 1,
                                  eType=osgeo.gdal.GDT_Int32)

        if src_ds.GetGCPs():
            src_ds.SetProjection(src_ds.GetGCPProjection())

        grayscale_src = (src_ds.RasterCount == 1)

        try:
            # Prepare output gdal datasource -----------------------------------

            area_ds = tiffdriver.Create('/vsimem/output', width, height, 3)

            if area_ds is None:
                raise Exception('uh oh.')


            merc = osgeo.osr.SpatialReference()
            merc.ImportFromProj4(srs)
            area_ds.SetProjection(merc.ExportToWkt())

            # note that 900913 points north and east
            x, y = xmin, ymax
            w, h = xmax - xmin, ymin - ymax

            gtx = [x, w/width, 0, y, 0, h/height]
            area_ds.SetGeoTransform(gtx)
            # Adjust resampling method -----------------------------------------

            resample = self.resample

            if resample == osgeo.gdal.GRA_CubicSpline:
                #
                # I've found through testing that when ReprojectImage is used
                # on two same-scaled datasources, GDAL will visibly darken the
                # output and the results look terrible. Switching resampling
                # from cubic spline to bicubic in these cases fixes the output.
                #
                xscale = area_ds.GetGeoTransform()[1] / src_ds.GetGeoTransform()[1]
                yscale = area_ds.GetGeoTransform()[5] / src_ds.GetGeoTransform()[5]
                diff = max(abs(xscale - 1), abs(yscale - 1))

                if diff < .001:
                    resample = osgeo.gdal.GRA_Cubic

            # Create rendered area ---------------------------------------------
            src_sref = osgeo.osr.SpatialReference()
            src_sref.ImportFromWkt(src_ds.GetProjection())

            osgeo.gdal.ReprojectImage(src_ds, area_ds, src_ds.GetProjection(), area_ds.GetProjection(), resample)

            channel = grayscale_src and (1, 1, 1) or (1, 2, 3)
            r, g, b = [area_ds.GetRasterBand(i).ReadRaster(0, 0, width, height) for i in channel]

            data = ''.join([''.join(pixel) for pixel in zip(r, g, b)])
            area = Image.fromstring('RGB', (width, height), data)

        finally:
            tiffdriver.Delete('/vsimem/output')

        return area
