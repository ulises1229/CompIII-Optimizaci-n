    from PIL import Image
    import numpy as np
    import ubelt as ub

    # Grab some test data
    fpath = ub.grabdata('http://www.topcoder.com/contest/problem/UrbanMapper3D/JAX_Tile_043_DTM.tif')

    # Open the tiff image
    pil_img = Image.open(fpath)

    # Map PIL mode to numpy dtype (note this may need to be extended)
    dtype = {'F': np.float32, 'L': np.uint8}[pil_img.mode]

    # Load the data into a flat numpy array and reshape
    np_img = np.array(pil_img.getdata(), dtype=dtype)
    w, h = pil_img.size