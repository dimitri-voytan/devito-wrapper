import numpy as np


def get_bottom_padding(vp, config, src_x, src_z):
    model_config = config['model']
    dz = model_config['spacing'][1]/1000  # vp is in km/s so need to convert

    # Take a depth slice
    slc = vp[src_x, src_z:]

    # Get the two-way traveltime for a vertical ray
    tz = 2*np.sum(dz*(1/slc))  # seconds
    tn = config['solver']['tn']

    # Get the padding if necessary
    if tz <= tn:
        # dz is converted from from m to km
        bottom_pad = (0.50)*(tn-tz)*(slc[-1])/(dz)
        bottom_pad = int(np.round(bottom_pad)) + 1
    else:
        bottom_pad = 0
    return bottom_pad


def get_side_padding(vp, config, src_x, src_z):
    '''
    Designed to remove the direct wave reflection propagating at 
    the surface (typically water) velocity, but may not remove edge 
    reflections e.g. if there is a fast medium in the shallow subsurface 
    such as a large salt block.
    '''
    model_config = config['model']
    dx = model_config['spacing'][0]/1000.0  # vp is in km/s so need to convert

    slc_right = vp[src_x:, src_z]
    slc_left = vp[:src_x, src_z]
    tn = config['solver']['tn']

    # Get the traveltime to the right and left side of the model.
    right_tt = np.sum((1/slc_right)*(dx))
    left_tt = np.sum((1/slc_left)*(dx))

    # Get padding amounts
    # For the direct wave we only need 1/2 the residual travel time in
    # padding. Using 0.65 gives a bit more room for error
    if right_tt <= tn:
        pad_right = (0.65)*(tn-right_tt)*slc_right[-1]/(dx)
        pad_right = int(np.round(pad_right)) + 1  # add an extra point for roundoff error
    else:
        pad_right = 0

    if left_tt <= tn:
        pad_left = (0.65)*(tn-left_tt)*slc_left[-1]/(dx)
        pad_left = int(np.round(pad_left)) + 1  # add an extra point for roundoff error
    else:
        pad_left = 0

    return pad_left, pad_right


def pad_based_on_source(vp, config, src):
    '''
    Pad the model to avoid edge reflections
    '''

    dx = config['model']['spacing'][0]
    dz = config['model']['spacing'][0]

    # Convert source coordinates to indices
    src_x = int(src[0]/dx)
    src_z = int(src[1]/dz)

    # check edge cases
    src_x, src_z = check_edge_cases(vp, src_x, src_z)
    pad_dict = {}

    pad_dict['bottom'] = get_bottom_padding(vp, config, src_x, src_z)
    pad_dict['top'] = 0
    pad_dict['left'], pad_dict['right'] = get_side_padding(vp, config,
                                                           src_x, src_z)

    return pad_dict


def check_edge_cases(vp, src_x, src_z):
    '''
    Source indices on the edge can't be used for slicing
    '''
    if src_x == 0:
        src_x += 1
    elif src_x == vp.shape[0]:
        src_x -= 1

    if src_z == 0:
        src_z += 1
    elif src_z == vp.shape[1]:
        src_z -= 1   
    return src_x, src_z
