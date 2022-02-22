import numpy as np


def get_bottom_padding(vp, config, src_x, src_z):
    model_config = config['model']
    dz = model_config['spacing'][1]  # vp is in km/s so need to convert

    # Take a depth slice
    slc = vp[src_x, src_z:]

    # Get the two-way traveltime for a vertical ray
    t = 2*np.sum(slc*(dz))  # seconds
    tn = config['solver']['tn']

    # Get the padding if necessary
    if t <= tn:
        # dz is converted from from m to km
        bottom_pad = (1./2.)*(tn-t)*slc[-1]/(dz/1000)
        bottom_pad = int(np.round(bottom_pad)) + 1
    else:
        bottom_pad = 0
    return bottom_pad


def get_side_padding(vp, config, src_x, src_z):
    '''
    Designed to remove the direct wave reflection propogating at 
    the surface (typically water) velocity, but may not remove edge 
    reflections e.g. if there is a fast medium in the shallow subsurface 
    such as a large salt block.
    '''
    model_config = config['model']
    dx = model_config['spacing'][0]  # vp is in km/s so need to convert

    slc_right = vp[src_x:, src_z]
    slc_left = vp[:src_x, src_z]
    tn = config['solver']['tn']

    # Get the right and left side traveltime
    right_tt = np.sum(slc_right*(dx/1000))
    left_tt = np.sum(slc_left*(dx/1000))

    # Get padding amounts
    if right_tt <= tn:
        pad_right = (1./2.)*(tn-right_tt)*slc_right[-1]/(dx/1000)
        pad_right = int(np.round(pad_right)) + 1
    else:
        pad_right = 0

    if left_tt <= tn:
        pad_left = (1./2.)*(tn-left_tt)*slc_left[-1]/(dx/1000)
        pad_left = int(np.round(pad_left)) + 1
    else:
        pad_left = 0

    return pad_left, pad_right


def pad_based_on_source(vp, config, src):
    '''
    Pad the model to avoid edge reflections
    '''

    dx = config['model']['spacing'][0]
    dz = config['model']['spacing'][0]
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