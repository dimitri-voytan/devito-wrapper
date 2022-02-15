import numpy as np
import tqdm
import matplotlib.pyplot as plt
import os
import yaml
from datetime import datetime, date
from argparse import ArgumentParser
from scipy.interpolate import interp1d

from devito import norm
from devito.logger import info
from devito import configuration

from examples.seismic.elastic import ElasticWaveSolver
from examples.seismic import demo_model, setup_geometry, seismic_args, AcquisitionGeometry
from examples.seismic.model import SeismicModel

def initialize_output(save_path):
    today = date.today()
    time = datetime.now()
    outdir = os.path.join(save_path, today.strftime("%b-%d-%Y"), time.strftime("%H:%M:%S"))
    os.makedirs(outdir)
    return outdir

def parse_config_to_kwargs(config):
    kwargs = {}
    for item in config.keys():
        for val in config[item]:
            kwargs[val] = config[item][val]
    return kwargs

def resample_gather(config, geometry, data):
    new_dt = config['solver']['resample_dt']
    t_final = config['solver']['tn']
    assert type(new_dt) == float

    #For aco, we resample to the tti sampling rate
    new_nt = int(t_final/new_dt)
    new_time_axis = np.linspace(0, t_final, new_nt)

    #allocate a new array for the resampled gather
    resample_data = np.zeros((new_nt, data.shape[-1]))

    for i in range(data.shape[-1]):
        interp = interp1d(geometry.time_axis.time_values, data[:,i])
        tmp = interp(new_time_axis)
        resample_data[:,i] = tmp

    return resample_data

def get_bottom_padding(vp, config):
    model_config = config['model']
    dx = model_config['spacing'][0]
    dz = model_config['spacing'][1]
    index = int(config['acquisition']['src_x']/dx)
    slc = vp[index, :]

    #Get the two-way traveltime for a vertical ray
    t = 2*np.sum(slc*(dz/1000)) #seconds
    tn = config['solver']['tn']

    if t <= tn:
        bottom_pad = (1./2.)*(tn-t)*slc[-1]/(dx/1000)
        bottom_pad = int(np.round(bottom_pad)) + 1
    else:
        bottom_pad = 0
    return bottom_pad
    
def pad_based_on_source(vp, config):
    '''Pad either the left or right side of the model so that the source is centered'''

    src_x = config['acquisition']['src_x']
    dx = config['model']['spacing'][0]
    pad_dict = {}
    pad_dict['bottom'] = get_bottom_padding(vp, config)
    pad_dict['top'] = 0
    model_width = dx*config['model']['shape'][0]
    half_model_width = model_width/2
    if src_x > half_model_width: #half the model width
        pad_dist = int((src_x - half_model_width)/dx)
        pad_dict['right'] = pad_dist
        pad_dict['left'] = 0
    else:
        pad_dist = int((half_model_width-src_x)/dx)
        pad_dict['left'] = pad_dist
        pad_dict['right'] = 0
    return pad_dict


def setup_model(config, **kwargs):

    model_config = config['model']

    vp_path = model_config['vp_path']
    vs_path = model_config['vs_path']
    rho_path =  model_config['rho_path']
    spacing = model_config['spacing']
    shape = model_config['shape'] if  model_config['shape'] is not None else vp.shape
    nbl = model_config['nbl']
    fs = model_config['fs']

    space_order = config['solver']['space_order']
    dtype = np.float32 if config['solver']['dtype']== 'float32' else np.float64
    
    #Load data and convert to km/s and kg/km^3
    vp = np.load(vp_path).astype(dtype)/1000.0
    vs = np.load(vs_path).astype(dtype)/1000.0
    rho = np.load(rho_path).astype(dtype)/1000.0

    padding = pad_based_on_source(vp, config)

    params = [vp, vs, rho]
    #Pad the params
    for i, param in enumerate(params):
        params[i] = np.pad(params[i], pad_width=((padding['left'],padding['right']),\
                                                (padding['top'],padding['bottom'])), mode='edge')
    vp, vs, rho = params

    #Correct origin based on npad if padding left (negative value)
    if padding['left'] > 0:
        x_origin = -padding['left']*spacing[0]
    else:
        x_origin = 0

    origin = tuple([x_origin, 0.])
    
    return SeismicModel(space_order=space_order, vp=vp, vs=vs, b=1/rho,
                            origin=origin, shape=shape, dtype=dtype, spacing=spacing,
                            nbl=nbl, **kwargs)

def setup_geometry(model, config):

    acq_config = config['acquisition']
    tn = config['solver']['tn']*1000
    f0 = config['source']['f0']/1000
    src_type = 'Ricker'
    src_coord = (acq_config['src_x'], acq_config['src_z'])
    rcv_coords = (acq_config['rec_x'], acq_config['rec_z'])

    src_coordinates = setup_src_coords(model, src_coord)
    rec_coordinates = setup_rec_coords(model, rcv_coords)

    geometry = AcquisitionGeometry(model, rec_coordinates, src_coordinates,
                                   t0=0.0, tn=tn, src_type=src_type, f0=f0)
    return geometry

def setup_src_coords(model, src_coord):
    src_coordinates = np.empty((1, model.dim))
    src_coordinates[0,0] = src_coord[0]
    src_coordinates[0,-1] = src_coord[-1]
    return src_coordinates

def setup_rec_coords(model, rcv_coords):
    recx = rcv_coords[0]
    recx = np.arange(recx[0], recx[1], recx[2])*model.spacing[0]

    rec_coordinates = np.empty((len(recx), model.dim))
    rec_coordinates[:, 0] = recx
    rec_coordinates[:, -1] = rcv_coords[-1]
    return rec_coordinates

def elastic_setup(config):

    model = setup_model(config)
    critical_dt = model.critical_dt
    
    # Source and receiver geometries
    geometry = setup_geometry(model, config)

    #solver
    space_order = config['solver']['space_order']

    # Create solver object to provide relevant operators
    solver = ElasticWaveSolver(model, geometry, space_order=space_order)
    return solver, critical_dt, geometry

def run(config):

    solver, critical_dt, geometry = elastic_setup(config)
    info("Applying Forward")

    rec1, rec2, v, tau, summary = solver.forward(autotune=config['solver']['autotune'])

    return rec1, rec2, v, tau, summary, geometry


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-c','--config', help='path to config.yaml file', default='./config.yaml')
    args = parser.parse_args()

    with open(args.config, 'rb') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    #Create output directory structure
    outdir=initialize_output(config['output']['outdir'])

    #save the config
    with open(os.path.join(outdir, "config.yaml"), "w") as output:
        yaml.dump(config, output, default_flow_style=False)

    #Run the solver
    rec1, rec2, v, tau, summary, geometry = run(config)

    #Resample gather to requested dt
    if config['solver']['resample_dt'] is not None:
        rec1 = resample_gather(config, geometry, rec1.data)
        rec2 = resample_gather(config, geometry, rec2.data)

    #save
    save_file = {'v_1':rec1,
                 'v_2':rec2,
                 'src':geometry.src_positions,
                 'rcv':geometry.rec_positions}

    f_name = config['output']['file_prefix']
    np.save(os.path.join(outdir, f_name), save_file)