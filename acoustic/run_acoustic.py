import numpy as np
# import tqdm
import os
import pickle
import yaml
from datetime import datetime, date
from argparse import ArgumentParser
from distributed import Client, LocalCluster, wait

from devito.logger import info
from devito_wrapper.utils.pad_utils import pad_based_on_source
from devito_wrapper.utils.utils import resample_gather

from examples.seismic.acoustic import AcousticWaveSolver
from examples.seismic import AcquisitionGeometry
from examples.seismic.model import SeismicModel


class AcousticWrapper():
    def __init__(self, config):
        self.config = config
        self.outdir = self.initialize_output(self.config['output']['outdir'])
        self.dump_config()
        
    def initialize_output(self, save_path):
        today = date.today()
        time = datetime.now()
        outdir = os.path.join(save_path,
                              today.strftime("%b-%d-%Y"),
                              time.strftime("%H:%M:%S"))
        os.makedirs(outdir)
        return outdir

    def dump_config(self):
        # Save the config
        with open(os.path.join(self.outdir, "config.yaml"), "w") as output:
            yaml.dump(self.config, output, default_flow_style=False)

    def setup_model(self, src):

        model_config = self.config['model']

        vp_path = model_config['vp_path']
        spacing = model_config['spacing']
        shape = model_config['shape']
        nbl = model_config['nbl']
        fs = model_config['fs']

        space_order = self.config['solver']['space_order']
        if self.config['solver']['dtype'] == 'float32':
            dtype = np.float32
        else: 
            dtype = np.float64

        # Load data and convert to km/s and kg/km^3
        vp = np.load(vp_path).astype(dtype)/1000.0

        if model_config['shape'] is None:
            model_config['shape'] = vp.shape

        padding = pad_based_on_source(vp, self.config, src=src)

        params = [vp]

        # Pad the params
        for i, _ in enumerate(params):
            params[i] = np.pad(params[i],
                               pad_width=((padding['left'], padding['right']),
                               (padding['top'], padding['bottom'])), 
                               mode='edge')
        vp = params[0]

        # Correct origin based on npad if padding left (negative value)
        if padding['left'] > 0:
            x_origin = -padding['left']*spacing[0]
        else:
            x_origin = 0

        origin = tuple([x_origin, 0.0])

        # Correct shape
        shape = vp.shape
       
        return SeismicModel(space_order=space_order, vp=vp,
                            origin=origin, shape=shape, 
                            dtype=dtype, spacing=spacing, nbl=nbl, fs=fs)

    def setup_geometry(self, model, nav_i):

        tn = self.config['solver']['tn']*1000  # to ms
        f0 = self.config['source']['f0']/1000  # to kHz
        src_type = 'Ricker'
        src_coord = nav_i[0]
        rcv_coords = nav_i[1]

        geometry = AcquisitionGeometry(model, rcv_coords, src_coord,
                                       t0=0.0, tn=tn, src_type=src_type, f0=f0)
        return geometry

    def acoustic_setup(self, nav_i):

        model = self.setup_model(src=nav_i[0])

        # Source and receiver geometries
        geometry = self.setup_geometry(model, nav_i)

        # solver
        space_order = self.config['solver']['space_order']

        # Create solver object to provide relevant operators
        solver = AcousticWaveSolver(model, geometry, space_order=space_order)
        return solver, geometry

    def single_shot(self, nav_i, shot_num):

        solver, geometry = self.acoustic_setup(nav_i)
        info("Applying Forward")

        out = solver.forward(autotune=self.config['solver']['autotune']) 
        self.dump_shot(out, geometry, shot_num)

    def dump_shot(self, out, geometry, shot_num):
        '''
        TODO: Interpret out and write pressure/displacement if requested
        '''
        # Unpack output
        rec, u, summary = out
        rec = rec.data

        # Resample gather to requested dt
        if self.config['solver']['resample_dt'] is not None:
            rec = resample_gather(self.config, geometry, rec)

        # Save the shot
        save_file = {'p': rec,
                     'src': geometry.src_positions,
                     'rcv': geometry.rec_positions}

        shot_num = str(shot_num)

        f_name = self.config['output']['file_prefix']
        np.save(os.path.join(self.outdir, f_name + shot_num), save_file)

    def load_nav(self):
        nav_file = self.config['acquisition']['nav_file']

        with open(nav_file, 'rb') as f:
            nav = pickle.load(f)

        return nav

    def run(self):

        par_config = self.config['distributed']

        # determine context
        if par_config['use_dask']:
            n_workers = par_config['n_workers']
            cluster = LocalCluster(n_workers=n_workers, death_timeout=600)
            c = Client(cluster)

        if par_config['use_mpi']:
            raise NotImplementedError('MPI for domain decomposition not yet implemented')

        # load the nav file
        nav = self.load_nav()

        # In parallel, each worker processes a shot
        if par_config['use_dask']:
            futures = c.map(self.single_shot, nav, np.arange(len(nav)))
            wait(futures)
        else:
            for i, nav_i in enumerate(nav):
                self.single_shot(nav_i=nav_i, shot_num=i)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-c', '--config', help='path to config.yaml file', 
                        default='./config.yaml')
    args = parser.parse_args()

    with open(args.config, 'rb') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    runner = AcousticWrapper(config)

    # Run the solver
    runner.run()
