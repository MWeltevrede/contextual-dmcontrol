import os
import numpy as np
from dm_control.suite import common
import dm_control
from dm_control.utils import io as resources
import xmltodict

_SUITE_DIR = os.path.dirname(os.path.dirname(__file__))
_FILENAMES = [
    "./common/materials.xml",
    "./common/skybox.xml",
    "./common/visual.xml",
]

def get_model_and_assets_from_setting_kwargs(model_fname, task_name, setting_kwargs=None):
    """"Returns a tuple containing the model XML string and a dict of assets."""
    assets = {filename: resources.GetResource(os.path.join(_SUITE_DIR, filename))
          for filename in _FILENAMES}
    
    if model_fname == "manipulator.xml":
        use_peg = True
        insert = False
        if "ball" in task_name:
            use_peg = False
        if "insert" in task_name:
            insert = True
        model_xml, _ = dm_control.suite.manipulator.make_model(use_peg, insert)
    elif model_fname == "stacker.xml":
        num_boxes = 2
        if "4" in task_name:
            num_boxes = 4
        model_xml, _ = dm_control.suite.stacker.make_model(num_boxes)
    else:
        model_xml = common.read_model(model_fname)

    if setting_kwargs is None:
        return model_xml, assets

    # Convert XML to dicts
    model = xmltodict.parse(model_xml)
    materials = xmltodict.parse(assets['./common/materials.xml'])
    skybox = xmltodict.parse(assets['./common/skybox.xml'])

    # Edit grid floor
    if 'grid_rgb1' in setting_kwargs:
        assert isinstance(setting_kwargs['grid_rgb1'], (list, tuple, np.ndarray))
        assert materials['mujoco']['asset']['texture']['@name'] == 'grid'
        materials['mujoco']['asset']['texture']['@rgb1'] = \
            f'{setting_kwargs["grid_rgb1"][0]} {setting_kwargs["grid_rgb1"][1]} {setting_kwargs["grid_rgb1"][2]}'
    if 'grid_rgb2' in setting_kwargs:
        assert isinstance(setting_kwargs['grid_rgb2'], (list, tuple, np.ndarray))
        assert materials['mujoco']['asset']['texture']['@name'] == 'grid'
        materials['mujoco']['asset']['texture']['@rgb2'] = \
            f'{setting_kwargs["grid_rgb2"][0]} {setting_kwargs["grid_rgb2"][1]} {setting_kwargs["grid_rgb2"][2]}'
    if 'grid_markrgb' in setting_kwargs:
        assert isinstance(setting_kwargs['grid_markrgb'], (list, tuple, np.ndarray))
        assert materials['mujoco']['asset']['texture']['@name'] == 'grid'
        materials['mujoco']['asset']['texture']['@markrgb'] = \
            f'{setting_kwargs["grid_markrgb"][0]} {setting_kwargs["grid_markrgb"][1]} {setting_kwargs["grid_markrgb"][2]}'
    if 'grid_texrepeat' in setting_kwargs:
        assert isinstance(setting_kwargs['grid_texrepeat'], (list, tuple, np.ndarray))
        assert materials['mujoco']['asset']['texture']['@name'] == 'grid'
        materials['mujoco']['asset']['material'][0]['@texrepeat'] = \
            f'{setting_kwargs["grid_texrepeat"][0]} {setting_kwargs["grid_texrepeat"][1]}'

    # Edit self
    if 'self_rgb' in setting_kwargs:
        assert isinstance(setting_kwargs['self_rgb'], (list, tuple, np.ndarray))
        assert materials['mujoco']['asset']['material'][1]['@name'] == 'self'
        materials['mujoco']['asset']['material'][1]['@rgba'] = \
            f'{setting_kwargs["self_rgb"][0]} {setting_kwargs["self_rgb"][1]} {setting_kwargs["self_rgb"][2]} 1'

    # Edit skybox
    if 'skybox_rgb' in setting_kwargs:
        assert isinstance(setting_kwargs['skybox_rgb'], (list, tuple, np.ndarray))
        assert skybox['mujoco']['asset']['texture']['@name'] == 'skybox'
        skybox['mujoco']['asset']['texture']['@rgb1'] = \
            f'{setting_kwargs["skybox_rgb"][0]} {setting_kwargs["skybox_rgb"][1]} {setting_kwargs["skybox_rgb"][2]}'
    if 'skybox_rgb2' in setting_kwargs:
        assert isinstance(setting_kwargs['skybox_rgb2'], (list, tuple, np.ndarray))
        assert skybox['mujoco']['asset']['texture']['@name'] == 'skybox'
        skybox['mujoco']['asset']['texture']['@rgb2'] = \
            f'{setting_kwargs["skybox_rgb2"][0]} {setting_kwargs["skybox_rgb2"][1]} {setting_kwargs["skybox_rgb2"][2]}'
    if 'skybox_markrgb' in setting_kwargs:
        assert isinstance(setting_kwargs['skybox_markrgb'], (list, tuple, np.ndarray))
        assert skybox['mujoco']['asset']['texture']['@name'] == 'skybox'
        skybox['mujoco']['asset']['texture']['@markrgb'] = \
            f'{setting_kwargs["skybox_markrgb"][0]} {setting_kwargs["skybox_markrgb"][1]} {setting_kwargs["skybox_markrgb"][2]}'

    # Convert back to XML
    model_xml = xmltodict.unparse(model)
    assets['./common/materials.xml'] = xmltodict.unparse(materials)
    assets['./common/skybox.xml'] = xmltodict.unparse(skybox)

    return model_xml, assets
