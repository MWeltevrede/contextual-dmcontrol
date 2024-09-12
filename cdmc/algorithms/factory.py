from cdmc.algorithms.sac import SAC
from cdmc.algorithms.rad import RAD
from cdmc.algorithms.curl import CURL
from cdmc.algorithms.pad import PAD
from cdmc.algorithms.soda import SODA
from cdmc.algorithms.drq import DrQ
from cdmc.algorithms.svea import SVEA

algorithm = {
	'sac': SAC,
	'rad': RAD,
	'curl': CURL,
	'pad': PAD,
	'soda': SODA,
	'drq': DrQ,
	'svea': SVEA
}


def make_agent(obs_shape, action_shape, args):
	return algorithm[args.algorithm](obs_shape, action_shape, args)
