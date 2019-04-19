from functools import partial
# from smac.env import MultiAgentEnv, StarCraft2Env
import sys
import os
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios

# def env_fn(env, **kwargs) -> MultiAgentEnv:
#     return env(**kwargs)

# def make_env(env_id, seed, rank, discrete_action, shaped_reward, 
#             num_agents, blind_agents, team_spirit, dist_threshold, arena_size):
#     def _thunk():
#         env = make_multiagent_env(env_id, discrete_action, shaped_reward, 
#                                 num_agents, blind_agents, team_spirit, dist_threshold, arena_size)
#         env.seed(seed + rank) # seed not implemented
#         return env
#     return _thunk

def env_fn(**kwargs):
    return make_multiagent_env(**kwargs)

def make_multiagent_env(env_id, discrete_action, shaped_reward, num_agents, 
                        blind_agents, team_spirit, dist_threshold, arena_size, seed):
    scenario = scenarios.load(env_id+".py").Scenario(num_agents=num_agents, blind_agents=blind_agents,
                                                     shaped_reward=shaped_reward, team_spirit=team_spirit,
                                                     dist_threshold=dist_threshold,arena_size=arena_size)
    world = scenario.make_world()

    env = MultiAgentEnv(world=world, 
                        reset_callback=scenario.reset_world, 
                        reward_callback=scenario.reward, 
                        observation_callback=scenario.observation,
                        info_callback=scenario.info if hasattr(scenario, 'info') else None,
                        discrete_action=discrete_action,
                        done_callback=scenario.done,
                        cam_range=arena_size
                        )
    env.seed(seed)
    
    return env

REGISTRY = {}
# REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["simple_spread"] = partial(env_fn, env_id='simple_spread',discrete_action=True,shaped_reward=True,
                            num_agents=6,blind_agents=False,team_spirit=0.5,dist_threshold=0.1,arena_size=1)

# if sys.platform == "linux":
#     os.environ.setdefault("SC2PATH",
#                           os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
