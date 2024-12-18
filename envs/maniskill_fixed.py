from mani_skill2.utils.registration import register_env


############################################
# PegInsertionSide
############################################

from mani_skill2.envs.assembly.peg_insertion_side import PegInsertionSideEnv

@register_env("PegInsertionSide-v2", max_episode_steps=200)
class PegInsertionSideEnv_fixed(PegInsertionSideEnv):
    '''
    The original PegInsertionSideEnv has a bug in the has_peg_inserted function.
    '''

    def has_peg_inserted(self):
        # Only head position is used in fact
        peg_head_pose = self.peg.pose.transform(self.peg_head_offset)
        box_hole_pose = self.box_hole_pose
        peg_head_pos_at_hole = (box_hole_pose.inv() * peg_head_pose).p
        # x-axis is hole direction
        x_flag = (-0.015 <= peg_head_pos_at_hole[0]) and (0.075 >= peg_head_pos_at_hole[0]) # fix here
        y_flag = (
            -self.box_hole_radius <= peg_head_pos_at_hole[1] <= self.box_hole_radius
        )
        z_flag = (
            -self.box_hole_radius <= peg_head_pos_at_hole[2] <= self.box_hole_radius
        )
        return (x_flag and y_flag and z_flag), peg_head_pos_at_hole

############################################
# TurnFaucet
############################################

from mani_skill2.envs.misc.turn_faucet import (
    TurnFaucetEnv,
    OrderedDict,
    vectorize_pose,
)

@register_env("TurnFaucet-v2", max_episode_steps=200)
class TurnFaucetEnv_COTPC(TurnFaucetEnv):
    '''
    Adapted from CoTPC: https://github.com/zj1a/CoTPC/blob/main/maniskill2_patches/turn_faucet.py
    '''

    def _get_curr_target_link_pos(self):
        """
        Access the current pose of the target link (i.e., the handle of the faucet).
        """
        cmass_pose = self.target_link.pose * self.target_link.cmass_local_pose
        return cmass_pose.p

    def _get_obs_extra(self):
        obs = OrderedDict(
            tcp_pose=vectorize_pose(self.tcp.pose),
            target_angle_diff=self.target_angle_diff,
            target_joint_axis=self.target_joint_axis,
            target_link_pos=self.target_link_pos,
            curr_target_link_pos=self._get_curr_target_link_pos(),  # Added code.
        )
        if self._obs_mode in ["state", "state_dict"]:
            angle_dist = self.target_angle - self.current_angle
            obs["angle_dist"] = angle_dist
        return obs
    
    def __init__(
        self,
        *args,
        model_ids = (),
        **kwargs,
    ):
        # The base policy is trained on these objects: 5002 5021 5023 5028 5029 5045 5047 5051 5056 5063
        # The env for online interactinos include these objects: 5014 5037 5053 5062
        model_ids = ('5014', '5037', '5053', '5062')
        super().__init__(*args, model_ids=model_ids, **kwargs)

############################################
# PushChair
############################################

from mani_skill2.envs.ms1.push_chair import PushChairEnv
import sapien.core as sapien
import numpy as np

@register_env("PushChair-v2", max_episode_steps=200)
class PushChairEnv_COTPC(PushChairEnv):
    
    def __init__(
        self,
        *args,
        model_ids = (),
        **kwargs,
    ):
        # The base policy is trained on these objects: 3022 3027 3030 3070 3076
        # The env for online interactinos include these objects: 3003 3013 3020
        model_ids = ('3003', '3013', '3020')
        super().__init__(*args, model_ids=model_ids, **kwargs)

    ##################################################################
    # The following code is to fix a bug in MS1 envs (0.5.3)
    ##################################################################
    def reset(self, seed=None, options=None):
        self._prev_actor_pose_dict = {}
        return super().reset(seed, options)
    
    def check_actor_static(self, actor: sapien.Actor, max_v=None, max_ang_v=None):
        """Check whether the actor is static by finite difference.
        Note that the angular velocity is normalized by pi due to legacy issues.
        """

        from mani_skill2.utils.geometry import angle_distance

        pose = actor.get_pose()

        if self._elapsed_steps <= 1:
            flag_v = (max_v is None) or (np.linalg.norm(actor.get_velocity()) <= max_v)
            flag_ang_v = (max_ang_v is None) or (
                np.linalg.norm(actor.get_angular_velocity()) <= max_ang_v
            )
        else:
            prev_actor_pose, prev_step, prev_actor_static = self._prev_actor_pose_dict[actor.id]
            if prev_step == self._elapsed_steps:
                return prev_actor_static
            assert prev_step == self._elapsed_steps - 1, (prev_step, self._elapsed_steps)
            dt = 1.0 / self._control_freq
            flag_v = (max_v is None) or (
                np.linalg.norm(pose.p - prev_actor_pose.p) <= max_v * dt
            )
            flag_ang_v = (max_ang_v is None) or (
                angle_distance(prev_actor_pose, pose) <= max_ang_v * dt
            )

        # CAUTION: carefully deal with it for MPC
        actor_static = flag_v and flag_ang_v
        self._prev_actor_pose_dict[actor.id] = (pose, self._elapsed_steps, actor_static)
        return actor_static