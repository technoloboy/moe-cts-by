from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg, VecEnvStepReturn
from robot_lab.tasks.boying.manager.action_manager import ActionManagerGo2, ActionManagerGo2WithDelay

import torch


class BoyingEnv(ManagerBasedRLEnv):
    cfg: ManagerBasedRLEnvCfg

    def load_managers(self):
        super().load_managers()
        self.action_manager = ActionManagerGo2(self.cfg.actions, self)
        print("[BoyingEnv-INFO] Overriding action manager with ActionManagerGo2: ", self.action_manager)


class ActionDelayBoyingEnv(ManagerBasedRLEnv):
    cfg: ManagerBasedRLEnvCfg

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        print(
            "[ActionDelayBoyingEnv-WARNING] You are using ActionDelayBoyingEnv; "
            "make sure all ActionTerms support multiple calls to process_actions() "
            "within a single step()."
        )

    def load_managers(self):
        super().load_managers()
        self.action_manager = ActionManagerGo2WithDelay(self.cfg.actions, self)
        print("[ActionDelayBoyingEnv-INFO] Overriding action manager with ActionManagerGo2WithDelay: ", self.action_manager)

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        self.action_manager.update_action(action.to(self.device))

        actions_start_decimation = torch.randint(0, self.cfg.decimation + 1, (self.num_envs, 1), device=self.device)

        self.recorder_manager.record_pre_step()

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        for i in range(self.cfg.decimation):
            self._sim_step_counter += 1

            action_delay_masks = (i < actions_start_decimation)
            self.action_manager.process_action_with_delay(action_delay_masks)

            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.recorder_manager.record_pre_reset(reset_env_ids)
            self._reset_idx(reset_env_ids)
            self.scene.write_data_to_sim()
            self.sim.forward()
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()
            self.recorder_manager.record_post_reset(reset_env_ids)

        self.command_manager.compute(dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        self.obs_buf = self.observation_manager.compute(update_history=True)

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
