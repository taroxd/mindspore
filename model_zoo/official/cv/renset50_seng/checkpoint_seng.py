import os
import time
import mindspore as ms
from mindspore.train.serialization import save_checkpoint, _save_graph, _get_merged_param_data
from mindspore.parallel._ps_context import _is_role_pserver, _get_ps_mode_rank
from mindspore.train.callback._callback import set_cur_net

def _should_save(name):
    if 'matrix_' in name:
        return False
    if 'sample_index' in name:
        return False
    if 'damped_UUt' in name:
        return False
    if 'input_tensor' in name:
        return False
    if 'layer_seng_type' in name:
        return False
    if 'damping_value' in name:
        return False
    return True

class CheckPointSENG(ms.train.callback.ModelCheckpoint):
    """Lightweight checkpoint only for evaluating."""

    # override
    def _save_ckpt(self, cb_params, force_to_save=False):
        """Save checkpoint files."""
        if cb_params.cur_step_num == self._last_triggered_step:
            return

        save_ckpt = self._check_save_ckpt(cb_params, force_to_save)
        step_num_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

        if save_ckpt:
            cur_ckpoint_file = self._prefix + "-" + str(cb_params.cur_epoch_num) + "_" \
                               + str(step_num_in_epoch) + ".ckpt"
            if _is_role_pserver():
                cur_ckpoint_file = "PServer_" + str(_get_ps_mode_rank()) + "_" + cur_ckpoint_file
            # update checkpoint file list.
            self._manager.update_ckpoint_filelist(self._directory, self._prefix)
            # keep checkpoint files number equal max number.
            if self._config.keep_checkpoint_max and 0 < self._config.keep_checkpoint_max <= self._manager.ckpoint_num:
                self._manager.remove_oldest_ckpoint_file()
            elif self._config.keep_checkpoint_per_n_minutes and self._config.keep_checkpoint_per_n_minutes > 0:
                self._cur_time_for_keep = time.time()
                if (self._cur_time_for_keep - self._last_time_for_keep) \
                        < self._config.keep_checkpoint_per_n_minutes * 60:
                    self._manager.keep_one_ckpoint_per_minutes(self._config.keep_checkpoint_per_n_minutes,
                                                               self._cur_time_for_keep)

            # generate the new checkpoint file and rename it.
            global _save_dir
            _save_dir = self._directory
            cur_file = os.path.join(self._directory, cur_ckpoint_file)
            self._last_time_for_keep = time.time()
            self._last_triggered_step = cb_params.cur_step_num

            if ms.context.get_context("enable_ge"):
                set_cur_net(cb_params.train_network)
                cb_params.train_network.exec_checkpoint_graph()

            save_obj = cb_params.train_network
            save_obj.init_parameters_data()
            param_dict = {}
            for _, param in save_obj.parameters_and_names():
                if _should_save(param.name):
                    param_dict[param.name] = param
            param_list = []
            for (key, value) in param_dict.items():
                each_param = {"name": key}
                param_data = ms.Tensor(value.data)

                # in automatic model parallel scenario, some parameters were spliteds to all the devices,
                # which should be combined before saving
                if self._config.integrated_save and key in save_obj.parameter_layout_dict:
                    param_data = _get_merged_param_data(save_obj, key, param_data)

                each_param["data"] = param_data
                param_list.append(each_param)
            save_obj = param_list

            save_checkpoint(save_obj, cur_file, self._config.integrated_save,
                            self._config.async_save)

            self._latest_ckpt_file_name = cur_file
