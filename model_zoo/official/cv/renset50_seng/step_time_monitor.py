import time

from mindspore import log as logger
from mindspore.train.callback import Callback

class StepTimeMonitor(Callback):
    """
    Monitor the time in training.

    Args:
        data_size (int): Dataset size. Default: None.
    """

    def __init__(self, data_size=None):
        super(StepTimeMonitor, self).__init__()
        self.data_size = data_size

    def epoch_begin(self, run_context):
        self.epoch_time = time.time()

    def epoch_end(self, run_context):
        epoch_seconds = (time.time() - self.epoch_time) * 1000
        step_size = self.data_size
        cb_params = run_context.original_args()
        if hasattr(cb_params, "batch_num"):
            batch_num = cb_params.batch_num
            if isinstance(batch_num, int) and batch_num > 0:
                step_size = cb_params.batch_num

        if not isinstance(step_size, int) or step_size < 1:
            logger.error("data_size must be positive int.")
            return

        step_seconds = epoch_seconds / step_size
        print("epoch time: {:5.3f} ms, per step time: {:5.3f} ms".format(epoch_seconds, step_seconds), flush=True)

    def step_begin(self, run_context):
        self.step_time = time.time()

    def step_end(self, run_context):
        step_seconds = (time.time() - self.step_time) * 1000
        step_size = self.data_size
        cb_params = run_context.original_args()
        if hasattr(cb_params, "batch_num"):
            batch_num = cb_params.batch_num
            if isinstance(batch_num, int) and batch_num > 0:
                step_size = cb_params.batch_num

        if not isinstance(step_size, int) or step_size < 1:
            logger.error("data_size must be positive int.")
            return

        print("step time: {:5.3f} ms".format(step_seconds), flush=True)
