from config_loading import TeacherForcingConfig

class TeacherForcing:
    def __init__(self):
        self._config = TeacherForcingConfig()
        assert self._config.ratio_type == "fixed" or self._config.ratio_type == "decay"

    @property
    def enabled(self):
        return self._config.use_teacher_forcing

    def get_current_ratio(self, step):
        if self._config.ratio_type == "fixed":
            return self._config.fixed_ratio
        else:
            return self._get_decayed_ratio(step)

    def _get_decayed_ratio(self, step):
        if step < self._config.decay_start_step:
            return self._config.decay_start_ratio
        elif step >= self._config.decay_end_step:
            return self._config.decay_end_ratio
        else:
            # Linear decay
            rate = (step - self._config.decay_start_step) / (self._config.decay_end_step - self._config.decay_start_step)
            return rate * self._config.decay_end_ratio + (1.0 - rate) * self._config.decay_start_ratio
