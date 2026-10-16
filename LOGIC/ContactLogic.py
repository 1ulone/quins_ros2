import numpy as np

SWING, STANCE, SEARCH = 0, 1, 2


class ContactEstimator:
    def __init__(
        self, 
        n_feet=4, 
        f_on=10.0,
        f_off=5.0,
        n_debounce=3,
        s_early=0.5,
        v_search=0.4,
        search_max=0.06,
        late_t=0.02,
    ):
        self.f_on = f_on
        self.f_off = f_off
        self.n_db = n_debounce
        self.s_early = s_early
        self.v_search = v_search
        self.search_max = search_max
        self.late_t = late_t

        self.contact = np.ones(n_feet, dtype=bool)
        self.count = np.zeros(n_feet, dtype=int)
        self.mode = np.full(n_feet, STANCE)
        self.prev_schedules = np.ones(n_feet, dtype=bool)
        self.search_t0 = np.zeros(n_feet)
        self.events = {"early": 0, "late": 0}

    def _debounce(self, fz):
        raw = np.where(self.contact, fz > self.f_off, fz > self.f_on)
        self.count = np.where(raw != self.contact, self.count + 1, 0)
        flip = self.count >= self.n_db
        self.contact[flip] = ~self.contact[flip]
        self.count[flip] = 0

    def update(self, t, fz, schedule, s):
        fz = np.asarray(fz, dtype=float)
        self._debounce(fz)
        # at the scheduled touchdown and while searching, the schedule already expects ground,
        # so one raw sample above f_on is enough (schedule as a prior, Bledt et al. 2018);
        # the debounce stays for early touchdowns in mid-swing, where a spike is more likely
        touch = self.contact | (fz > self.f_on)
        schedule = np.asarray(schedule) > 0.5
        for i in range(len(schedule)):
            lift = self.prev_schedules[i] and not schedule[i]
            land = schedule[i] and not self.prev_schedules[i]
            m = self.mode[i]
            if lift:
                if m == SEARCH:
                    self.events["late"] += 1            # the whole stance passed without finding ground
                m = SWING
            elif m == SWING:
                if land:
                    if touch[i]:
                        m = STANCE
                    else:
                        m = SEARCH
                        self.search_t0[i] = t
                elif self.contact[i] and s[i] >= self.s_early:
                    m = STANCE
                    self.events["early"] += 1
            elif m == SEARCH and touch[i]:
                m = STANCE
                if t - self.search_t0[i] > self.late_t:  # longer than the foot just settling in
                    self.events["late"] += 1
            self.mode[i] = m
        self.prev_schedules = schedule.copy()
        return self.mode == STANCE

    def search_drop(self, t):
        d = np.clip(self.v_search * (t - self.search_t0), 0.0, self.search_max)
        return np.where(self.mode == SEARCH, d, 0.0)
